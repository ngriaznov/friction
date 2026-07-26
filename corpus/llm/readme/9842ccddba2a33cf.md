# tf-microvpc

Terraform modules for a deliberately small VPC on Bitwing Cloud: one network, one to three subnets in a single zone, and an optional NAT instance instead of a managed NAT gateway.

This exists because most VPC modules are written for production networks — three availability zones, redundant NAT, flow logs into a log bucket, transit gateway attachment points — and then people copy them into a side project and spend eighty dollars a month on network plumbing for an application that serves four hundred requests a day. `tf-microvpc` makes the opposite tradeoff. It is single-zone by design. If the zone goes down, your project is down. For a hobby API, a personal Grafana, or a staging environment that nobody pages about, that is usually the correct call.

If you need multi-AZ, do not use this module. It will not grow into one gracefully and it is not trying to.

## Requirements

| Name | Version |
| --- | --- |
| Terraform | >= 1.5.0 |
| `bitwing` provider | >= 0.9.0, < 2.0.0 |

## Provider configuration

The module does not configure a provider — it inherits the one from your root configuration, so you control credentials and region.

```hcl
terraform {
  required_providers {
    bitwing = {
      source  = "bitwing-cloud/bitwing"
      version = "~> 0.9"
    }
  }
}

provider "bitwing" {
  region = "eu-fra-1"
  # token read from BITWING_API_TOKEN
}
```

Do not put your API token in the provider block. The provider reads `BITWING_API_TOKEN` from the environment, and that is the only method we test against.

## Usage

```hcl
module "vpc" {
  source  = "github.com/hnwll/tf-microvpc//modules/vpc?ref=v0.6.1"

  name       = "sidecar"
  region     = "eu-fra-1"
  zone       = "eu-fra-1b"
  cidr_block = "10.42.0.0/20"

  subnet_count = 2
  enable_nat   = true

  tags = {
    project = "sidecar"
    owner   = "hnwll"
  }
}
```

That produces:

- one VPC on `10.42.0.0/20`
- a public subnet `10.42.0.0/24` with a route to the internet gateway
- a private subnet `10.42.1.0/24` routed through the NAT instance
- a NAT instance (`s1.nano`) in the public subnet with source/destination checking disabled
- security groups for each subnet tier

Subnets are carved from the supplied CIDR as sequential `/24`s starting at the base address. With `subnet_count = 1` you get only the public subnet and `enable_nat` is ignored, since there is nothing behind the NAT to serve.

Pin the module with `?ref=` to a tag. The `main` branch is not a release channel and has broken people before.

## Inputs

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `string` | — | Name prefix for all created resources. Required. |
| `region` | `string` | — | Bitwing region. Must match the provider's region. Required. |
| `zone` | `string` | — | Zone within the region. All subnets land here. Required. |
| `cidr_block` | `string` | `"10.0.0.0/20"` | Address space for the VPC. Must be at least a `/22`. |
| `subnet_count` | `number` | `2` | 1, 2, or 3. Index 0 is public; 1 and 2 are private. |
| `enable_nat` | `bool` | `true` | Create a NAT instance for the private subnets. Ignored when `subnet_count = 1`. |
| `nat_instance_type` | `string` | `"s1.nano"` | Instance size for the NAT. `s1.nano` handles roughly 60 Mbit/s sustained. |
| `nat_ssh_key_id` | `string` | `null` | Optional SSH key to attach to the NAT instance for debugging. |
| `allowed_ssh_cidrs` | `list(string)` | `[]` | Sources permitted to reach port 22 on the public security group. Empty means no SSH ingress at all. |
| `enable_flow_logs` | `bool` | `false` | Ship flow logs to a bucket. Adds cost; see below. |
| `flow_log_bucket` | `string` | `null` | Required when `enable_flow_logs` is true. |
| `tags` | `map(string)` | `{}` | Applied to every resource that supports tags. |

`subnet_count` is validated in the module. Values outside 1–3 fail at plan time rather than producing a confusing apply error.

## Outputs

| Name | Description |
| --- | --- |
| `vpc_id` | ID of the created VPC. |
| `public_subnet_id` | ID of the public subnet. |
| `private_subnet_ids` | List of private subnet IDs, empty when `subnet_count = 1`. |
| `public_security_group_id` | Security group intended for internet-facing instances. |
| `private_security_group_id` | Security group for instances behind the NAT. |
| `nat_instance_id` | NAT instance ID, or `null` when NAT is disabled. |
| `nat_public_ip` | The address your private-subnet egress appears to come from. Useful for allowlisting with third-party APIs. |
| `route_table_ids` | Map of `{ public = ..., private = ... }` for attaching extra routes downstream. |

Typical downstream use:

```hcl
resource "bitwing_instance" "app" {
  subnet_id          = module.vpc.private_subnet_ids[0]
  security_group_ids = [module.vpc.private_security_group_id]
  # ...
}
```

## What it costs

Rough monthly figures for the default configuration (`subnet_count = 2`, `enable_nat = true`, `s1.nano` NAT) in `eu-fra-1`, at list price, as of the v0.6 release:

| Item | Monthly |
| --- | --- |
| VPC, subnets, route tables, internet gateway | $0.00 |
| NAT instance (`s1.nano`, 730 hours) | $3.65 |
| Elastic IP attached to the NAT | $0.00 (Bitwing charges only for unattached addresses) |
| Egress, first 100 GB | included |
| **Total** | **≈ $3.65** |

For comparison, Bitwing's managed NAT gateway is $0.048/hour plus $0.0045/GB processed, which is about $35/month before any traffic. That gap is the entire reason this module exists.

Two things will change the number:

- **Egress above 100 GB** is billed at $0.009/GB. A project pushing 500 GB adds about $3.60.
- **Flow logs** cost roughly $0.50/GB ingested. On a quiet network that is well under a dollar, but it is not free, which is why `enable_flow_logs` defaults to `false`.

Setting `enable_nat = false` brings the total to zero. You then have private subnets with no outbound path, which is fine if the instances there only receive traffic and never call out — but package installs and OS updates will hang, so plan for a bastion or a temporary NAT during provisioning.

## Known limitations

- **Single zone.** Stated once more because people miss it. There is no `az_count` variable and there will not be one.
- **The NAT instance is a single point of failure.** No autoscaling group, no health-check-driven replacement. If it dies, `terraform apply` recreates it in about ninety seconds, and you have to notice first.
- **No IPv6.** The Bitwing provider's IPv6 support was still marked experimental when this was written.
- **Changing `cidr_block` destroys the VPC** and everything Terraform knows lives in it. Treat it as immutable after the first apply.

## Contributing

Issues and pull requests welcome. Run `terraform fmt -recursive` and `tflint` before opening one. New variables need a reason beyond "another cloud has this" — scope creep here turns the module into the thing it was written to avoid.

MIT licensed.
