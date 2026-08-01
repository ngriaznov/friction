Reasonably clean module for what it does. A couple of quick notes.

You're hardcoding the CIDR block `10.0.0.0/16` as a default inside `variables.tf` rather than requiring callers to pass it explicitly. That's convenient for a single-VPC setup, but the moment someone needs a second VPC peered to the first, they'll hit an overlap because nobody realized there was a default. I'd drop the default and make `vpc_cidr` a required variable — forcing the caller to think about it once is cheaper than debugging a peering conflict later.

The subnet count is derived from `length(var.availability_zones)`, which is fine, but you're using `count` instead of `for_each` on the `aws_subnet` resource. With `count`, removing an AZ from the middle of the list shifts every subsequent subnet's index and Terraform will plan to destroy and recreate resources that didn't actually change. Since subnets carry route table associations and NAT gateway attachments, that's a bigger blast radius than it needs to be. Switching to `for_each` keyed on the AZ name would make the plan output much more predictable when the AZ list changes.

You've got `enable_dns_hostnames = true` but no `enable_dns_support` set explicitly — it defaults to true in AWS but I'd set it explicitly in the resource block anyway since this module is meant to be reused; explicit is better than relying on provider defaults that could change.

Nothing here rises to "must fix before merge" — the `for_each` point is the only one I'd call a real design issue, and even that's low urgency unless you expect the AZ list to churn. Approve with the suggestion to swap `count` for `for_each` before this module gets more consumers depending on it.
