This PR stands up the `modules/static-site` Terraform module (S3 bucket + CloudFront distribution + Route53 record) for hosting the marketing site, and overall the structure is good — separating the bucket policy, the CloudFront distribution, and the DNS record into their own `.tf` files within the module keeps things readable.

A few points before this merges:

The S3 bucket resource still has `acl = "public-read"` set directly on `aws_s3_bucket_acl.this`, alongside a `aws_cloudfront_origin_access_control` block that's also being created. If you're using OAC (which is the current recommended approach, replacing the older OAI), the bucket itself should not be public at all — CloudFront should be the only thing with read access via the bucket policy's `Service: cloudfront.amazonaws.com` principal condition. Having both `public-read` ACL and OAC configured means the bucket is reachable directly via its S3 URL, bypassing CloudFront (and any caching, WAF rules, or custom error pages you've configured there) entirely. Drop the public ACL and rely solely on the OAC-scoped bucket policy.

`aws_cloudfront_distribution.this` doesn't set `viewer_certificate` to use a specific ACM certificate — it looks like it's falling through to the default CloudFront certificate, which won't support the custom domain referenced in the Route53 record you're also creating in this PR. Double check `viewer_certificate.acm_certificate_arn` is wired to a cert in `us-east-1` (CloudFront requires ACM certs in that region specifically regardless of where the rest of your infra lives) — if that's missing, `terraform apply` may succeed but visiting the custom domain over HTTPS will fail with a certificate mismatch.

Smaller items: no `default_root_object = "index.html"` set on the distribution, which means requests to `/` will 403 rather than serving your index page — worth confirming that's intentional if you're relying on a Lambda@Edge or CloudFront Function for routing instead, otherwise add it. Also the module doesn't expose `distribution_id` as an output, which you'll want for cache invalidation in your deploy pipeline later.

Good use of `for_each` over a `var.error_responses` map for the custom error page configuration rather than hardcoding 403/404 blocks — that's a nice touch for reusability across sites with different error-handling needs.

Requesting changes on the public ACL / OAC conflict and the certificate region — those two would produce a working `terraform apply` that doesn't actually behave the way this PR intends.
