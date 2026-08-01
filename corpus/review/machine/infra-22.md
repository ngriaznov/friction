This PR adds a `.github/workflows/build-and-push.yml` workflow that builds the Docker image on push to `main` and pushes it to ECR, authenticating via `aws-actions/configure-aws-credentials` with OIDC federation rather than long-lived access keys. Good to see OIDC used here instead of storing an `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` pair as repo secrets — that's the right default for GitHub-Actions-to-AWS auth at this point, and the `permissions: id-token: write` block at the job level is correctly scoped rather than set workflow-wide.

The `role-to-assume` ARN is read from a repo variable (`vars.AWS_ECR_ROLE_ARN`) rather than hardcoded inline, which is a nice touch for reusability if this workflow file ever gets templated across repos.

A couple of things worth tightening:

The `docker/build-push-action@v5` step tags the image with `${{ github.sha }}` only — no `latest` tag and no shorter semantic tag. That's actually a defensible choice (immutable SHA tags avoid the ambiguity problems I'd normally flag with `:latest`), but if any downstream tooling or manual `docker pull` workflow expects a `latest` tag to exist for convenience, this is going to surprise people. Worth confirming that's intentional and, if so, maybe leave a comment in the workflow noting the deliberate omission so nobody adds `latest` back in without thinking about why it wasn't there.

No `cache-from`/`cache-to` configured on the build-push action. Given this looks like a moderately-layered Dockerfile (multi-stage per the referenced Dockerfile diff in this same PR), builds are going to be slower than they need to be without registry-based layer caching. `cache-from: type=gha` / `cache-to: type=gha,mode=max` is a one-line addition that typically cuts rebuild time substantially when only the app layer changed.

The ECR repository name is hardcoded as a workflow-level `env: ECR_REPOSITORY: my-service` — fine for a single-service repo, but worth double-checking it matches exactly what's provisioned in the Terraform for this service, since a mismatch here would fail at push time with a not-particularly-obvious "repository does not exist" error rather than something that points back at a naming mismatch.

Minor: no `docker scout` or Trivy image scan step before the push. Not necessarily a blocker for this PR specifically if scanning is planned as a separate follow-up, but worth flagging since this workflow is the natural place to gate a push on a scan result if that's part of the security requirements here.

Approving — OIDC auth and SHA-based tagging are the two things I most wanted to see done right, and both are handled correctly. The caching addition would be a nice quick follow-up but isn't blocking.
