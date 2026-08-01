This adds the Helm chart for the new `notifications` service under `charts/notifications/`, following the same layout as the existing `charts/billing` chart. Good to see the team standardizing on one chart skeleton rather than each service inventing its own structure.

Things that look right: the `Chart.yaml` version bump process matches the existing convention, `values.yaml` exposes replica count, image tag, and resource requests/limits as overridable top-level keys, and the `templates/hpa.yaml` is gated behind `{{- if .Values.autoscaling.enabled }}` so services that don't need autoscaling don't get a dangling HPA resource with no metrics backing it.

A few things I'd want changed:

The Deployment template sets `env` entries directly from a flat `values.yaml` map for what look like Kafka broker addresses and topic names, but there's also a `notifications-secrets` Secret referenced elsewhere in the PR (in the Kubernetes manifests directory, not this chart) that seems to hold the same kind of connection info for other services. Worth checking whether broker addresses belong in a ConfigMap consistently across charts rather than inline `env` values in `values.yaml` — right now if someone changes the Kafka bootstrap servers, they need to know to edit this specific chart's values rather than a shared config source. Not blocking, but flag for consistency with how `charts/billing` does it, if it does it differently.

`templates/ingress.yaml` hardcodes `nginx.ingress.kubernetes.io/rewrite-target: /` as an annotation regardless of the ingress class configured via `.Values.ingress.className`. If someone deploys this chart against a cluster using a different ingress controller (Traefik, say), that nginx-specific annotation is inert but confusing. Consider making annotations fully configurable via `.Values.ingress.annotations` rather than any hardcoded ones baked into the template, which is more idiomatic Helm chart design anyway and matches how most public charts handle this.

No `NetworkPolicy` template, and I notice the service this replaces (before it was folded into the monolith, per the PR description) sat behind a fairly locked-down network policy. If that requirement still applies, this chart should probably include an equivalent — right now the notifications service can reach and be reached by anything else in the namespace by default.

Also small: `templates/tests/test-connection.yaml` (the Helm test hook) is missing. Not required, but the `billing` chart has one and it's a nice cheap smoke test for `helm test` after install — would be good for consistency.

Nothing here is a hard blocker for merging — the chart is functional and follows the established pattern closely enough. I'd like the ingress annotation hardcoding addressed before merge since it's a quick fix, and would appreciate a follow-up issue for the NetworkPolicy gap so it doesn't get lost.
