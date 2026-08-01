The alert rules are functional but a few thresholds and the `for:` durations need a second look before you page anyone on this.

`HighErrorRate` fires when `rate(http_requests_total{status=~"5.."}[1m]) / rate(http_requests_total[1m]) > 0.05` with `for: 1m`. A one-minute window on a `rate()` that's already computed over one minute is going to be noisy — a single bad minute of traffic (a deploy, a brief upstream blip) will trip this. I'd widen the rate window to `5m` and keep `for: 2m` so you're reacting to a sustained trend rather than a blip; as written this is going to page someone during every rolling deploy that briefly 500s a few requests.

`HighMemoryUsage` at `container_memory_working_set_bytes / container_spec_memory_limit_bytes > 0.9` with no `for:` clause at all means this fires instantly on a single scrape sample above 90%, which for anything with GC-driven memory sawtooth (JVM, anything with a generational collector) is going to be constantly flapping. Add a `for: 10m` at minimum, and consider whether 90% is even the right threshold if the workload's normal operating range already brushes up against it.

Good things: labeling alerts with `severity` and routing critical vs warning through different Alertmanager receivers is the right pattern, and including a `runbook_url` annotation on each rule is genuinely above average — most teams skip that and then nobody remembers what the alert means six months later.

One gap: no alert on Alertmanager or Prometheus itself being down (the classic "who alerts on the alerting system" problem) — worth adding a dead-man's-switch style alert routed through a separate channel like a heartbeat check.

Small thing, not blocking: `expr` lines are getting long enough that they'd benefit from being split with YAML block scalars for readability, but that's purely cosmetic.
