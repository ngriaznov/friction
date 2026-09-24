# Millrace Environment Variables

Millrace reads its configuration from environment variables at startup. Variables are grouped below by the subsystem they control. Values are parsed once when the process starts; changing a variable requires restarting the runner.

Booleans accept `true`, `false`, `1`, `0`, `yes`, and `no` (case-insensitive). Durations accept an integer followed by a unit: `ms`, `s`, `m`, or `h`. An invalid value causes Millrace to exit with a non-zero status and a message naming the variable.

## Logging

| Variable | Default | Accepted values | Description |
|---|---|---|---|
| `MILLRACE_LOG_LEVEL` | `info` | `trace`, `debug`, `info`, `warn`, `error` | Minimum severity written to the log. Levels below the threshold are discarded. |
| `MILLRACE_LOG_FORMAT` | `text` | `text`, `json` | Output format. Use `json` when shipping logs to an aggregator; each line is one JSON object. |
| `MILLRACE_LOG_OUTPUT` | `stderr` | `stdout`, `stderr`, or a file path | Where log lines are written. A file path is opened in append mode and created if missing. |
| `MILLRACE_LOG_JOB_PAYLOAD` | `false` | boolean | Include the full job payload in job start and finish entries. Off by default because payloads may contain sensitive data. |
| `MILLRACE_LOG_TIMESTAMP` | `rfc3339` | `rfc3339`, `unix`, `none` | Timestamp format for each entry. `none` is useful when a supervisor such as systemd adds its own timestamps. |

## Concurrency

| Variable | Default | Accepted values | Description |
|---|---|---|---|
| `MILLRACE_WORKERS` | Number of CPU cores | `1` to `1024` | Number of worker goroutines that pull and execute jobs. Values above the core count are appropriate for I/O-bound workloads. |
| `MILLRACE_QUEUE_PREFETCH` | `2` | `1` to `100` | Jobs fetched from the queue per worker ahead of execution. Higher values reduce round trips but hold more jobs in memory. |
| `MILLRACE_MAX_PER_QUEUE` | `0` | `0` to `1024` | Upper bound on workers that may run jobs from a single named queue at once. `0` means no per-queue limit. Must not exceed `MILLRACE_WORKERS`. |
| `MILLRACE_JOB_TIMEOUT` | `10m` | `1s` to `24h` | Maximum wall-clock time a single job may run before it is cancelled and marked failed. Individual jobs may override this with a shorter value. |
| `MILLRACE_SHUTDOWN_GRACE` | `30s` | `0s` to `1h` | Time to wait for in-flight jobs to finish after receiving `SIGTERM`. Jobs still running when the grace period ends are cancelled and returned to the queue. |

## Retry policy

| Variable | Default | Accepted values | Description |
|---|---|---|---|
| `MILLRACE_RETRY_MAX` | `5` | `0` to `100` | Number of retry attempts after the initial failure. `0` disables retries; failed jobs go straight to the dead-letter queue. |
| `MILLRACE_RETRY_BASE_DELAY` | `2s` | `10ms` to `1h` | Delay before the first retry. Subsequent delays are computed from this value using the backoff strategy. |
| `MILLRACE_RETRY_MAX_DELAY` | `10m` | `1s` to `24h` | Cap on the delay between any two attempts. Must be greater than or equal to `MILLRACE_RETRY_BASE_DELAY`. |
| `MILLRACE_RETRY_BACKOFF` | `exponential` | `constant`, `linear`, `exponential` | How the delay grows between attempts. `exponential` doubles the delay each time; `linear` adds the base delay each time; `constant` uses the base delay for every attempt. |
| `MILLRACE_RETRY_JITTER` | `0.2` | `0.0` to `1.0` | Fraction of the computed delay added or subtracted at random to spread out retries. `0.0` disables jitter. |
| `MILLRACE_RETRY_ON_TIMEOUT` | `true` | boolean | Whether a job cancelled by `MILLRACE_JOB_TIMEOUT` counts as a retryable failure. Set to `false` to send timed-out jobs directly to the dead-letter queue. |
| `MILLRACE_DEAD_LETTER_QUEUE` | `millrace.dead` | Queue name, 1 to 128 characters | Queue that receives jobs after all retries are exhausted. Set to an empty string to discard exhausted jobs instead. |

## Precedence

Command-line flags override environment variables, and environment variables override values in the optional `millrace.toml` config file. Run `millrace config show` to print the effective configuration with the source of each value.
