# Millrace Environment Variables

Millrace reads the following environment variables when a worker process starts. Changes take effect only after you restart the worker. When a value is set both in the environment and in `millrace.toml`, the environment variable wins.

If a value is outside its accepted range or can't be parsed, the worker logs an error and exits during startup. It doesn't fall back to the default.

## Logging

| Variable | Default | Accepted values | Description |
|---|---|---|---|
| `MILLRACE_LOG_LEVEL` | `info` | `trace`, `debug`, `info`, `warn`, `error` | Minimum severity of log records the worker emits. |
| `MILLRACE_LOG_FORMAT` | `text` | `text`, `json` | Output format. Use `json` when logs are shipped to a collector. |
| `MILLRACE_LOG_DEST` | `stdout` | `stdout`, `stderr`, or an absolute file path | Where logs are written. If you give a file path, the worker opens it in append mode. |
| `MILLRACE_LOG_JOB_ARGS` | `false` | `true`, `false` | Includes job arguments in the `job.started` and `job.failed` records. Leave this off if arguments may contain sensitive data. |
| `MILLRACE_LOG_SLOW_JOB_MS` | `0` | `0`–`3600000` | Logs a warning when a job runs longer than this many milliseconds. `0` turns off the warning. |

## Concurrency

| Variable | Default | Accepted values | Description |
|---|---|---|---|
| `MILLRACE_CONCURRENCY` | Number of CPU cores | `1`–`512` | Maximum number of jobs one worker process runs at the same time. |
| `MILLRACE_QUEUES` | `default` | Comma-separated queue names, each 1–64 characters | Queues the worker polls. Add an optional weight with `name:weight`, for example `critical:5,default:1`. Weights must be between `1` and `100`. |
| `MILLRACE_POLL_INTERVAL_MS` | `1000` | `50`–`60000` | How often the worker checks for new jobs when its queues are empty. |
| `MILLRACE_PREFETCH` | `0` | `0`–`1000` | Number of extra jobs to reserve beyond `MILLRACE_CONCURRENCY`. `0` turns off prefetching. |
| `MILLRACE_SHUTDOWN_TIMEOUT_S` | `30` | `0`–`3600` | Seconds the worker waits for in-flight jobs to finish after receiving `SIGTERM`. When the timeout ends, any remaining jobs are released back to the queue. |

## Retry policy

These variables set the default retry policy for every job. Individual job classes can override them with the `retry` option.

| Variable | Default | Accepted values | Description |
|---|---|---|---|
| `MILLRACE_RETRY_MAX_ATTEMPTS` | `5` | `0`–`100` | Maximum number of attempts, counting the first one. With `0` or `1`, failed jobs aren't retried. |
| `MILLRACE_RETRY_BACKOFF` | `exponential` | `fixed`, `linear`, `exponential` | Strategy used to calculate the delay between attempts. |
| `MILLRACE_RETRY_BASE_DELAY_MS` | `1000` | `0`–`86400000` | Base delay. With `fixed`, every retry waits this long. With `linear`, the wait is `base × attempt`. With `exponential`, it's `base × 2^(attempt−1)`. |
| `MILLRACE_RETRY_MAX_DELAY_MS` | `3600000` | `0`–`604800000` | Upper limit on any single retry delay. It must be greater than or equal to `MILLRACE_RETRY_BASE_DELAY_MS`. |
| `MILLRACE_RETRY_JITTER` | `0.1` | `0.0`–`1.0` | Fraction of random variation applied to each delay. For example, `0.1` means ±10%. |
| `MILLRACE_RETRY_DEAD_LETTER` | `true` | `true`, `false` | Moves a job to the `dead` queue after its final failed attempt. When set to `false`, the job is discarded instead. |

## Example

```sh
export MILLRACE_LOG_FORMAT=json
export MILLRACE_CONCURRENCY=16
export MILLRACE_QUEUES=critical:5,default:1
export MILLRACE_RETRY_MAX_ATTEMPTS=8
export MILLRACE_RETRY_MAX_DELAY_MS=600000
millrace worker
```
