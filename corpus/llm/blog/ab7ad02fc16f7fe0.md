# Rate limiting our public API with token buckets

We opened up a public read API in March. By late April one integrator had written a sync loop with no backoff, and on a bad afternoon they accounted for about 60% of our request volume. Nothing fell over, but our p99 got ugly and the on-call engineer spent an hour figuring out why. Time to add rate limiting.

## What we actually needed

The requirements were boring, which is good:

- A per-API-key limit, roughly 60 requests per minute.
- Tolerance for short bursts. Our own client SDK fires six or seven requests in parallel on startup to warm a cache, and rejecting those would have been self-inflicted.
- Correct behavior across four API servers behind a load balancer, so the state has to be shared.
- Cheap. This runs on every request; it cannot cost more than a millisecond.

That last point mattered more than it sounds like it should, and it's most of the reason we landed where we did.

## Why not a sliding window

The obvious alternative was a sliding window log: store a timestamp for every request the key made, drop entries older than 60 seconds, count what's left. It's exact. It has no edge effects. It's easy to explain to a customer when they ask why they got a 429.

We rejected it on memory and write volume. A key doing its full 60 requests per minute needs 60 timestamps held for a minute. That's fine for one key. We have around 4,000 active keys, and the busiest ones would be at their ceiling constantly. In Redis that's 4,000 sorted sets, each getting a ZADD plus a ZREMRANGEBYSCORE on every single request. We prototyped it and the Redis CPU graph was not encouraging. The sorted set operations are O(log n), which is fine in isolation, but we were doing three round trips per request in the naive version.

There's a middle option — the sliding window *counter*, which keeps two fixed buckets and interpolates between them — and it's genuinely good. Two integers per key, one INCR, approximately correct. We would probably have shipped it if we hadn't wanted burst tolerance to be an explicit, tunable thing rather than an artifact of where the window boundary happened to fall.

## The token bucket

So: token bucket. Each key gets a bucket with a capacity of 90 tokens that refills at 1 token per second. A request costs one token. Empty bucket, 429.

Capacity 90 with a 1/sec refill gives you a sustained rate of 60/minute and a burst headroom of a minute and a half of saved-up allowance. A client that's been idle can fire 90 requests immediately; a client hammering us settles into one request per second.

The state per key is two values: the token count and the timestamp of the last refill. We don't run a background job to add tokens — that would be a lot of writes for buckets nobody is using. Instead we compute lazily on read:

```
elapsed = now - last_refill
tokens  = min(capacity, tokens + elapsed * refill_rate)
```

Then decrement if there's anything left, and write both fields back. In Redis that's a hash with two fields, read-modify-write, done inside a Lua script so the whole thing is atomic. One round trip. Measured at about 0.3ms at p50 from our app servers, which is inside the budget.

The Lua script is maybe fifteen lines and it's the only clever part of the implementation. Everything else is plumbing: pulling the key off the request, mapping the key to a tier (we have three: free, standard, partner, differing only in capacity and refill rate), and formatting the 429.

## What we send back

Every response, not just the rejections, carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`. On a 429 we also set `Retry-After` to the number of seconds until one token is available, rounded up. That number falls straight out of the bucket math, which is a small nice thing about this approach — a sliding window log has to do more work to answer "when can I try again."

The integrator that started all this now backs off correctly. They read the header.
