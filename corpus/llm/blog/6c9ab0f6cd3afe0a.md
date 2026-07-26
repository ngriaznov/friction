# Postmortem: The 40-Minute Outage from a Migration That Should Have Been Boring

Last Tuesday, a routine-looking database migration locked our primary `orders` table for the better part of forty minutes and took our checkout flow down with it. This is the timeline, what actually caused it, and the changes we're making to our deploy process so it doesn't happen again.

## Timeline

**14:02** — A deploy goes out containing a migration that adds a new `fulfillment_status` column to the `orders` table, along with a `NOT NULL` default and a new index on it.

**14:03** — Checkout error rates start climbing. The on-call engineer gets paged by our latency alert before the error-rate alert even fires.

**14:06** — First responder checks the deploy log, sees the migration went out three minutes earlier, and pulls up `pg_stat_activity` on the primary. There's a single query holding an `ACCESS EXCLUSIVE` lock on `orders`, and a growing queue of everything else waiting behind it.

**14:09** — Incident declared. A second engineer joins to help assess whether to kill the migrating transaction or let it finish.

**14:14** — We decide killing it mid-migration risks leaving the table in a half-altered state, so we let it run and start routing new checkout attempts to a maintenance page instead of letting them queue and time out.

**14:41** — The migration completes. The lock releases, the backlog of queued queries drains within about ninety seconds, and checkout traffic returns to normal.

**14:45** — Maintenance page removed. Incident downgraded to resolved, full retro scheduled for the next day.

## Root cause

The migration did two things in a single transaction: it added the `fulfillment_status` column with a `NOT NULL DEFAULT 'pending'`, and it created an index on that column. On the Postgres version we're running, adding a column with a non-null default requires rewriting every existing row to backfill the value, and that rewrite happens under the same `ACCESS EXCLUSIVE` lock that a plain `ALTER TABLE` takes to change the table's schema. With roughly 40 million rows in `orders`, the rewrite took long enough that the lock became the bottleneck, and everything else touching that table — including every checkout — queued up behind it.

The index creation would have been a smaller problem on its own; it was the column-with-default rewrite, done in the same statement, that turned a routine schema change into a production outage. Locally and in staging this migration ran in under a second, because both environments have a few thousand rows in `orders`, not tens of millions.

## What we're changing

A few concrete follow-ups came out of the retro:

1. **Split default-bearing column additions into two steps.** Add the column as nullable with no default, backfill the value in batches with a separate script, then add the `NOT NULL` constraint once the backfill is complete. Each step takes a much shorter lock, and none of them block the table for the full duration of a 40-million-row rewrite.

2. **Add a migration size check to CI.** Any migration touching a table above a row-count threshold now gets flagged for manual review before it can merge, rather than relying on the author to remember which tables are large.

3. **Run migrations against a staging replica seeded with production-scale data**, not just the small staging dataset we'd been using. This is the change we expect to catch the most future incidents like this one, since the failure mode here was entirely invisible until the table size crossed a threshold none of our existing tests exercised.

4. **Add lock-wait monitoring** as a first-class alert, separate from our general latency alert. We got lucky that latency paged us quickly; a slower-building lock queue on a lower-traffic table might not have tripped that alert as fast.

Nobody did anything reckless here — the migration was reviewed, it passed CI, and it behaved exactly as intended everywhere we'd tested it. The gap was entirely in what "tested" meant: correct isn't the same as safe at scale, and our staging environment wasn't shaped enough like production to tell the difference.
