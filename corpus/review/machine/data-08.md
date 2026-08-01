This is a solid, pragmatic set of checks for a first validation layer, and I'd be comfortable merging it with a couple of small tweaks rather than a full rework.

The null-rate check — asserting each column's null percentage stays under a configured threshold and raising with the actual percentage and column name in the message — is exactly the kind of thing that saves someone an afternoon of debugging later, and I like that the thresholds are per-column rather than one global number, since `optional_notes` and `customer_id` obviously shouldn't share a tolerance.

Two things worth adjusting. First, the schema drift check compares `set(df.columns) == set(expected_columns)`, which will fail loudly (as intended) if a column is added or removed, but it doesn't distinguish those two cases in the error message — right now it just says "schema mismatch." Splitting into "unexpected columns: {...}" and "missing columns: {...}" would make the failure immediately actionable instead of requiring someone to diff the lists by hand.

Second, you're running these checks with plain `assert` statements. That's fine for a notebook or ad hoc script, but if this ever runs with Python's `-O` flag (or in any environment where assertions get stripped), the checks silently vanish and bad data flows through undetected. Raising an explicit exception (or moving to something like Great Expectations or pandera, since you're clearly already thinking in those terms) removes that failure mode for good.

Minor nit: `check_nulls(df, threshold=0.05)` — worth documenting in the docstring whether that's a fraction or a percentage, since a caller passing `5` for "5 percent" by mistake would sail past the check test entirely. Otherwise this is a good foundation to build on.
