This is a clean migration from the old `PythonOperator`-based DAG to the TaskFlow API, and it's a genuine readability improvement — the `@task` decorators plus implicit XCom passing via return values and function arguments make the actual data flow between `extract`, `validate`, and `load` much easier to follow than the previous explicit `xcom_pull`/`xcom_push` calls scattered through each callable.

I checked that task dependencies are preserved correctly through the new syntax — `load(validate(extract()))` produces the same DAG structure as the old explicit `>>` chaining, and the task IDs remain stable, which matters for anyone with dashboards or alerts keyed on task names from the previous version.

Two small things. `extract()` is typed to return a `pd.DataFrame`, which TaskFlow will serialize through XCom same as before — this migration doesn't change the underlying size concern if that DataFrame is ever large, it's just cleaner syntax over the same mechanism. Not a blocker for this PR since it's a straight refactor and not the place to also solve that, but worth a comment or follow-up ticket noting it's still there.

Also, `@task(retries=3)` is set on `load` but not on `extract` or `validate` — was that intentional, or did it just get missed in the port from the old operator kwargs? If `extract` is the one hitting an external API, that seems like the task most likely to benefit from retries on transient failures, so I'd expect it to have them too unless there's a reason it doesn't.

Good refactor otherwise, no functional concerns — just want confirmation on the retry configuration before merging.
