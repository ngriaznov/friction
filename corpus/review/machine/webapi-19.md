This PR replaces two hand-written raw SQL queries in the Django reporting views with equivalent QuerySet code, and separately fixes an N+1 in the `InvoiceListView` by adding `select_related('customer', 'billing_address')`. Touches `reports/views.py` and `invoices/views.py`, plus updates to two existing tests' expected query counts.

Good cleanup. The raw SQL being replaced (`SELECT customer_id, SUM(total) FROM invoices WHERE created_at >= %s GROUP BY customer_id`) maps cleanly onto `Invoice.objects.filter(created_at__gte=cutoff).values('customer_id').annotate(total=Sum('total'))`, and moving it into the ORM gets you back parameterization consistency with the rest of the codebase, plus the query now participates correctly in whatever `using()` database routing the rest of the app relies on — I noticed the raw SQL version was calling `connection.cursor()` directly, which bypasses that routing entirely, so this is a real fix, not just a style preference.

The `select_related` addition on `InvoiceListView` is correct and the updated test assertion (`assertNumQueries(2)`, down from what I assume was significantly more before) is good evidence it's working as intended.

Two small things. The migrated `values().annotate()` query returns dicts rather than model instances, and the view code that consumes it does `[row['customer_id'] for row in results]` a few lines down — that's fine, just flagging that if this data ever needs to flow through a DRF serializer that expects model instances (e.g., if this view later gets a JSON API sibling), that assumption will need to change; not a problem today.

Also, `total=Sum('total')` shadows the field name `total` with the annotation name `total` — it works because Django resolves annotations before falling back to fields in this context, but it reads confusingly on a second pass and I'd rename the annotation to `total_amount` just for clarity, since "which `total` is this" is a legitimate question when skimming.

Nothing here concerns me enough to block. Approving — nice, low-risk cleanup that also happens to fix a routing bug you may not have set out to fix.
