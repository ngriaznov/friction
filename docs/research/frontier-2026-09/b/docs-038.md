# Automatically Categorizing Bank Transactions in Ledgerbox

Ledgerbox can tag imported transactions for you using rules you define. A rule matches on fields like the payee name or amount and applies one or more tags whenever a new transaction fits. Once a handful of rules are in place, most of your monthly activity is categorized the moment it is imported, and only the leftovers need a manual look.

This guide walks through creating your first rule, ordering rules sensibly, and setting up a short weekly review for anything the rules missed.

## Before you start

- Import at least one statement so you have real transactions to test against. Rules are much easier to write when you can see the actual payee strings your bank uses.
- Decide on a small set of tags to begin with (for example `groceries`, `rent`, `transport`, `subscriptions`). You can always add more later, but starting with a dozen categories keeps rules manageable.

## Create a rule

1. Open **Settings > Rules** and click **New rule**.
2. Give the rule a name that describes what it catches, such as `Weekly supermarket`.
3. Under **Conditions**, add one or more matchers. Each condition has a field, an operator, and a value:
   - **Payee contains** `WHOLEFDS` catches every Whole Foods transaction regardless of the store number appended to the name.
   - **Amount is between** `-20.00` and `-5.00` narrows to small purchases.
   - **Account is** `Joint checking` limits the rule to one account.
   When you add more than one condition, all of them must match.
4. Under **Actions**, choose the tags to apply. You can apply several, for example `groceries` and `shared`.
5. Click **Preview**. Ledgerbox shows every existing transaction the rule would match. Scan the list for false positives; a payee string like `SHELL` may match both a petrol station and an unrelated merchant.
6. Adjust the conditions until the preview looks right, then click **Save**.

New rules apply to future imports automatically. To apply a rule to transactions already in your ledger, click **Run on existing** on the rule's page.

### Tips for reliable matching

- Match on the stable part of the payee string. Banks often append dates, locations, or reference numbers, so `contains` is usually safer than `equals`.
- Use the amount range for recurring bills whose payee names change but whose totals do not.
- Rules are evaluated top to bottom, and the first matching rule wins by default. Put specific rules above general ones: a rule for `AMAZON PRIME` should sit above a broad `AMAZON` rule. Drag rules to reorder them on the Rules page.
- If you want a transaction to collect tags from several rules, enable **Continue after match** on the earlier rule.

## Review uncategorized transactions weekly

Even good rules miss things: a new merchant, a refund, a one-off transfer. Ledgerbox keeps these in a single view so they do not pile up unnoticed.

1. Open **Transactions** and choose the **Uncategorized** filter from the sidebar. This shows every transaction with no tags.
2. Work through the list. For each item you have two choices:
   - **Tag it once.** Click the tag field, pick a tag, and move on. Use this for genuinely unusual items.
   - **Create a rule from it.** Click the three-dot menu on the row and choose **Create rule from transaction**. Ledgerbox pre-fills the payee and account conditions, and you only need to trim the payee string and pick the tags. Use this whenever you expect the merchant to show up again.
3. When the list is empty, you are done for the week.

To make this a habit, turn on **Settings > Notifications > Weekly uncategorized summary**. Ledgerbox sends a short email every Monday listing the count and total of untagged transactions, with a link straight to the filtered view.

Ten minutes a week is usually enough. After the first month, most of your review time goes into refining rules rather than tagging by hand, and the uncategorized list stays short.
