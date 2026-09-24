# Automatically Categorize Imported Transactions with Rules

Ledgerbox can tag your imported bank transactions for you, so you don't have to categorize every coffee and utility bill by hand. This guide shows you how to create a categorization rule, apply it to transactions you've already imported, and set up a weekly habit for reviewing anything the rules miss.

## Before you begin

- You have at least one bank account connected, or you've imported a CSV/OFX file.
- You've created the tags you want to use, such as `Groceries`, `Dining`, or `Utilities`. To add tags, go to **Settings > Tags**.

## How rules work

A rule has two parts:

- **Conditions**: what a transaction must match, for example "description contains `TRADER JOE`".
- **Actions**: what Ledgerbox does when a transaction matches, for example "add tag `Groceries`".

Ledgerbox checks rules from top to bottom each time new transactions are imported. By default, the first matching rule wins. To let later rules keep evaluating after a match, turn on **Continue matching** for that rule.

## Create a rule

1. Go to **Transactions > Rules** and click **New rule**.
2. Enter a descriptive **Name**, such as `Groceries – Trader Joe's`.
3. Under **Conditions**, click **Add condition** and set:
   - **Field**: `Description`
   - **Operator**: `contains`
   - **Value**: `TRADER JOE`

   Matching isn't case-sensitive. To match more than one merchant, add another condition and set the rule to match **Any** condition.
4. Optionally, add an amount condition to narrow the rule. For example, `Amount` `is less than` `200` keeps a large one-off purchase from being tagged as routine groceries.
5. Under **Actions**, choose **Add tag** and select `Groceries`. You can also add actions to **Rename payee** (for example, to `Trader Joe's`) or **Assign to budget**.
6. Click **Preview matches**. Ledgerbox lists existing transactions that the rule would affect. If the list includes something unexpected, tighten the conditions.
7. Click **Save**.

> **Tip:** Bank descriptions often include store numbers or dates, such as `TRADER JOE'S #552 09/14`. Use `contains` with the stable part of the text rather than `equals`.

## Apply the rule to past transactions

New rules apply only to future imports by default. To categorize transactions you've already imported:

1. On the **Rules** page, open the rule's **⋯** menu.
2. Select **Run on existing transactions**.
3. Choose a date range and whether to **Skip transactions that already have a tag**. Keep this option on if you've tagged some transactions by hand and don't want the rule to overwrite them.
4. Click **Run**.

## Order your rules

Specific rules should sit above general ones. For example, a rule that tags `AMAZON PRIME` as `Subscriptions` should appear before a rule that tags any `AMAZON` transaction as `Shopping`. To change the order, drag rules by their handle on the **Rules** page.

## Review uncategorized transactions weekly

No rule set catches everything. A short weekly review keeps your reports accurate.

1. Go to **Transactions** and select the **Uncategorized** saved filter. It shows transactions with no tag.
2. For each transaction, either:
   - Assign a tag manually, or
   - Click **Create rule from this** if the merchant will appear again. Ledgerbox pre-fills a condition from the transaction's description, so you can adjust it and save.
3. Open **Transactions > Rules > Activity** and check which rules fired this week. If a rule is tagging the wrong transactions, edit its conditions.

To make the review routine, go to **Settings > Notifications** and turn on **Weekly uncategorized summary**. Choose a day, such as Sunday evening. Ledgerbox will send you a count and a direct link to the filter.

## Troubleshooting

- **A rule didn't fire.** Check that a rule higher in the list didn't match first. Also compare the raw bank description, which is shown under **Details** on the transaction, with your condition value.
- **Too many matches.** Add an amount condition or an account condition, or switch from `contains` to `starts with`.
