## P-Hacking and the Multiple Comparisons Problem: A Worked Example

**Question:** I'm running 20 independent hypothesis tests, each with an alpha level of 0.05. What are the chances of getting at least one statistically significant result purely by chance? How can I adjust my alpha level to account for this increased risk?

**Answer:** This question delves into the tricky territory of **p-hacking** and the **multiple comparisons problem**. Let's break it down:

**P-Hacking:**

P-hacking refers to questionable research practices that increase the likelihood of finding statistically significant results, often by manipulating data or analysis methods. It can involve:

* **Selective reporting:** Only publishing studies with positive results while ignoring null findings.
* **Data dredging:** Analyzing data in various ways until a significant result emerges, without a clear hypothesis beforehand.
* **Changing significance thresholds:** Adjusting the alpha level after seeing initial results to achieve significance.

**Multiple Comparisons Problem:**

When conducting multiple hypothesis tests, the probability of obtaining at least one statistically significant result purely by chance increases. This is because each test has its own probability of producing a false positive (Type I error).

**Example:**

Let's say you have 20 independent hypotheses and an alpha level of 0.05 for each test. The probability of getting a Type I error for *each* individual test is 0.05 (5%).  However, the probability of getting at least one Type I error across all 20 tests is much higher than 5%.

**Calculating the Probability:**

Using the complement rule, we can calculate the probability of *not* getting any Type I errors:

(1 - 0.05)^20 ≈ 0.36

Therefore, the probability of getting at least one Type I error across all tests is approximately 1 - 0.36 = **0.64 or 64%**.

**Bonferroni Correction:**

To address the multiple comparisons problem, we can use a correction like the **Bonferroni correction**. This method adjusts the alpha level for each individual test by dividing it by the number of tests:

Adjusted alpha = Original alpha / Number of tests

In our example:

Adjusted alpha = 0.05 / 20 = 0.0025

This means that to achieve a desired overall significance level of 0.05, each individual test now needs to reach a p-value of less than 0.0025 to be considered statistically significant.

**Conclusion:**

The multiple comparisons problem significantly increases the risk of finding false positives when conducting multiple hypothesis tests.  P-hacking exacerbates this issue by introducing bias and manipulation into the research process. Using corrections like the Bonferroni method can help mitigate these risks, but it's crucial to remember that statistical significance alone does not guarantee scientific validity.
