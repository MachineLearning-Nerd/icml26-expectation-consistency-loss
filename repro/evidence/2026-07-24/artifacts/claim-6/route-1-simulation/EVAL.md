# Claim 6 route 1 evaluation

Route assessment: **DIVERGENT_OR_MIXED**. This is not a claim verdict.

Run `0b604220-28c9-4944-b35b-61d7e8f1652f` at Git
`4c1dd91f690434b048e19c8b7a4133bd89eb9f4e` passed 18/18 cumulative steps and
147 tests. Mean target calibration errors (uncalibrated / Soft-ECE / ECL) were:

- Top-label: `0.076821 / 0.097668 / 0.154860`
- Class-wise: `0.061386 / 0.084160 / 0.063803`
- Canonical: `0.085390 / 0.150460 / 0.149937`

ECL failed the predeclared alignment rule in all three paradigms. The result is
not an assumption-complete falsification because the paper and released
notebook conflict on distribution, sample size, optimizer, and epochs, and
this route does not test the Table 1 or PACS conjuncts.
