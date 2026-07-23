# Reproduction: Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift

Claim-focused reproduction and source audit of Dong, Jiang, and Yang
([OpenReview `gFPPTokv9C`](https://openreview.net/forum?id=gFPPTokv9C), arXiv `2605.21552v1`).

The current challenge prompt has six anchored claims. The live legacy judge still scores three
defaults: its C1 maps to anchored C1, C2 compatibility maps to anchored C4, and C3 sample
complexity maps to anchored C2. The last confirmed official verdict is **5/6** at Space SHA
`1abb0c87beb604420d3a0e6140ea122511c63e93`: legacy C1/C2 are `verified`, while legacy C3
sample complexity is `toy`. The final real-MNIST trained-model attempt below is a local assessment
until the updated Space is judged.

## Pages

| Page |
| --- |
| [Executive summary](#/executive-summary) |
| [Claim 1 — Expectation-consistency theorem](#/claim-1-calibration-iff) |
| [Claim 2 — ECL sample complexity](#/claim-2-sample-complexity) |
| [Claim 3 — Mini-batch gradient unbiasedness](#/claim-3-gradient-unbiasedness) |
| [Claim 4 — Calibration extensions](#/claim-4-calibration-extensions) |
| [Claim 5 — Digit benchmark Table 2](#/claim-5-digit-table) |
| [Claim 6 — Capability matrix and broad empirics](#/claim-6-capabilities-empirics) |
| [Conclusion](#/conclusion) |
