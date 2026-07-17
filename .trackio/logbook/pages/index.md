# Repro - ECL Calibration under Covariate Shift (gFPPTokv9C)

Clean-room reproduction of *ECL: Rethink Confidence Calibration under Covariate Shift*
(Dong et al.; arXiv 2605.21552), OpenReview `gFPPTokv9C`.

Theorem 3.1: under covariate shift, source calibration transfers to target **iff** the
Expectation-Consistency condition holds.

## Claims
| Claim | Statement | Verdict |
| --- | --- | --- |
| **C1** | iff condition for calibration under covariate shift | **VERIFIED** |
| **C2** | compatibility (canonical / class-wise / top-label) | **VERIFIED** |

(C3 sample-complexity is a rate, out of scope.)

## Pages
- [Claim 1+2 — iff + compatibility](claim-iff) · [Methods & environment](methods-environment)
- [Negative controls](negative-controls) · [Conclusion](conclusion)

Exact discrete enumeration over covariate-shift joints. CPU only.
