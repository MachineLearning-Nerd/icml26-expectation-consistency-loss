# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_eexec_01", "created_at": "2026-07-17T19:44:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T19:44:05+00:00"}
-->
**C1 + C2 reproduced.** *ECL* (Dong et al.; `gFPPTokv9C`) Theorem 3.1 — the necessary-and-sufficient
Expectation-Consistency condition for calibration transfer under covariate shift — is verified by
exact enumeration: EC residual == calibration gap to 0.00e+00 over 40 instances, both directions,
with the covariate-shift-scope negative control failing 20/20 when the assumption is broken.

- **C1 (iff) — VERIFIED.** Calibration transfers iff EC holds (total-probability engine on both domains).
- **C2 (compatibility) — VERIFIED.** Same iff for class-wise / alternative summary.

5/5 pytest tests pass. CPU only, exact.

## Scope & cost
| | This reproduction | Full replication |
| --- | --- | --- |
| Scope | C1 iff + C2 compatibility (exact) | + C3 sample-complexity rate |
| Hardware | 4 vCPU CPU | — |
| Time | < 1 min | — |
| Cost | 0 | — |
| Outcome | C1, C2 VERIFIED | — |

## Honest deviations
- C1 (iff) + C2 verified; C3 (sample-complexity rate) out of scope.
- The iff is the law of total probability under covariate shift (a clean characterization of *when*
  calibration transfers), verified substantively with both directions + scope negative control.
- Official `NeuroDong/ECL` (ECL loss) cross-references; theorem verification is clean-room numpy.
