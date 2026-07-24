# Claim 5 — four routes and a BLOCKED verdict

**Verdict: BLOCKED. Confidence: LOW.**

The exact claim is historical: Table 2 reports ten-run digit-transfer
mean/standard-deviation pairs for LeNet-5, ResNet20, and DenseNet40. A valid
counterexample must reproduce the same data construction, objectives,
architectures, hyperparameters, and ten-run schedule, independently recompute
all six cells, and contradict rather than merely sample a different valid
stochastic realization.

Exactly three materially different verification routes were completed:

1. The literal Algorithm 2 route reached the Soft-ECE square-root singularity
   and produced nonfinite values on full data.
2. A repaired public-predecessor post-hoc route produced `54.3883%`
   uncalibrated and `10.1819%` ECL ECE for one full LeNet seed.
3. A stabilized Appendix J in-training route produced `41.9553%` and
   `68.4514%` for one full LeNet seed.

Confidence remained LOW, so the mandatory fourth route explicitly sought an
assumption-complete falsification. It rejected the first route because a
nonfinite execution cannot contradict a finite historical summary, and the
other two because they contain declared repairs, cover one seed and one
architecture, and do not match the unavailable generating protocol.

The fourth route independently verified that all six printed mean/SD pairs
have valid bounded ten-run realizations. That does not verify the historical
runs; it rules out internal arithmetic impossibility as a falsification route.
No valid counterexample was found.

Unblocking requires the exact ten raw prediction sets or checkpoints and seed
schedule, executable digit objectives for all three architectures, and the
missing loss coefficient and preprocessing/split details. The route-4
contract, raw results, checker, controls, and limitations are under
`repro/evidence/2026-07-24/artifacts/claim-5/route-4-falsification/`; the first
three route records are in adjacent directories.
