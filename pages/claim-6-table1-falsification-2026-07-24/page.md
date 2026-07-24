# Claim 6 — exact Table 1 conjunction falsification

**Verdict: FALSIFIED. Confidence: HIGH.**

The compound claim says ECL alone simultaneously satisfies all five Table 1
capabilities, with Figure 2 and Appendix Table 3 as corroboration. Table 1
calls one required capability “theoretically mini-batch trainable,” and
Section 3.5 defines that phrase as Theorem 3.3's unbiased mini-batch gradient
identity.

Assumption-complete rational witnesses contradict that identity:

| Witness | Full or true quantity | Claimed mini-batch/profiled quantity |
| --- | ---: | ---: |
| Appendix H scaling | `3` | `3/2` |
| Same-batch direction | `1` | `0` |
| Omitted soft-weight derivative | `3/4` | `0` |
| Eq. 8 versus profiled Eq. 10 gradient | `0` | `1/4` |

The Eq. 8 and profiled Eq. 10 losses are also `3/4` and `65/128`.
An independent stdlib `Fraction` checker recomputes the first two witnesses
exhaustively. Controls show that a correctly scaled fixed-direction estimator
with independent frozen state can be unbiased.

Because one required all-true Table 1 cell is false, the simultaneous-five
conjunction is false. This is a direct falsification of Claim 6's Table 1
assertion and does not depend on treating missing PACS evidence as a failure.

A separate five-seed, 400-source/400-target simulation following the paper
text found ECL errors `0.15486`, `0.06380`, and `0.14994` for top-label,
class-wise, and canonical calibration. It diverges from the paper but is not
used as the falsification because the paper text and released notebook
conflict materially.

Exact raw results, source linkage, independent checker, controls, and
limitations are under
`repro/evidence/2026-07-24/artifacts/claim-6/route-2-table1/`. The simulation
evidence is in `route-1-simulation/`.
