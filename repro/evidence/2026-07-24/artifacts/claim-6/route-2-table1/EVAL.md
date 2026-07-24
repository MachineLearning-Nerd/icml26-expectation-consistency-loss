# Claim 6 route 2 evaluation

Verdict: **FALSIFIED** for the simultaneous-all-five Table 1 assertion.

Run `8905e5d7-cbae-49a4-9b09-9723793da633` regenerated the inherited exact
certificate, the logical linkage, the independent `Fraction` checker, and all
cumulative controls. It passed 20/20 campaign steps and 149 tests on local CPU
in 338 seconds.

Table 1 labels ECL as “theoretically mini-batch trainable,” and Section 3.5
defines that capability by the unbiased-gradient identity in Theorem 3.3.
Assumption-complete rational witnesses give full versus expected mini-batch
gradients `3` versus `3/2` and `1` versus `0`; separate witnesses give the
omitted soft-weight derivative `3/4` versus `0` and Eq. 8 versus profiled Eq.
10 gradients `0` versus `1/4`. The independent checker recomputes the first
two witnesses exhaustively.

Because at least one required Table 1 cell is false, ECL does not satisfy the
claimed conjunction of all five capabilities. This exact logical
falsification does not claim an independent PACS reproduction.
