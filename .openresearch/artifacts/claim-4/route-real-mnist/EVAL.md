# Claim 4 real-MNIST evaluation

Verdict: **VERIFIED** for the real-data Appendix F implementation contract.

Local run `92cf7d05-fdaa-4e0a-ab94-2901859aed04` at Git
`71002c3017d2951c0f16d6f7d6f088cf2272fa1c` passed all 14 cumulative steps.
The official MNIST hashes matched and both trained heads used disjoint
30,000-image training halves. On all 10,000 held-out test images, the three
loss/temperature-gradient pairs were:

- top-label: `0.0211294191681 / -0.0306350725125`;
- class-wise: `0.0812220206965 / 0.00429677193796`;
- canonical: `0.0374209547883 / -0.0137004196946`.

The independent NumPy checker reproduced every loss within `2.02e-16` and every
centered finite-difference gradient within scaled error `2.91e-08`. Detaching
scores removed the gradient; posterior shuffling changed every loss; and
class-wise/canonical computations did not collapse to top-label.

This verdict does not cover Table 2 performance or claim that a logistic model
is the paper's deep digit architecture.
