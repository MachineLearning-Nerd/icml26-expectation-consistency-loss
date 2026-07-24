# Claim 2 route C source audit

Theorem 3.2 (Section 3.4, Eq. 9) covers empirical Eq. 5 “or Eq. 8,” with
hard target proportions or soft analogues, per-bin source/target counts, and
an unspecified absolute constant. It states a one-sided high-probability gap
and derives `O(B/epsilon^2)` after suppressing logarithmic factors.

Appendix G does not prove the displayed formula: its coordinate union creates
an extra `sqrt(K)`, and it does not bound estimation of target bin weights.
This route does not reuse that derivation. It proves the stronger absolute gap
using a dimension-free Hilbert-space Bernstein bound, self-normalized weighted
means, and a separate simplex target-mass bound.

The vector concentration step is anchored to Iosif Pinelis, “Optimum Bounds
for the Distributions of Martingales in Banach Spaces,” *Annals of
Probability* 22(4), 1994, DOI `10.1214/aop/1176988477`. Its 2-smooth
Banach-space result specializes to the Euclidean/Hilbert-space posterior
vectors used here.

The paper leaves fixed-function independence and positive denominators
implicit. They are necessary for the population estimand and displayed
conditional means to be defined, and are made explicit in the contract.
