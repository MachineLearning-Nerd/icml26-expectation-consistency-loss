# Claim 2 route B source audit

Theorem 3.2 explicitly names empirical Eq. 8 and says soft analogs of bin
weights and counts may be used. Eq. 8 uses Gaussian soft assignments,
self-normalized source/target conditional means, and a `1e-5` denominator
stabilizer in the published implementation.

The exact quantifier is existential in the unknown absolute constant `C`.
Therefore a violation at a hand-picked small constant or a single rare sample
cannot falsify the theorem. A valid counterexample must contradict the
probability statement while preserving fixed-function, iid, positive-mass,
and defined-denominator assumptions.
