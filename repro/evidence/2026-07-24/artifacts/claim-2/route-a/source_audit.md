# Claim 2 route A source audit

Theorem 3.2 (`S3.Thmtheorem2`) quantifies over
`epsilon, delta in (0,1)` and asserts existence of an absolute constant in the
displayed Eq. 9 bound for empirical Eq. 5 “or Eq. 8.” Appendix G (`A7`) does
not derive that display as printed: its coordinate-wise route retains a
`sqrt(K)` factor and it omits empirical target-bin-mass error.

This route does not reuse that proof. It proves the fixed hard-bin Eq. 5
statement directly in Euclidean norm using the second moment of a
simplex-valued sample mean, bounded differences, weighted Cauchy-Schwarz, and
a scalar Hoeffding term for empirical target bin masses. The derivation gives
the explicit absolute constant `2*sqrt(2)+1 < 4`.

The soft self-normalized Eq. 8 reading is intentionally outside this route and
is tested separately rather than silently treated as proved.
