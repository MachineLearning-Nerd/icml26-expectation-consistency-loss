# Corrected proof for hard and soft ECL bins

Let `a_j(X) in [0,1]` be the fixed assignment of a sample to bin `j`, with
`sum_j a_j(X)=1`, and let `p(X)` lie in the `K`-simplex. For domain `d`, write

`q_dj = E[a_j(X)]`, `mu_dj = E[a_j(X)p(X)]/q_dj`,
`N_dj = sum_i a_j(X_i)`, and
`muhat_dj = sum_i a_j(X_i)p(X_i)/N_dj`.

All maps are fixed independently of the evaluation samples and every displayed
denominator is positive.

## Lemma 1: realized-count conditional mean

For a fixed bin, set `Q=n*q` and
`S=sum_i a_j(X_i)(p(X_i)-mu_j)`. The summands are centered Hilbert-space
vectors, have norm at most `sqrt(2)`, and total second moment at most `2Q`.
A Hilbert-space Bernstein inequality (Pinelis, 1994,
DOI `10.1214/aop/1176988477`) bounds `||S||` by a constant times
`sqrt(Q*l)+l`. Scalar Bernstein gives the matching deviation bound between
`N_j` and `Q`.

On their joint event, if `N_j >= 16l`, the scalar bound implies `Q <= 2N_j`;
substitution into the vector bound gives a coefficient below 8 times
`sqrt(l/N_j)`. If `N_j < 16l`, simplex diameter gives
`||muhat_j-mu_j|| <= sqrt(2) < 8sqrt(l/N_j)`. Allocating failure probability
over two domains and `B` bins and absorbing
`log(8B/delta)/log(2BK/delta) <= 3/2` for `K>=2` yields

`||muhat_dj-mu_dj|| <= 8 sqrt(L/N_dj)`,
where `L=log(2BK/delta)`.

## Lemma 2: weighted contraction

Let `what_j=N_tj/N_t`. The assignments sum to one, so
`sum_j what_j=1`. Weighted Cauchy–Schwarz gives

`sum_j what_j/sqrt(N_dj) <= sqrt(sum_j what_j/N_dj)`.

For the target counts specifically,

`sum_j what_j/N_tj = sum_j (N_tj/N_t)/N_tj = B/N_t`.

Applying Lemmas 1–2 to both domains therefore costs at most

`8 sqrt(2L * sum_j what_j(1/N_sj+1/N_tj))`.

## Lemma 3: target-bin mass

The empirical target assignment mean `what` and its population mean `q_t`
are both in the `B`-simplex. Independence, Jensen, and
`||a(X)||_2<=1` give

`E||what-q_t||_1 <= sqrt(B/N_t)`.

Changing one target sample changes the L1 norm by at most `2/N_t`, so
McDiarmid adds `sqrt(2log(2/delta)/N_t)`. Each per-bin population
source-target mean distance is at most `sqrt(2)`. Thus the target-weight
fluctuation costs no more than 4 times the displayed Eq. 9 radius, because
that radius already contains `L*B/N_t`.

## Conclusion

The conditional-mean budget is `8sqrt(2)` and the target-mass budget is at
most 4. Their sum is `15.3137085 < 16`. Consequently, with probability at
least `1-delta`,

`abs(Lhat_ecl-L_ecl) <= 16 sqrt(L * sum_j what_j(1/N_tj+1/N_sj))`.

This is stronger than the paper's one-sided Eq. 9 and covers hard assignments
as a special case. Balanced counts give the stated
`O(B log(2BK/delta)/epsilon^2)` total sample requirement, conventionally
written `O(B/epsilon^2)` when logarithmic confidence and class factors are
suppressed.
