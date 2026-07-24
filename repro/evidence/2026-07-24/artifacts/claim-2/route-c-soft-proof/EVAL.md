# Claim 2 route C evaluation

Verdict: **VERIFIED** under the explicit fixed-function assumptions in the
claim contract.

Run `d7777b89-7141-4e80-a127-65e4f9e42f2d` regenerated the proof obligations,
all 1,024 soft-bin diagnostics, the independent checker, and every earlier
claim check. It passed 22/22 campaign steps and 151 tests on local CPU in
453.075 seconds.

The corrected argument certifies the paper's finite-sample form with absolute
constant `C=16`: its dimension-free conditional-mean budget is `8*sqrt(2)`,
the simplex mass term is at most `4`, and their sum `15.3137085` is below 16.
The largest diagnostic error/radius ratio was `0.087334129`. These diagnostics
support the algebra but are not substituted for the proof.

The printed Appendix G derivation remains invalid because its coordinate union
leaves a `sqrt(K)` factor and omits target-bin-mass estimation. The verified
contract therefore relies on the necessary assumptions stated in
`claim_contract.json`: iid evaluation data, fixed posterior and assignment
functions, positive realized denominators, and `K >= 2`.
