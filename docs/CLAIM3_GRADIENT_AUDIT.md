# Anchored Claim 3 Gradient Audit

Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* (`gFPPTokv9C`, arXiv `2605.21552v1`).

Claim audited exactly:

> Theorem 3.3 shows the mini-batch ECL formulation using auxiliary variables `u_j^s` and `u_j^t` yields an unbiased gradient estimator, `E[grad_theta Lhat_ecl^mini] = grad_theta Lhat_ecl`, enabling standard SGD training.

Paper anchors: Section 3.5, Theorem 3.3 and Eq. 10; Appendix H, especially Eqs. 32-33. The paper PDF has SHA-256 `fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`.

## Verdict

`contradicted_as_stated`

The exact unrestricted identity in Claim 3 does not hold for Eq. 10 or Algorithm 1. A narrower linear estimator is unbiased only after adding conditions and a normalization absent from Appendix H: the posterior estimator, weights/counts, auxiliaries, and norm direction must be fixed independently of the current batch; the finite-population scaling must be correct; and all derivatives through soft weights and denominators must either be included or those quantities must be frozen.

This is a theorem audit, so one exact counterexample is decisive. The executable certificate provides four independent counterexamples plus a positive certificate for the restricted corrected estimator. Recommended leaderboard verdict: `not_reproduced__theorem_3_3_contradicted_as_written`.

## What is valid

For one finite-data bin, let

`G_full = w_j <v_j, (1/n_sj) sum_i omega^s_ij grad p_i - (1/n_tj) sum_i omega^t_ij grad p_i>`.

Conditional on all of the following, linearity of expectation does give an unbiased mini-batch estimator:

- `p`, `omega`, `n_dj`, `w_j`, the auxiliaries, and the norm direction `v_j` are fixed before sampling the current batch;
- `v_j` is not selected from the same batch whose gradients it multiplies;
- uniform finite-population sampling uses the factor `N_d/n_dj` in front of the mini-batch average `|D_d^m|^-1 sum omega_ij grad p_i`, or an equivalent inclusion-probability correction;
- for a population expectation, differentiation may be interchanged with expectation, for example via an integrable dominating derivative;
- away from a zero norm difference an ordinary gradient exists; at zero, a measurable compatible subgradient selection must be specified.

The exact positive certificate uses valid Eq. 6 weights `(1/4,3/4)` for the audited bin, derivatives `(0,4)`, population size `2`, and batch size `1`. The complementary-bin weights are `(3/4,1/4)`, so each sample's two assignments lie in `[0,1]` and sum to one. The full weighted gradient is `3`. Appendix H's displayed unscaled batch estimator has expectation `3/2`; multiplying by the missing factor `N/n_j = 2` gives expectation exactly `3`.

This positive result is a fixed-direction linear estimator. It is not the gradient of the squared-residual Eq. 10 objective returned by the official implementation.

## Why the printed proof does not establish the claim

### 1. Appendix H has a soft-count normalization mismatch

For a uniform batch of size `m` from `N` observations,

`E[(1/m) sum_(i in batch) omega_i grad p_i] = (1/N) sum_i omega_i grad p_i`,

not `(1/n_j) sum_i omega_i grad p_i` where `n_j=sum_i omega_i`. The displayed transition on Appendix-H page 17 requires the missing factor `N/n_j`. The exact witness above gives `3/2 != 3` before correction.

### 2. Self-normalization and same-batch norm directions are biased

Eq. 7/8 uses a ratio `sum omega_i p_i / (sum omega_i + epsilon)`. Even if the numerator and denominator are separately unbiased, their ratio is not generally unbiased.

At `theta=0`, take source probabilities `(1/4+theta, 3/4+theta)`, valid source bin weights `(1/4,3/4)` with complementary-bin weights `(3/4,1/4)`, and target probability `1/2`. The full source weighted mean is `5/8`, so the full absolute ECL loss is `1/8` and its gradient is `1`. With a uniform one-item batch:

- expected mini-batch absolute loss is `1/4`, not `1/8`;
- expected gradient using the current batch's sign/norm direction is `0`, not `1`;
- using the independent full-data direction and the corrected `N/n_j` scaling restores gradient expectation `1`.

Thus unbiased loss estimation, unbiased ratio estimation, and unbiased gradient estimation are separate statements. Detaching a same-batch quantity from autograd does not make it statistically independent of that batch.

### 3. Eq. 32 omits derivatives of the soft assignments and denominators

The paper's soft weights depend on `S=f_theta(X)` through Eq. 6, the soft count `n_dj=sum_i omega_ij` depends on `theta`, and the outer target-bin weight `w_j` also depends on the target soft counts. Nevertheless, Appendix H writes

`grad Ehat_s,j = (1/n_sj) sum_i omega^s_ij grad p_i`,

which retains none of `grad omega`, `grad n_sj`, `grad w_j`, or the derivative of the stabilizer-normalized quotient.

The exact quotient witness uses the audited-bin weights `omega_1(theta)=1/4+theta` and `omega_2(theta)=3/4`, with complementary-bin weights `3/4-theta` and `1/4`, and fixed posteriors `p_1=1`, `p_2=0`. At `theta=0`, both samples' assignment vectors are strictly interior to the two-bin simplex. Every assignment remains strictly inside the simplex throughout `|theta| <= 1/8`. The full weighted mean is `1/4` and its true derivative is `3/4`; the posterior-derivative-only expression is `0`.

### 4. Eq. 10 is not Eq. 8 with auxiliary variables

For one scalar bin, Eq. 10 contains

`sum_i (u_s-p_i)^2 + sum_i (u_t-p_i)^2`.

Writing `mu_d` for a domain mean gives the exact decomposition

`sum_i (u_d-p_i)^2 = sum_i (p_i-mu_d)^2 + n_d (u_d-mu_d)^2`.

Appendix H discards the first term when substituting the auxiliary minimizer. That term is within-bin posterior variance, depends on `theta`, and is absent from Eq. 8. The L1 coupling also means the finite-sample auxiliary minimizers are shifted away from the empirical means.

Exact counterexample at `theta=1/16`:

- source probabilities: `(15/16,13/16)`, with derivatives `(1,-1)`;
- target probabilities: `(1/8,1/8)`, with derivatives `(0,0)`;
- one bin with `w=1`;
- exact Eq. 10 minimizer: `u_s=5/8`, `u_t=3/8`;
- Eq. 8 loss: `3/4`; profiled Eq. 10 loss: `65/128`;
- Eq. 8 gradient: `0`; fixed-auxiliary/envelope Eq. 10 gradient: `1/4`.

Replicating every observation does not repair the claimed asymptotic equivalence. At replication `64`, Eq. 8 remains `3/4` with gradient `0`, while profiled Eq. 10 is `319/256` with gradient `16`. The unnormalized within-bin variance term grows with the replicated sample count.

### 5. The gradient written in Appendix H is not the gradient of Eq. 10

With auxiliaries held fixed, differentiating Eq. 10's squared residuals gives terms of the form

`2 omega_i <p_i-u_dj, grad p_i> + (grad omega_i) ||u_dj-p_i||^2`.

Appendix H instead writes a fixed-direction expression of the form

`w_j <v_j, average(omega_i grad p_i)>`.

The paper does not derive an equality between these expressions, and the exact Eq. 10 counterexample shows that their targets differ.

## Official-source audit

The vendored `upstream/losses.py` is byte-identical to the independently read-back `NeuroDong/ECL@aae77f890f1e4ebc13dad135b5e29758d98d318d` file, SHA-256 `1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754`.

- `upstream/losses.py:225-237` constructs differentiable current-batch soft weights and statistics.
- `upstream/losses.py:257-275` computes current-batch proximal auxiliaries and then detaches them.
- `upstream/losses.py:282-284` reuses those auxiliaries on the same current batch in a squared-residual loss. The same structure occurs for class-wise and canonical modes at lines `341-378` and `412-455`.
- `upstream/main.ipynb:346` separately trains `classifier2`; `upstream/main.ipynb:505` later optimizes only `model.fc2.parameters()`.
- In that final phase, the posterior head output is fixed with respect to the `fc2` optimizer, but the soft assignments depend on `fc2` logits and are not detached. Consequently the official loss can have nonzero `fc2` gradients through `omega`, exactly the derivative path omitted by Appendix H's `omega*grad(p)` calculation.

The code implements a useful stateful proximal/EMA training heuristic. It does not implement the independent fixed-direction estimator for which the corrected expectation identity was verified.

## Posterior-estimator and gradient-interchange conditions

- A learned `P-hat(Y|X)` must be trained on independent data or frozen before the audited batch for a conditional-unbiasedness statement. If it is estimated from the same batch, its dependence must be analyzed explicitly.
- A slow running average is independent of a fresh current batch only when it contains past data exclusively. Algorithm 1 first updates the auxiliary from the current batch and then uses that auxiliary on the same batch; `.detach()` blocks derivatives but does not create statistical independence.
- If the score-dependent assignments are frozen, the audited gradient is for that frozen-assignment surrogate, not the full differentiable Eq. 8 objective.
- For true population expectations, a valid Leibniz/dominated-differentiation condition is needed. Finite-data exact sums need no interchange theorem.
- At zero ECL difference the norm is nonsmooth, so the statement must use subgradients and coordinate their selections between the full and mini-batch objectives.

## Probability-domain audit

Every exact witness is checked against its probability domain:

- normalization witness: two-bin assignments are `(1/4,3/4)` and `(3/4,1/4)`; the posterior path starts at `(1/2,1/2)` with derivatives `(0,4)` and remains strictly inside `[0,1]` for the certified two-sided radius `1/16`;
- self-normalization witness: the same strictly interior assignment rows are used, while posterior paths `(1/4+theta,3/4+theta)` remain interior for the certified radius `1/8`;
- soft-weight derivative witness: assignment rows remain strictly inside their probability simplices for every `|theta| <= 1/8`; fixed posteriors `(1,0)` stay valid probabilities;
- Eq. 10 witness: all evaluated posteriors `(15/16,13/16,1/8,1/8)` are strictly interior, their affine paths remain interior for radius `1/32`, and its single-bin assignment weight is exactly `1`.

The generated report exposes these checks under `probability_domain_audit`, and the test suite requires every gate to pass.

## Reproduce

```bash
.venv/bin/python -m pytest repro/tests/test_claim3_gradient_certificate.py -q
.venv/bin/python repro/src/claim3_gradient_certificate.py --output outputs/claim3_gradient_certificate.json
```

Expected scoped test result: `12 passed`.

Artifacts:

- `repro/src/claim3_gradient_certificate.py`
- `repro/tests/test_claim3_gradient_certificate.py`
- `outputs/claim3_gradient_certificate.json`
- `outputs/claim3_gradient_stdout.txt`

## Scope and limitations

- This audit evaluates Theorem 3.3's mathematical identity and the pinned source structure; it does not rerun the paper's calibration benchmarks.
- The project dependencies intentionally exclude PyTorch. The source audit is static, while every counterexample and positive estimator check is exact standard-library rational arithmetic.
- The verdict does not claim the proximal/EMA heuristic is empirically ineffective. It says the published Eq. 8 unbiased-gradient theorem does not justify that heuristic as written.
