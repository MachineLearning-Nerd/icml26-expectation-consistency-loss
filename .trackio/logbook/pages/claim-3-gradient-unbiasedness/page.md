# Claim 3 — Mini-batch gradient unbiasedness

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c3_grad_01", "created_at": "2026-07-19T14:20:05+00:00", "title": "Theorem 3.3 exact audit"}
-->
**Current anchored claim:** Theorem 3.3 says the auxiliary-variable Eq. 10
mini-batch formulation is asymptotically equivalent to differentiable Eq. 8 and
has an unbiased gradient, enabling ordinary SGD.

**Assessment: falsified as stated.** Four independent exact failures occur in
Appendix H, while a narrower corrected linear estimator is unbiased under explicit
fixed-state and independence conditions.

1. For uniform mini-batches, Appendix H equates a `1/|D_m|` weighted sum to a
   `1/n_j` full weighted mean but omits `N/n_j`. With valid soft assignments, the
   printed expectation is `3/2`, the full gradient is `3`, and the required scale is `2`.
2. A self-normalized one-item batch has expected absolute loss `1/4` versus full loss
   `1/8`; selecting the norm direction from that same batch gives expected gradient
   `0` versus the full gradient `1`.
3. Soft assignments, counts, denominators, and outer bin weights depend on `theta`,
   but Eq. 32 retains only posterior derivatives. A valid interior weight path gives
   true quotient derivative `3/4` versus printed derivative `0`.
4. Eq. 10 retains within-bin squared-residual variance, so it is not Eq. 8 after
   profiling the auxiliaries.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c3_grad_02", "created_at": "2026-07-19T14:20:06+00:00", "title": "Executed exact counterexamples", "command": [".venv/bin/python", "repro/src/claim3_gradient_certificate.py", "--output", "outputs/claim3_gradient_certificate.json"], "exit_code": 0}
-->
````bash
$ .venv/bin/python repro/src/claim3_gradient_certificate.py \
    --output outputs/claim3_gradient_certificate.json
````

````output
anchored claim assessment: contradicted_as_stated
corrected fixed-direction estimator: verified_with_explicit_conditions_and_corrected_scaling
Eq. 10 objective parity: falsified_by_exact_within_bin_variance_counterexample
all audit gates pass: True
certificate sha256: c7dab6e2caa93bc9c6a23ef081fddc0fbfcffc8aba14e2cecb700ddf63bd80b0
````

For the decisive one-bin Eq. 10 witness, Eq. 8 loss/gradient are `3/4` and `0`,
whereas the exact profiled Eq. 10 loss/gradient are `65/128` and `1/4`. Replicating
each observation 64 times produces Eq. 10 loss `319/256` and gradient `16`, so the
omitted variance term does not vanish under this asymptotic family.

The official code computes auxiliaries from the current batch, detaches them, and
reuses them on that same batch. Detachment blocks autograd but does not create
statistical independence; soft assignment weights still carry gradients. The audit
does not claim the heuristic is empirically ineffective—it falsifies the published
Eq. 8/Eq. 10 identity and unrestricted unbiased-gradient theorem. Full derivation:
`docs/CLAIM3_GRADIENT_AUDIT.md`.
