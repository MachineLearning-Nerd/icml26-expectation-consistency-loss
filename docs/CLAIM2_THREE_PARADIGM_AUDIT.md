# Claim 2 - Exact three-paradigm ECL compatibility audit

## Outcome

**Local assessment: full verification of the stated compatibility claim at the
population/hard-group formula level.** The certificate separately executes the
canonical, class-wise, and top-label constructions and proves, with exact rational
arithmetic, that each ECL source-target residual is the corresponding direct
calibration gap. This is a theorem-level result; it is not evidence that ECL improves
accuracy or calibration on the paper's image benchmarks.

The approach is materially stronger and different from the retained toy check. The
old check sampled 20 random six-state floating-point instances, changed one summary,
and covered only a class-wise alternative. This attempt uses a clean-room exact
three-class construction with 12 common-support atoms, distinct source and target
covariate distributions, separate grouping and observation semantics for all three
paradigms, zero and nonzero certificates, four adversarial controls, and a semantic
audit of every mode in the pinned official source.

## Claim and paper anchors

- Challenge claim: "ECL is compatible with canonical calibration, class-wise
  calibration, and top-label calibration."
- Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under
  Covariate Shift*, OpenReview `gFPPTokv9C`.
- Local PDF: `repro/evidence/claim3/2605.21552v1.pdf`.
- PDF SHA256:
  `fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`.
- Canonical: Theorem 3.1 and Eq. 4.
- Top-label: Appendix D Theorem D.1/Eqs. 15-16, Appendix E Eqs. 20-21,
  Appendix F Eqs. 24-25.
- Class-wise: Appendix D Theorem D.2/Eqs. 17-19, Appendix E Eqs. 22-23,
  Appendix F Eqs. 26-29.

The PDF was read using Poppler text extraction and visually checked after rendering
PDF pages 4-5 and 13-15. The rendered pages confirm that canonical calibration groups
the full vector `S`, class-wise calibration separately groups each `S_k`, and top-label
calibration groups `S_hat=max_k S_k` while observing the correctness event
`Y*=Y_hat`.

## Independent derivation

Let `T=T(X)` be a deterministic summary and let `a(X)=P(A|X)` be the relevant
posterior observable. On either domain `d` and every positive-mass summary value
`t`, the law of total probability gives

```text
P_d(A | T=t)
  = sum_{x:T(x)=t} P_d(X=x) P_d(A|X=x) / P_d(T=t)
  = E_{X~P_d(X|T=t)}[a(X)].
```

Under covariate shift the posterior observable is shared. Subtracting the target
identity from the source identity makes the ECL residual *identically equal* to the
direct source-target calibration gap. The three constructions are not interchangeable:

| Paradigm | Exact summary `T(X)` | Exact observable `a(X)` |
|---|---|---|
| Canonical | complete confidence vector `S(X)` | complete vector `P(Y|X)` |
| Class-wise | scalar `S_k(X)`, separately for each `k` | scalar `P(Y_k=1|X)` |
| Top-label | `S_hat(X)=max_k S_k(X)` | `P(Y=Y_hat(X)|X)=q_{Y_hat(X)}(X)` |

For top-label calibration, `Y_hat(X)` comes from the fixed classifier score vector.
It is incorrect to replace the event posterior with `max_k P(Y=k|X)`; the posterior's
argmax need not equal the fixed classifier's predicted class.

## Executed exact construction

The implementation uses Python `fractions.Fraction`; no scientific value is converted
to floating point. It defines six unique three-class score vectors and duplicates each
one, producing 12 covariate atoms. Every atom has positive probability in both domains.
Within each duplicate pair, source masses are `(1/24, 3/24)` and target masses are
`(3/24, 1/24)`, so `P_s(X) != P_t(X)` while support and `P(Y|X)` are shared.

Two fixtures are evaluated:

1. **Calibrated zero-loss fixture:** `P(Y|X)=S(X)`. All three exact ECL losses are
   zero, and the direct calibration gap is also zero at every correct summary value.
2. **Diagnostic nonzero fixture:** duplicate posteriors are
   `S +/- (1/12, -1/24, -1/24)`. Every correct construction detects a nonzero gap,
   and the independently assembled joint-probability calibration tables match the
   ECL residuals exactly.

Observed target-weighted L1 losses are:

| Paradigm | Calibrated fixture | Diagnostic fixture | Exact identity |
|---|---:|---:|---|
| Canonical | `0` | `1/6` | ECL residual = direct vector calibration gap |
| Class-wise, all 3 classes | `0` | `1/6` | ECL residual = direct scalar gap for every `k,S_k` |
| Top-label, 3 predicted classes/2 confidences | `0` | `1/48` | ECL residual = direct correctness gap |

## Negative controls

All controls are constructed so an incorrect implementation must be rejected.

| Control | Correct result | Deliberately wrong result | Outcome |
|---|---:|---:|---|
| Canonical grouped by complete `S` vs. incorrectly by `max(S)` | loss `1/12`, 6 states | loss `1/18`, 2 states | rejected |
| Class-1 posterior grouped by `S_1` vs. incorrectly by `S_0` | loss `1/72` | loss `1/48` | rejected |
| Top-label event `q_{Y_hat}` vs. incorrectly using `max(q)` | `1/5` | `7/10` | rejected by exact counterexample |
| Break shared posterior while reusing source `q` on the theorem side | exact identity should fail | fails for canonical, every class-wise coordinate, and top-label | rejected |

The top-label semantic counterexample uses score `(3/5,1/5,1/5)` and posterior
`(1/5,7/10,1/10)`: the fixed classifier predicts class 0, so the correctness-event
posterior is `1/5`, even though the largest posterior coordinate is `7/10`.

## Official source audit

- Repository: <https://github.com/NeuroDong/ECL>
- Pinned commit: `aae77f890f1e4ebc13dad135b5e29758d98d318d`
- Vendored file: `repro/evidence/claim3/official_losses.py`
- Vendored SHA256:
  `1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754`

The exact pin check and semantic source audit both pass:

- `TopLabel` has scalar `B`-bin caches, groups classifier confidence with
  `train_probs.max`, and uses a binary auxiliary-head probability of correctness.
- `Classwise` has `K x B` caches, loops over every class, groups by
  `train_probs[:,k]`, and aggregates auxiliary posterior coordinate `k`.
- `Canonical` has `B x K` caches, groups by full-vector distance to simplex anchors,
  and aggregates full auxiliary posterior vectors.

This audit establishes that the official implementation preserves the three distinct
semantics. Its returned training objective is the paper's auxiliary/proximal mini-batch
objective; the exact certificate here evaluates the population and hard-group formulas,
not numerical equivalence of that training surrogate.

## Commands actually executed

All commands ran from the repository root on local CPU only:

```bash
shasum -a 256 repro/evidence/claim3/2605.21552v1.pdf
pdftotext -layout repro/evidence/claim3/2605.21552v1.pdf /tmp/gfp-paper-claim2.txt
pdftoppm -f 4 -l 5 -png -r 130 repro/evidence/claim3/2605.21552v1.pdf tmp/pdfs/claim2/main
pdftoppm -f 13 -l 15 -png -r 130 repro/evidence/claim3/2605.21552v1.pdf tmp/pdfs/claim2/appendix
hf auth whoami --format json
hf version --format json
git ls-remote https://github.com/NeuroDong/ECL.git HEAD
curl -fsSL https://raw.githubusercontent.com/NeuroDong/ECL/aae77f890f1e4ebc13dad135b5e29758d98d318d/losses.py | shasum -a 256
uv run python -m py_compile repro/src/claim2_compatibility_certificate.py
/usr/bin/time -p uv run python repro/src/claim2_compatibility_certificate.py
/usr/bin/time -p uv run pytest -q repro/tests/test_claim2_compatibility_certificate.py
/usr/bin/time -p uv run pytest -q
git diff --check
```

Observed certificate stdout:

```text
Claim 2 exact three-paradigm certificate
  exact calibrated zero-loss fixture: True
  exact diagnostic nonzero fixture: True
  diagnostic canonical ECL L1 loss: 1/6
  diagnostic classwise ECL L1 loss: 1/6
  diagnostic toplabel  ECL L1 loss: 1/48
  negative controls all rejected: True
  official source audit: True
  all success criteria: True
```

Repeated scientific-calculation runs took `0.16-0.20 s` wall time. The scoped test
run reported `15 passed in 0.39s` and took `1.53 s` wall time. The current full
repository suite reported `71 passed in 3.04s` and took `3.50 s` wall time.
`git diff --check` produced no output and exited successfully.

## Environment, inputs, and cost

- Python `3.12.11` via `uv`.
- NumPy `2.5.1` and SciPy `1.18.0` are installed for the wider project, but this
  certificate uses only the Python standard library.
- Platform: macOS `26.5.2`, arm64, local CPU only.
- Hugging Face CLI `1.24.0`; authenticated readback user was `DineshAI`.
- Random seeds: none; the computation is deterministic exact arithmetic.
- Remote or paid compute: none. Estimated cost: `$0`.
- No model, dataset, checkpoint, network inference, or stochastic sampling is used.

## Evidence and rerun artifacts

- Implementation: `repro/src/claim2_compatibility_certificate.py`
- Regression tests: `repro/tests/test_claim2_compatibility_certificate.py`
- Machine-readable exact evidence:
  `outputs/claim2_three_paradigm_certificate.json`
- This audit: `docs/CLAIM2_THREE_PARADIGM_AUDIT.md`

Artifact SHA256 values from the executed version:

```text
763bb648cf11a13312c4a624e288bf7e08b58f93df4789a178a74a1769ccda82  repro/src/claim2_compatibility_certificate.py
af33e5c67c95c4ea7f13bc394852ad7a0a3f5e5aed4b576bb657beb8f99243e2  repro/tests/test_claim2_compatibility_certificate.py
f4523356a68e8818976947d9151ffc0081c5bccafcf64030318227686726d777  outputs/claim2_three_paradigm_certificate.json
```

## Limitations and deviations

- This verifies the mathematical compatibility claim and hard-group construction,
  not benchmark-scale image training, empirical performance improvement, or the
  quality of a learned `P(Y|X)` auxiliary head.
- It does not numerically reproduce soft-anchor Eqs. 25/27-29 or the proximal
  mini-batch optimizer. The source audit checks their grouping/observable semantics.
- The exact finite support is a certificate, not a proof assistant formalization.
  The general derivation above supplies the universal law-of-total-probability step;
  executable fixtures guard the three specializations and failure modes.
- The paper does not specify a unique norm in the population claim. The certificate
  uses target-weighted L1, matching the pinned official default `reduction="l1"`.
- No attempt is made to re-evaluate Claim 1 or Claim 3.

## Recommended integration

Replace the current Claim 2 toy/partial logbook text with a concise version of this
audit, the exact command, the `0 / {1/6,1/6,1/48}` result table, the four negative
controls, and the official-source pin. Add the implementation, test, JSON evidence,
and this audit to the public reproduction bundle. Describe the result as full
verification of *formula-level three-paradigm compatibility*, while retaining the
explicit boundary that no benchmark training or performance claim was tested.

This counts as **one substantive scientific attempt**: it reaches the claimed
mathematical computation, is materially different from the earlier random float
check, independently reconstructs both sides of every identity, and includes
falsifiable negative controls.
