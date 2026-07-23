# Claim 1 general proof audit

Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* (`gFPPTokv9C`).

## Assessment

- Literal Theorem 3.1: **verified with support and conditional-version qualifications**.
- Stronger reading, “Expectation consistency alone implies absolute calibration”: **falsified unless source calibration is assumed**.
- Toy / inconclusive: **no** for the literal theorem. This is an analytical reproduction with an exact formal-coefficient certificate and assumption-breaking controls, not another sampled six-state experiment.
- Substantive scientific attempt: **yes**. It independently rederives the theorem, identifies its minimum assumptions, proves both directions at once, and finds the boundary at which the informal calibration interpretation becomes false.

The paper PDF is pinned at SHA-256
`fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`.
The audited anchors are Section 3.1, Theorem 3.1 (PDF page 3), and Appendix B,
Eqs. 13-14 (PDF page 12). Both pages were rendered and visually inspected in addition to text extraction.

## Why this is materially different from the retained evidence

The retained verifier chooses 40 random floating-point models with `nX=6` and observes zero numerical residual. That is a useful implementation check but does not independently establish a quantified theorem.

This attempt instead has three layers:

1. A measure-theoretic tower-property proof for general random variables.
2. A formal linear-form certificate. For fixed but arbitrary exact source/target masses, it compares the coefficients of every symbolic posterior value `q_k(x)`; it therefore certifies the iff for **all** posterior values, rather than for sampled numeric values.
3. An independent exact-rational joint-distribution construction on 257 X states, 17 S levels, and 11 classes, plus counter-controls that deliberately remove one theorem assumption at a time.

All executable probability arithmetic uses Python `fractions.Fraction`; there is no floating-point tolerance or random seed.

## Independent proof

Let `d` denote source or target, let `Y_k` be the class indicator, and define

`C_d,k = E_d[Y_k | S]`.

The paper defines the classifier output as `S=f(X)` in Section 2. Consequently,
`sigma(S) subset sigma(X)`. The tower property gives, domain by domain,

`C_d,k = E_d[E_d[Y_k | X] | S]`.

This is the rigorous form of Appendix B Eq. 13. The appendix describes the relevant step as “X contains all the information that S can provide.” Determinism of `S=f(X)` is sufficient; the more general minimum condition is

`E_d[Y_k | X,S] = E_d[Y_k | X]`.

Under covariate shift, take a single regular-conditional kernel

`q_k(x) = P_s(Y_k=1|X=x) = P_t(Y_k=1|X=x)`

that is a version for both domains on the union of their X supports. Then

`C_d,k = E_d[q_k(X) | S] = M_d,k`.

It follows immediately that

`C_s,k - C_t,k = M_s,k - M_t,k`.

The left and right differences are the same conditional random variable (after choosing comparable versions), so

`C_s,k = C_t,k` iff `M_s,k = M_t,k`.

This proves both directions; no asymptotics, approximation, or empirical premise is involved. The requirement `P_s(X) != P_t(X)` is not used algebraically—the identity also holds in the no-shift special case—but the executed positive witnesses do use genuinely different marginals.

## Support and version qualifications

The PDF writes the conditional equalities pointwise in `S`. For continuous S, regular conditional probabilities are defined only almost everywhere and may be changed arbitrarily at null values. A cross-domain equality is therefore meaningful only where both domains' conditional versions can be compared.

A clean sufficient statement is:

- X and S are standard Borel (they are Euclidean/simplex-valued in the paper);
- a common conditional kernel `q(y|x)` is a version in both domains;
- `P_s^S` and `P_t^S` are mutually absolutely continuous; and
- all equalities hold almost everywhere under their common S-support measure.

One may instead restrict the statement to explicitly selected conditional kernels on shared S support. No theorem conclusion is identified at an S value that has zero probability in one domain. The executable disjoint-support control reports this case rather than silently filling a zero-mass conditional with zeros, which the retained six-state helper does.

`S=f(X)` is not missing from the paper as a whole: it is defined in Section 2. It is, however, only described informally inside Appendix B. The exact counter-control `X=constant`, `S=Y`, `Y~Bernoulli(1/2)` shows why the condition matters: at `S=1`,

`P(Y=1|S=1)=1`, while `E[P(Y=1|X)|S=1]=1/2`.

Thus Eq. 13 fails if S is allowed to carry label information beyond X.

## What “calibration” requires

The literal theorem equates the source and target **calibration curves**. It does not, by itself, say either curve equals the classifier score.

To derive target perfect calibration, add the source-calibration premise

`P_s(Y_k=1|S=s)=s_k`.

Then expectation consistency is necessary and sufficient for

`P_t(Y_k=1|S=s)=s_k`

on common support. This is the premise used in the paper's remark (“the source domain can usually be easily calibrated well”), but it is not a premise of the displayed Theorem 3.1 equality itself.

The exact negative control sets `S_k=3/4` and a shared posterior mean `1/4` under different source and target X marginals. EC holds and both calibration curves equal `1/4`, but both domains are miscalibrated because `1/4 != 3/4`. Hence any stronger “EC alone creates absolute calibration” reading is false.

## Executed certificate and controls

The machine-readable result is [`outputs/claim1_general_certificate.json`](../outputs/claim1_general_certificate.json).

Observed exact results:

- all 17 formal source/target coefficient maps agree between the joint-probability and tower-property derivations;
- equality of those linear forms certifies identical zero sets for every `q_k(x)`;
- 374 exact Eq. 13 class/bin/domain component comparisons pass;
- 187 exact cross-domain gap/EC comparisons pass;
- the large certificate uses 257 X states, 17 S levels, and 11 classes;
- the exact certificate digest is `2d60d40d6458fe8ad6339054db3096588f83122a0b5ab9804dcda38e90a3342b`;
- a genuine covariate-shift witness has EC and calibration-curve gaps exactly zero;
- a second witness has matching nonzero EC/calibration gaps of magnitude `1/2`;
- removing `S=f(X)` produces an Eq. 13 residual of exactly `1/2`;
- removing the shared conditional kernel makes a source-kernel EC residual `0` while the true cross-domain calibration-curve gap is `-1/2`;
- disjoint S support is diagnosed as non-identifiable, not counted as an iff success; and
- EC without source calibration leaves both domains exactly miscalibrated.

## Commands, environment, and runtime

Commands actually used for the final executable verification:

```bash
env PYTHONHASHSEED=0 .venv/bin/python repro/src/claim1_general_certificate.py \
  --output outputs/claim1_general_certificate.json
env PYTHONHASHSEED=0 .venv/bin/python -m pytest -q \
  repro/tests/test_claim1_general_certificate.py
env PYTHONHASHSEED=0 .venv/bin/python -m pytest -q repro/tests
```

Paper inspection also used `pdfinfo`, `pdftotext -layout`, and `pdftoppm -png -r 144` on the pinned PDF. The system has no unqualified `python` executable, so all successful reproduction commands use the repository's `.venv/bin/python` explicitly. No Docker Compose file is present.

Environment recorded by the runner:

- CPython 3.12.11;
- macOS 26.5.2, arm64, Apple M2;
- Python standard library `fractions.Fraction`; and
- local CPU only, no remote or paid compute.

Observed `/usr/bin/time -p` results:

- certificate: real `0.26 s`, user `0.22 s`, sys `0.03 s`;
- 9 scoped tests: real `1.20 s`, `9 passed in 0.88 s`; and
- all repository tests: real `2.92 s`, `56 passed in 2.58 s`.

Captured outputs:

- [`outputs/claim1_stdout.txt`](../outputs/claim1_stdout.txt)
- [`outputs/claim1_tests_stdout.txt`](../outputs/claim1_tests_stdout.txt)
- [`outputs/claim1_full_tests_stdout.txt`](../outputs/claim1_full_tests_stdout.txt)

## Limitations and deviations

- The executable coefficient/rational certificate is finite. The tower-property derivation above, rather than increasing the finite state count, is what establishes the general measurable-space result.
- Conditional equalities remain version-dependent on null S events. The audit intentionally does not claim a pointwise result there.
- The common conditional kernel is a semantic assumption. It cannot be verified from unlabeled target observations alone.
- The approach reproduces Theorem 3.1, not ECL optimization, learned-posterior accuracy, or the empirical dataset results.
- Official source is pinned as `NeuroDong/ECL@aae77f890f1e4ebc13dad135b5e29758d98d318d`, but no official-code execution is needed for this analytical theorem claim.

## Recommended integration

Replace the Claim 1 logbook's “exact six-state verifier” headline with this proof audit while retaining the old numerical experiment as a regression sanity check. The logbook should report:

- `verified` for the literal Theorem 3.1 equality on common S support;
- the exact formal/rational certificate and assumption-breaking controls;
- the almost-everywhere/common-kernel qualifications; and
- the explicit source-calibration premise needed to turn curve equality into target perfect calibration.

Do not state that EC alone creates absolute calibration, and do not claim conclusions at source-only or target-only confidence values.
