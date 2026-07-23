# Claim 1 — Expectation-consistency theorem

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c1_general", "created_at": "2026-07-19T14:20:00+00:00", "title": "General proof and exact certificate"}
-->
**Current anchored claim:** Theorem 3.1 establishes that, under covariate shift, the
source and target class-`k` calibration curves agree if and only if the corresponding
conditional expectations of the shared posterior agree given the score `S`.

**Assessment: verified with support/version qualifications.** For either domain `d`,
`S=f(X)` and the shared conditional kernel `q_k(x)=P(Y_k=1|X=x)` give

`E_d[Y_k|S] = E_d[E_d[Y_k|X]|S] = E_d[q_k(X)|S]`.

The calibration-curve gap and EC residual are therefore the same conditional random
variable. Their zero sets coincide on common `S` support. This is an analytical proof,
not an extrapolation from a small simulation.

The executable exact-rational certificate compares the two formal coefficient maps for
every posterior value, then checks a deterministic `257 X × 17 S × 11 classes`
construction: all `374` Eq. 13 components and `187` cross-domain components agree
exactly. Certificate digest:
`2d60d40d6458fe8ad6339054db3096588f83122a0b5ab9804dcda38e90a3342b`.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c1_run", "created_at": "2026-07-19T14:20:01+00:00", "title": "Executed exact theorem audit", "command": [".venv/bin/python", "repro/src/claim1_general_certificate.py", "--output", "outputs/claim1_general_certificate.json"], "exit_code": 0}
-->
````bash
$ .venv/bin/python repro/src/claim1_general_certificate.py \
    --output outputs/claim1_general_certificate.json
````

````output
formal all-q coefficient maps equal: True
exact Fraction checks: Eq13=374, cross-domain=187
literal theorem assessment: verified_with_support_and_version_qualifications
stronger EC-alone absolute-calibration reading: falsified_without_source_calibration_premise
all certificate gates pass: True
````

Controls are decisive: removing `S=f(X)` produces residual `1/2`; breaking the shared
conditional gives EC residual `0` but true gap `-1/2`; disjoint score support is rejected
as unidentified. EC alone transfers a calibration curve—it does not make that curve
equal `S`. Source calibration is the additional premise needed for target calibration.

All conditional equalities are almost-everywhere statements on comparable common score
support. Full derivation: `docs/CLAIM1_GENERAL_PROOF_AUDIT.md`.
