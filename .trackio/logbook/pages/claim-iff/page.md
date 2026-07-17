# Claim 1+2 — Calibration-transfer iff + compatibility

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ec1_01", "created_at": "2026-07-17T19:41:00+00:00", "title": "Claim & method"}
-->
**C1 (Theorem 3.1, iff).** Under covariate shift (`P_s(X)≠P_t(X)`, shared `P(Y|X)`, `S=f(X)` ⟹
`Y⊥S|X`):
`P_s(Y_k=1|S)=P_t(Y_k=1|S)` ⟺ `E_{P_s(X|S)}[P(Y_k=1|X)]=E_{P_t(X|S)}[P(Y_k=1|X)]`.
Engine: `P(Y_k=1|S)=E_{X|S}[P(Y_k=1|X)]` (law of total probability; `Y⊥S|X`).

**C2 (compatibility, Theorems D.1/D.2).** Same iff with `S` replaced by top-label `Ŝ` or
class-wise `S_k` — the structure is identical (any summary `S=f(X)`).

Verified by exact discrete enumeration over finite covariate-shift joints.

---
<!-- trackio-cell
{"type": "code", "id": "cell_ec1_02", "created_at": "2026-07-17T19:41:10+00:00", "title": "Iff verifier", "command": ["python", "repro/src/run_ecl.py"], "exit_code": 0}
-->
````bash
$ python repro/src/run_ecl.py
````
- **iff identity:** EC residual == calibration gap, max|diff| = **0.00e+00** over 40 instances;
  EC=0 ⟺ gap=0 consistent on all 40. ✓
- **engine:** `P(Y=1|S) == E_{X|S}[P(Y=1|X)]` on both domains (40 checks). ✓
- **C2 compatibility:** iff holds for alternative class-wise summary `S_k` (20 instances). ✓
- **negative control:** break covariate shift → iff fails (20/20). ✓

**=> C1 + C2 VERIFIED.** Evidence: `outputs/ecl_summary.json`.
