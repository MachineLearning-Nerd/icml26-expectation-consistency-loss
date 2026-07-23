# Claim 4 — Canonical, class-wise, and top-label ECL

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c4_01", "created_at": "2026-07-19T14:20:03+00:00", "title": "Exact three-paradigm certificate"}
-->
**Current anchored Claim 4 / legacy Claim 2:** ECL extends to canonical, class-wise,
and top-label calibration, including the distinct groupings and posterior observables
defined in Appendix D–F.

**Assessment: verified at the population/hard-group formula level.** A clean-room
`Fraction` certificate separately executes:

| Paradigm | Grouping | Observable | Zero fixture | Diagnostic loss |
| --- | --- | --- | ---: | ---: |
| Canonical | complete vector `S` | complete posterior vector | `0` | `1/6` |
| Class-wise | each scalar `S_k` | `P(Y_k=1|X)` for all 3 classes | `0` | `1/6` |
| Top-label | confidence `max(S)` | `P(Y=argmax(S)|X)` | `0` | `1/48` |

The source and target distributions share all 12 covariate atoms and one posterior
kernel but assign different exact masses. In every summary state, the independently
assembled direct calibration gap equals the ECL residual exactly.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c4_02", "created_at": "2026-07-19T14:20:04+00:00", "title": "Executed three-paradigm audit", "command": [".venv/bin/python", "repro/src/claim2_compatibility_certificate.py", "--output", "outputs/claim2_three_paradigm_certificate.json"], "exit_code": 0}
-->
````bash
$ .venv/bin/python repro/src/claim2_compatibility_certificate.py \
    --output outputs/claim2_three_paradigm_certificate.json
````

````output
exact calibrated zero-loss fixture: True
exact diagnostic nonzero fixture: True
diagnostic canonical ECL L1 loss: 1/6
diagnostic classwise ECL L1 loss: 1/6
diagnostic toplabel  ECL L1 loss: 1/48
negative controls all rejected: True
official source audit: True
all success criteria: True
````

Controls reject grouping canonical calibration only by `max(S)` (`1/12` versus
`1/18`), grouping class 1 by the wrong coordinate (`1/72` versus `1/48`), replacing
top-label correctness `q_argmax(S)=1/5` by `max(q)=7/10`, and breaking the shared
posterior in all three paradigms.

The byte-pinned official `losses.py` contains distinct `TopLabel`, `Classwise`, and
`Canonical` branches with the corresponding semantics. This result verifies formula
compatibility; it does not claim image-benchmark improvement, learned posterior-head
quality, or optimizer equivalence. Full audit: `docs/CLAIM2_THREE_PARADIGM_AUDIT.md`.
