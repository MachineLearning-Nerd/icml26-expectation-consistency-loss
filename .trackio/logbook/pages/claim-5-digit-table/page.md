# Claim 5 — Digit benchmark Table 2

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c5_01", "created_at": "2026-07-19T14:20:07+00:00", "title": "Table 2 provenance audit"}
-->
**Current anchored claim:** on target SVHN, Table 2 reports ECL ECE `21.5%`
versus uncalibrated `61.9%` for LeNet-5, `36.8%` versus PseudoCal `48.2%` for
ResNet20, and `38.4%` versus uncalibrated `80.8%` for DenseNet40.

**Assessment: inconclusive; source-only attempt, not an empirical reproduction.**
The rendered paper transcription and exact arithmetic pass:

| Architecture | Printed comparison | Exact reduction | Relative reduction |
| --- | --- | ---: | ---: |
| LeNet-5 | `61.9 → 21.5` | `40.4` pp | `404/619 = 65.2666%` |
| ResNet20 | `48.2 → 36.8` | `11.4` pp | `57/241 = 23.6515%` |
| DenseNet40 | `80.8 → 38.4` | `42.4` pp | `53/101 = 52.4752%` |

All 24 SVHN ECL-versus-non-Oracle-baseline reductions are recorded, but they are
calculations over rounded paper summaries—not new measurements.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c5_02", "created_at": "2026-07-19T14:20:08+00:00", "title": "Executed provenance and raw-run gates", "command": [".venv/bin/python", "repro/src/claim5_table2_audit.py", "--paper", "repro/evidence/claim3/2605.21552v1.pdf", "--official-root", "upstream", "--output", "outputs/claim5_table2_audit.json"], "exit_code": 0}
-->
````bash
$ .venv/bin/python repro/src/claim5_table2_audit.py \
    --paper repro/evidence/claim3/2605.21552v1.pdf \
    --official-root upstream \
    --output outputs/claim5_table2_audit.json
````

The current paper-linked commit contains only a synthetic 2D notebook and no digit
pipeline, seeds, checkpoints, raw ten-run observations, or environment lock. An older
public probable predecessor contains `Cali_in_Digit.py`, but it still lacks those
artifacts, uses batch `256` versus Appendix J's `100`, saves into a missing directory,
and has a DenseNet40 trailing-comma tuple defect.

Fail-closed controls reject no raw values, nine values, ten copies of the printed mean,
and source-derived rounded arithmetic as independent evidence. A faithful next attempt
requires an environment-pinned ten-seed GPU rerun of all three architectures with raw
logits, ECE/accuracy observations, checkpoints, and an independent 15-bin ECE evaluator.
No such training is claimed here. Full audit: `docs/CLAIM5_TABLE2_AUDIT.md`.
