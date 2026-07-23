# Claim 6 — Capability matrix and broad empirical evidence

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_c6_01", "created_at": "2026-07-19T14:20:09+00:00", "title": "Source capability and provenance audit"}
-->
**Current anchored claim:** Table 1 identifies ECL as the only compared method with
all five listed capabilities; Figure 2 and Appendix Table 3 provide simulated/PACS
reliability and source-target calibration-gap evidence.

**Assessment: source-capability portion supported; global uniqueness and broad
empirics inconclusive.** The complete six-file official snapshot matches commit
`aae77f890f1e4ebc13dad135b5e29758d98d318d` by SHA-256. It implements separate
`TopLabel`, `Classwise`, and `Canonical` branches, consumes both domains, maintains
proximal/EMA mini-batch state, and contains no density-ratio operation.

The exact Table 1 transcription has ECL as the sole all-true row among its selected
nine methods. That establishes uniqueness only within the table—not across all
calibration literature—and implementation presence does not validate Claim 3's
unbiased-gradient theorem.

---
<!-- trackio-cell
{"type": "code", "id": "cell_gfp_c6_02", "created_at": "2026-07-19T14:20:10+00:00", "title": "Executed capability and evidence audit", "command": [".venv/bin/python", "repro/src/claim6_capability_audit.py", "--output", "outputs/claim6_capability_audit.json"], "exit_code": 0}
-->
````bash
$ .venv/bin/python repro/src/claim6_capability_audit.py \
    --output outputs/claim6_capability_audit.json
````

Paper-derived arithmetic checks show:

- Figure 2 ECL calibration error is below both displayed baselines in all six panels.
- PACS ECL accuracy is nevertheless below NLL by `-8.0`, `-1.7`, and `-0.1`
  percentage points across top-label, class-wise, and canonical panels.
- All nine Table 3 source means are below target means; PACS target-minus-source gaps
  are `18.46`, `7.29`, and `7.16` points for ECE, CwECE, and ECE-KDE.

These are arithmetic checks of paper figures/tables, not independent measurements.
The official repository exposes one saved uniform-shift TopLabel notebook with two
embedded PNGs, an incomplete execution-count trail, and no PACS/ImageNet pipeline,
checkpoints, per-run predictions, structured metrics, or environment lock. Its setting
does not match Figure 2. A substantive next attempt needs an author-produced PACS
pipeline or raw predictions/configs, followed by a source-pinned PACS→Photo ResNet-50
GPU rerun. Full audit: `docs/CLAIM6_CAPABILITY_AUDIT.md`.
