# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_conclusion_01", "created_at": "2026-07-19T13:34:19+00:00", "title": "Download and rerun"}
-->
The reproduction bundle contains the paper/code/verdict hashes, all six claim scripts,
configuration, tests, human-readable audits, captured stdout, and machine-readable results.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pytest
.venv/bin/python repro/src/claim1_general_certificate.py
.venv/bin/python repro/src/run_claim3_sample_complexity.py
.venv/bin/python repro/src/run_claim3_soft_sample_complexity.py
.venv/bin/python repro/src/run_claim3_real_mnist_sample_complexity.py \
  --data-root /path/to/uncompressed-mnist-idx
.venv/bin/python repro/src/claim3_gradient_certificate.py
.venv/bin/python repro/src/claim2_compatibility_certificate.py
.venv/bin/python repro/src/claim5_table2_audit.py \
  --paper repro/evidence/claim3/2605.21552v1.pdf --official-root upstream
.venv/bin/python repro/src/claim6_capability_audit.py
.venv/bin/python -m pytest repro/tests/ -q
```

Public repository: [MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift](https://github.com/MachineLearning-Nerd/icml26-repro-gFPPTokv9C-ecl-calibration-covariate-shift).



---
<!-- trackio-cell
{"type": "artifact", "id": "cell_acc88dbaf82a", "created_at": "2026-07-19T15:52:00+00:00", "title": "Portable six-claim reproduction bundle", "artifact": "reproduction-ecl-sample-complexity/repro-bundle:v4", "artifact_type": "reproduction"}
-->
**📦 Artifact** `reproduction-ecl-sample-complexity/repro-bundle:v4` · reproduction

https://huggingface.co/buckets/DineshAI/gFPPTokv9C-artifacts#reproduction-ecl-sample-complexity/repro-bundle:v4
