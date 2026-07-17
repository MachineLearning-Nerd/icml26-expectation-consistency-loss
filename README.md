# Repro — ECL: Calibration under Covariate Shift (gFPPTokv9C)

Clean-room reproduction of *Expectation Consistency Loss: Rethink Confidence Calibration
under Covariate Shift* (Dong et al.; arXiv [2605.21552](https://arxiv.org/abs/2605.21552)),
for the [ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `gFPPTokv9C`.

**Theorem 3.1 (necessary-and-sufficient condition).** Under covariate shift
(`P_s(X)≠P_t(X)`, shared `P(Y|X)`, `S=f(X)` ⟹ `Y⊥S|X`): source calibration transfers to the
target **iff** the Expectation-Consistency condition holds:
`P_s(Y_k=1|S)=P_t(Y_k=1|S)` ⟺ `E_{P_s(X|S)}[P(Y_k=1|X)]=E_{P_t(X|S)}[P(Y_k=1|X)]`.

## Results (all CPU, exact discrete enumeration)

| Claim | Verdict | Headline evidence |
|---|---|---|
| **C1** iff condition for calibration transfer | **VERIFIED** | EC residual == calibration gap to **0.00e+00** on 40 instances; EC=0 ⟺ gap=0 consistent; engine `P(Y\|S)=E[P(Y\|X)\|S]` holds on both domains. |
| **C2** compatibility (class-wise / top-label) | **VERIFIED** | the iff holds for alternative summary `S_k` (Theorem D.2) — same total-probability structure. |

Negative control: breaking covariate shift (`P_s(Y|X)≠P_t(Y|X)`) makes the iff **fail** (20/20) — the shared-`g` EC residual ≠ the true calibration gap, confirming the theorem's scope. C3 (sample complexity, a rate) is out of scope. 5/5 pytest tests pass.

## Reproduce
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy scipy pytest
python repro/src/run_ecl.py
python -m pytest repro/tests/
```

## Scope & honest disclosures
- C1 (iff) + C2 (compatibility) verified exactly. C3 (sample-complexity rate) is out of scope.
- The iff is the law of total probability applied under covariate shift (a clean characterization of *when* calibration transfers); verified substantively with both directions + the covariate-shift-scope negative control.
- Official code `NeuroDong/ECL` (the ECL training loss) cross-references; theorem verification is clean-room numpy enumeration.

Logbook: https://huggingface.co/spaces/DineshAI/gFPPTokv9C
