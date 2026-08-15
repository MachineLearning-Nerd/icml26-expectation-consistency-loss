# 2026-07-24 cumulative reproduction

Previous live judged score: **6/12**.

Conservative projected score range after this candidate: **8–10/12**.

Best-supported possible new score: **10/12**, a forecast rather than a judge
result. No score increase is claimed.

| Claim | Terminal verdict | Confidence | Current points | Possible points |
| --- | --- | --- | ---: | ---: |
| 1 | VERIFIED | HIGH | 2 | 2 |
| 2 | VERIFIED | HIGH | 1 | 2 |
| 3 | FALSIFIED | HIGH | 2 | 2 |
| 4 | VERIFIED | HIGH | 1 | 2 |
| 5 | BLOCKED | LOW | 0 | 0 |
| 6 | FALSIFIED | HIGH | 0 | 2 |

The candidate branch is
`release/candidate-evidence` at Git SHA
`27f6b268ab59e159c572033852f62bb9a884088e`. The fixed command was:

```text
uv run --frozen --python 3.12 python repro/src/run_campaign.py
```

OpenResearch run `db201363-31a9-4e20-97c3-06414dc13d21` passed 23/23 steps
and 153 tests in 590.571 seconds on local CPU. It regenerated every accepted
claim check, each independent checker, all negative controls, and the
reader-facing release gate. No GPU was used.

The exact judged Space revision
`b864c4b287cffb41d35d51e471f0f23013a787e4` was preserved before this
candidate was assembled. All 26 original old paths remain present; all old page bytes
remain unchanged. `logbook.json` is the only pre-existing text path modified,
solely to make these additive pages reachable.
