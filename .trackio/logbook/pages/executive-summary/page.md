# Executive summary

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_gfp_exec_20260719_v2", "created_at": "2026-07-19T14:25:01+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-19T14:25:01+00:00"}
-->
This revision audits all six current anchored claims while preserving the evidence that the
legacy judge already accepted. The last confirmed public verdict is **5/6** at SHA
`1abb0c87beb604420d3a0e6140ea122511c63e93`: legacy C1/C2 are `verified`, and legacy C3 sample
complexity is `toy`. The judge accepted fixed-`B` soft Eq. 8 evidence but requested a real-data
trained model; this revision adds that final materially different MNIST experiment. No additional
score increase is claimed before a fresh official verdict.

| Current claim | Local assessment | Decisive evidence |
| --- | --- | --- |
| C1 — Theorem 3.1 | **Verified with support qualifications** | General tower-property proof; exact 257-state/17-score/11-class certificate; 561 exact components |
| C2 — Sample complexity | **Hard-bin rate plus synthetic and real-trained-model soft Eq. 8 scaling supported** | Real MNIST: `0.9097/0.9076` accuracy; ECL/ECE tail slopes `-0.6809/-0.7426`; B exponents `0.889/1.014` |
| C3 — Gradient unbiasedness | **Falsified as stated** | Valid exact soft-weight counterexamples; Eq. 8 gradient `0` vs profiled Eq. 10 `1/4` |
| C4 — Three calibration paradigms | **Formula-level verified** | Exact canonical/class-wise/top-label losses `{1/6,1/6,1/48}` plus four rejected controls |
| C5 — Digit Table 2 | **Inconclusive source-only** | Table arithmetic authentic; no raw ten-run data/checkpoints/faithful released pipeline |
| C6 — Capabilities/broad empirics | **Partial source support; empirics inconclusive** | All source hashes/modes pass; PACS/raw multi-run evidence absent |

Theorem 3.3 fails for four separate reasons: Appendix H's batch scaling is wrong, same-batch
self-normalization/norm directions are biased, score-dependent soft-weight derivatives are
omitted, and Eq. 10 retains within-bin variance absent from Eq. 8. The official proximal/EMA
implementation may still be a useful heuristic; the audit falsifies its published unrestricted
objective/gradient identity, not empirical usefulness.

## Scope & cost

| Item | This revision | Full empirical replication |
| --- | --- | --- |
| Hardware | Local CPU | Isolated GPU capacity |
| Executed | Exact proofs/counterexamples, source audits, synthetic/real rate studies, **122 tests** | Ten-seed digit/PACS/ImageNet training matrices |
| Time | Seconds for deterministic local suite | Not attempted |
| Cost | USD 0 | Requires separate compute authority |
| Public provenance gaps | Explicitly fail closed | Author pipeline/raw runs/checkpoints needed |

The portable `repro-bundle:v4` contains all scripts, tests, machine-readable certificates,
captured evidence, source pins, and detailed audits.

---
<!-- trackio-cell
{"type": "figure", "id": "cell_gfp_poster_20260719_v2", "created_at": "2026-07-19T14:25:02+00:00", "title": "ECL six-claim audit poster", "pinned": true, "pinned_at": "2026-07-19T14:25:02+00:00"}
-->
````html
<!-- poster_embed.html -->
<iframe src="poster_embed.html" title="ECL six-claim reproduction audit poster" width="100%" height="820" loading="lazy"></iframe>
````
