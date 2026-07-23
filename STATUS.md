# Status — gFPPTokv9C

- Campaign owner: `perfect-score campaign`
- Current official score: **5/6** at judged Space SHA `1abb0c87beb604420d3a0e6140ea122511c63e93`
- Current legacy verdicts: C1 `verified`, C2 `verified`, C3/sample complexity `toy`
- Local/publication state: final real-MNIST revision published at Space SHA `c2a00fa8085ad607f19a97ad64b906739cdc7d2d`; public bundle and Space readback verified; official reevaluation pending
- Compute/cost ceiling: local CPU only; no paid/remote compute
- GitHub: no commit or push authorized or performed in this revision

## Completed local revision

- Current claim source: the live prompt overlays six anchored claims; legacy/default mapping is documented in the logbook.
- C1: general conditional-expectation proof plus exact `257 X × 17 S × 11 classes` certificate; 561 components and assumption controls pass.
- C2: preserved the judged hard-bin evidence and added the actual soft, self-normalized Eq. 8
  estimator: tail RMSE slope `-0.537951`, matched ECE slopes near `-0.5`, exact population
  reference, independent calculation agreement, and explicit tiny-mass/universal-`B` limits.
- C2 attempt 2 official result: still `toy`; the judge accepted comparable fixed-`B`
  `epsilon^-2` scaling but requested real-data trained-model evidence and noted that the
  empirical sweep does not establish universal `B` dependence.
- C2 final attempt 3: two disjoint MNIST-trained classifiers, held-out official test split,
  X-only covariate shift, ECL/ECE tail slopes `-0.680940/-0.742584`, construction-specific
  B exponents `0.889/1.014`, and independent implementation agreement below `5e-17`.
- C3: exact falsification of Theorem 3.3 as written; repaired certificate uses only valid interior soft-assignment paths and seven probability-domain gates.
- C4: exact canonical/class-wise/top-label certificate, all three predicted classes, four rejected semantic controls, and official mode-source audit.
- C5: exact Table 2/provenance audit; no independent empirical result claimed because raw ten-run data and faithful released training assets are absent.
- C6: official capability surface and paper arithmetic audited; broad PACS/ImageNet empirical reproduction remains blocked by missing public artifacts.
- Verification: **122 passed**; the real-MNIST rerun reproduced every scientific field exactly.

## Publication gates

- [x] Build, upload, and read back `repro-bundle:v2` with 48 files including the soft-estimator evidence.
- [x] Build, upload, and read back `repro-bundle:v4` with the real-MNIST trained-model evidence; the remote manifest digest is `ef50869768de1e9dd271157e30b0433ddb7f04ff98616b9fe7668067cf167a46`.
- [x] Upload and hash-read back every artifact blob and Trackio manifest from `DineshAI/gFPPTokv9C-artifacts` (42/42 v1 entries; 46 retained CAS blobs; zero deletions).
- [x] Run the current official validator: the canonical title-derived target passes; the existing historical target reports only the authorized legacy-slug exception.
- [x] Update the existing public Space: 18/18 scoped logbook files match byte-for-byte at
  `c2a00fa8085ad607f19a97ad64b906739cdc7d2d`; both real-MNIST cells and the v4 artifact
  link are visible in remote Trackio readback.
- [x] Refresh the official verdict dataset after publication: SHA
  `1abb0c87beb604420d3a0e6140ea122511c63e93` was judged at `2026-07-19T15:31:26+00:00`
  and remains 5/6 (`verified`, `verified`, `toy`).

## Known external constraint

The existing authorized Space uses historical slug `gFPPTokv9C`. The canonical validator requires a `repro-<title>` slug. Creating or moving to another Space is outside the current authority; content validation can still be demonstrated against a hypothetical canonical target while the exact existing target reports only that slug error.
