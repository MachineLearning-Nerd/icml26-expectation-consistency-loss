# Anchored Claim 6 Capability and Evidence Audit

Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift* (`gFPPTokv9C`, arXiv `2605.21552v1`).

Claim audited exactly:

> Table 1 shows that ECL uniquely combines covariate-shift handling, class-wise and canonical calibration support, unbounded-density-ratio handling, and mini-batch trainability; Figure 2 and Appendix Table 3 provide simulated/PACS reliability-diagram and source-vs-target calibration-gap evidence.

Paper anchors: Table 1 on PDF page 3, Figure 2 on PDF page 7, Appendix Table 3 on PDF page 12. The complete 23-page PDF was read, and those three pages were also rendered and inspected visually. The paper PDF SHA-256 is `fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`.

## Verdict

`partially_reproduced_source_capabilities_only__broad_empirics_blocked`

At official source commit `aae77f890f1e4ebc13dad135b5e29758d98d318d`, the implementation surface supports a narrower part of the claim: `ECLossMiniBatch` has separate `TopLabel`, `Classwise`, and `Canonical` branches, consumes source and target batches, maintains proximal/EMA auxiliary state, and contains no density-ratio or importance-weighting operation.

That does not independently reproduce the composite claim. Table 1 is a selected literature matrix, not an exhaustive uniqueness proof. The repository exposes one saved synthetic notebook, not the PACS/ImageNet or multi-run evidence needed to recompute Figure 2 or Table 3. The reported numerical comparisons can be checked arithmetically, but remain paper-reported values.

Recommended leaderboard classification: the source-capability portion is supported; the global uniqueness and broad empirical portions are inconclusive because the public evidence is incomplete.

## Official-source pin and history

The independently read-back GitHub repository has:

- one branch: `main` at `aae77f890f1e4ebc13dad135b5e29758d98d318d`;
- no tags and no releases as observed on 2026-07-19;
- three commits total;
- source introduced in `6957faeda392d719de1c394458450e51498be139` on 2026-01-29;
- only the README title changed between that source commit and HEAD.

The pinned repository contains exactly six files. Their SHA-256 hashes are:

| File | SHA-256 |
|---|---|
| `README.md` | `b33cbce6741c9a509511087708b55b3b356864f58bfb86a77ad7488649c1c285` |
| `losses.py` | `1c2de34967f34b98faae5025368edac88f46a709d6e1e0c063e2c01f4d6e9754` |
| `main.ipynb` | `93904d9cd7b6d86aef24a48a590cb2b9c380359bf87e823f81465034a1e879df` |
| `metrics.py` | `fb5f22f5f0533175dcbbb605149f7bdd1b74d311a43213ae695c7564fae165f7` |
| `networks.py` | `96d4848cc1cfb1e136f5716fc51637ec51e9c0505bbb1835a00671d030ad1a9f` |
| `utils.py` | `3655861e165fa11680afeee030fe5be4cd30e8ea3608315b992bbc18b695c890` |

The local `upstream/` snapshot matches all six hashes.

## Table 1 transcription

The paper's exact boolean matrix is:

| Method | Covariate shift | Class-wise | Canonical | Unbounded density ratio | Mini-batch trainable |
|---|---:|---:|---:|---:|---:|
| SB-ECE | no | no | no | yes | no |
| DECE | no | no | no | yes | no |
| ECE-KDE | no | yes | yes | yes | yes |
| Weighted TS | yes | no | no | no | no |
| FL + IW + Temp | yes | no | no | no | no |
| TransCal | yes | no | no | no | no |
| DRL | yes | no | no | no | no |
| PseudoCal | yes | no | no | yes | no |
| ECL | yes | yes | yes | yes | yes |

Within these nine selected rows, ECL is the only all-true row. This exact statement is verified.

### Primary-citation semantic cross-check

- [SB-ECE](https://proceedings.neurips.cc/paper/2021/hash/f8905bd3df64ace64a68e154ba72f24c-Abstract.html) is a soft differentiable top-confidence ECE objective used during training and post hoc. Its citation supports differentiability/train-time use, but does not directly establish the narrower unbiased-mini-batch criterion used by Table 1.
- [DECE](https://openreview.net/forum?id=R2hUure38l) is a differentiable ECE surrogate in a meta-calibration framework; the primary paper does not claim covariate-shift or canonical calibration.
- [ECE-KDE](https://papers.neurips.cc/paper_files/paper/2022/hash/33d6e648ee4fb24acec3a4bbcd4f001e-Abstract-Conference.html) explicitly targets canonical calibration and says the estimator supports small-subset/mini-batch updates; it is not a covariate-shift method.
- [Weighted TS](https://arxiv.org/abs/2006.16405), [FL + IW + Temp](https://proceedings.mlr.press/v108/park20b.html), and [TransCal](https://proceedings.neurips.cc/paper/2020/hash/df12ecd077efc8c23881028604dbb8cc-Abstract.html) address covariate/domain shift through importance weighting. The Park paper explicitly states that importance weighting requires the source and target distributions to be sufficiently close.
- [DRL](https://www.ijcai.org/proceedings/2023/162) explicitly learns a differentiable density-ratio estimator for domain-shift calibration.
- [PseudoCal](https://proceedings.mlr.press/v235/hu24i.html) is a post-hoc, target-specific method using inference-stage mixup and temperature scaling, rather than a density-ratio objective.
- [ECL source](https://github.com/NeuroDong/ECL/tree/aae77f890f1e4ebc13dad135b5e29758d98d318d) exposes all three calibration branches, two-domain inputs, and stateful mini-batch proximal/EMA code without a density-ratio operation.

The global word "uniquely" remains unverified: eight comparators are not an exhaustive method census, and Table 1 gives no uniform operational test for its five booleans. In particular, differentiable train-time use is not the same statement as an unbiased mini-batch-gradient theorem. The official class implements a mini-batch heuristic, but implementation presence alone cannot establish the theory.

The absence of density-ratio code is also a bounded conclusion. It shows ECL does not compute such a ratio; it does not by itself prove empirical robustness for every distribution with an unbounded ratio.

## Official notebook provenance

`main.ipynb` contains eight code cells and no markdown cells. Seven cells have non-null execution counts, yet code cell 5 retains output while its execution count is null. The retained session is therefore not a clean, fully ordered execution record.

The saved source configures:

- `calibration_paradigm = "TopLabel"`;
- `is_normal = False`, hence a uniform synthetic shift;
- one branch selected per run;
- a seeded initializer (`set_seed(42)` through `initFun`).

Paper Figure 2 instead presents the normal synthetic shift for all three calibration paradigms plus PACS. The notebook has no `PACS` or `ImageNet` loader/path, no checkpoint save/load, and no structured result export. Its two embedded PNGs have:

| PNG | Size | SHA-256 |
|---|---:|---|
| synthetic distributions | 278,837 bytes | `47c27d9d49507c64871bd4a43e6562c7609a5276e9fcf1741bc1cc01545e6ee9` |
| top-label reliability diagram | 145,357 bytes | `65e6409c74790834403d09fb513514be3eb991cfc2b0054297af1da86700477d` |

The plotted reliability metrics are pixels inside the second image, not a machine-readable metric table or raw predictions. The notebook metadata reports Python `3.12.12`, whereas Appendix J reports Python `3.11.11` and Torch `2.4.1+cu118`; the repository has no requirements file or lockfile to reconstruct that environment.

These facts reject the negative control "an embedded notebook image is an independent reproduction of Figure 2."

## Figure 2 arithmetic

The printed ECL calibration error is lower than both displayed baselines in all six dataset/paradigm panels:

| Setting | Paradigm | NLL error | Soft-ECE error | ECL error | ECL lower than both? |
|---|---|---:|---:|---:|---:|
| simulated normal | top-label | 3.2 | 5.6 | 2.8 | yes |
| simulated normal | class-wise | 2.5 | 2.8 | 2.4 | yes |
| simulated normal | canonical | 5.7 | 5.2 | 4.8 | yes |
| PACS (three classes) | top-label | 6.5 | 6.1 | 4.4 | yes |
| PACS (three classes) | class-wise | 4.0 | 3.9 | 3.2 | yes |
| PACS (three classes) | canonical | 9.9 | 6.5 | 6.1 | yes |

The displayed accuracy does not uniformly satisfy a strict non-decrease interpretation of "preserves or improves." Relative to NLL, ECL is `-8.0`, `-1.7`, and `-0.1` percentage points on the three PACS panels, respectively. This does not negate the calibration-error comparisons; it narrows what the figure arithmetic supports.

None of these values was independently recomputed. The saved notebook's setting and output do not match Figure 2.

## Appendix Table 3 arithmetic

Table 3 contains Digit, PACS, and ImageNet-Sketch; it does not contain the simulated dataset. Across three metrics and three dataset rows, all nine printed source means are lower than the corresponding target means.

For PACS (Art + Cartoon + Sketch -> Photo), the exact target-minus-source gaps are:

- ECE: `22.3 - 3.84 = 18.46` points;
- CwECE: `7.87 - 0.58 = 7.29` points;
- ECE-KDE: `7.58 - 0.42 = 7.16` points.

The printed means and standard deviations show large descriptive gaps. Table 3's caption does not state a run count, no significance test is reported, and the public repository provides no per-run records. Therefore "source means are lower" is arithmetically supported, while statistical significance and independent empirical reproduction are not.

## Proven blocker

The public source lacks:

- a PACS/ImageNet-Sketch data-loading and evaluation pipeline;
- model checkpoints and exact architecture/preprocessing configs;
- per-run predictions, seeds, and ten-run metric records;
- structured Figure 2 and Table 3 outputs;
- a dependency lockfile matching the paper environment.

The available code can rerun one small synthetic setting only. Rebuilding PACS experiments from the paper description would not be a source-pinned reproduction because material pipeline choices are absent, and it would require substantial training excluded from this light attempt while other campaign compute is active.

The next materially distinct approach is to obtain an author-produced artifact with the PACS pipeline, exact configs/seeds, checkpoints or raw per-run predictions. After verifying its hashes, rerun one PACS -> Photo ResNet-50 protocol on isolated GPU capacity before scaling to the full ten-run matrix.

## Reproduce this audit

```bash
.venv/bin/python -m pytest repro/tests/test_claim6_capability_audit.py -q
.venv/bin/python repro/src/claim6_capability_audit.py --output outputs/claim6_capability_audit.json
```

This audit uses only standard-library parsing, hashes, exact decimal arithmetic, and static source/notebook inspection. It performs no training and creates no remote resources.
