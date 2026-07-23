# Claim 5 Table 2 audit - attempt 1

## Result

This attempt verifies the Table 2 transcription and every ECL-versus-baseline reduction on the SVHN target with exact rational arithmetic. It does **not** independently reproduce the ten-run empirical result. The defensible local classification is **inconclusive (source-only audit)**, with no evidence for falsification.

Table 2 (PDF page 8) says the entries are mean and standard deviation from ten runs. The three challenge comparisons transcribe as:

| Architecture | Printed comparison | Exact reduction | Relative reduction |
| --- | ---: | ---: | ---: |
| LeNet-5 | Uncal 61.9 vs ECL 21.5 | 40.4 percentage points (`202/5`) | `404/619` = 65.2665589660% |
| ResNet20 | PseudoCal 48.2 vs ECL 36.8 | 11.4 percentage points (`57/5`) | `57/241` = 23.6514522821% |
| DenseNet40 | Uncal 80.8 vs ECL 38.4 | 42.4 percentage points (`212/5`) | `53/101` = 52.4752475247% |

The generated JSON also computes all 24 ECL-minus-non-Oracle-baseline comparisons across the three SVHN architectures. Those are calculations over rounded paper summaries, not new experimental measurements.

## Paper evidence

- Paper: *Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift*
- OpenReview ID: `gFPPTokv9C`
- arXiv file: `2605.21552v1.pdf`
- SHA-256: `fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`
- Table anchor: PDF page 8 (printed page 8)
- Render reviewed: the table caption explicitly says mean and standard deviation derived from ten runs.

The paper's real-world setup merges the other two digit domains into the source and uses the remaining domain as target. The reported calibration metric is 15-bin top-label ECE.

Appendix J reports Ubuntu 20.04.3 LTS, Python 3.11.11, Torch 2.4.1+cu118, an Intel Core i7-10700 CPU, 125.5 GB memory, and ten NVIDIA GeForce RTX 3090 GPUs with 24 GB each. It also states a uniform batch size of 100, Adam at learning rate 0.001, 100 training epochs, and digit images converted to 3-channel RGB at 28x28.

## Current official source pin

The paper links `https://github.com/NeuroDong/ECL`. The pinned commit is:

```text
aae77f890f1e4ebc13dad135b5e29758d98d318d
tree 7572ba68c4ee432cbc7ce866b304b3d84ec1c9b8
```

Readback of that commit found one branch (`main`), no tags, no GitHub releases, no workflows, no Actions artifacts, and no forks. The complete tree is six files: `README.md`, `losses.py`, `main.ipynb`, `metrics.py`, `networks.py`, and `utils.py`. Their SHA-256 hashes are recorded and checked by `claim5_table2_audit.py`.

This commit's notebook is a three-class synthetic 2D `SimpleNet` demonstration. It has eight code cells and saved synthetic outputs, but it contains no MNIST/USPS/SVHN loader, LeNet-5, ResNet20, DenseNet40, Table 2 values, ten-run driver, digit checkpoint, digit result file, dependency manifest, or dataset revision/checksum. Therefore the current official commit cannot run Table 2.

## Older public source discovered

Search also found `https://github.com/Anonymous-user-code/ECL` at:

```text
944d492b9d542ebbc0d0396fc57a187b2ce6b293
tree 7fc9969656b5c31e32d853dcc66785e2655b0534
```

The repository has the same project description, and its commit author is `NeuraDong`. These facts make it a probable pre-deanonymization predecessor, but that relationship is an inference; it is not treated as the current official source pin.

The older repository materially improves provenance because it contains:

- `Cali_in_Digit.py` with MNIST, USPS, and SVHN downloads;
- LeNet-5, ResNet20, and DenseNet40 definitions;
- source construction by concatenating the other two domains;
- Uncal, TS, TransCal, DRL, PseudoCal, ECL, and Oracle paths;
- 100 training epochs, batch size 256, and 15-bin ECE.

It still does not contain the evidence required to reproduce the table:

- no ten-run loop or seed list;
- no explicit digit seed at all;
- no CLI/config sweep (target and architecture are manual source edits);
- no dependency manifest or version lock;
- no dataset hashes/revisions;
- no released checkpoints;
- no saved digit metrics or per-run observations;
- batch size 256 in the public digit script, conflicting with Appendix J's uniform batch size 100;
- no `weights/Digit` directory even though the fresh-training path saves there;
- DenseNet40 assigns `self.classifier = nn.Linear(...),` with a trailing comma, making it a tuple and causing its forward path to fail as written.

The public source search found no GitHub release assets or workflow artifacts in either repository. Hugging Face exact-title model and dataset searches found no released models or datasets. One third-party challenge logbook Space exists, but it contains synthetic claim checks rather than Table 2 raw runs or checkpoints, so it is not evidence for Claim 5.

## Negative controls

The audit fails closed in each of these cases:

1. A printed mean and standard deviation with no raw values is rejected as an independent reproduction.
2. Nine values are rejected because the paper specifies ten runs.
3. Repeating the printed mean ten times is rejected because it gives zero variance rather than the printed standard deviation.
4. Exact subtraction of two rounded table means is classified only as source-derived arithmetic.
5. A public training script without raw outputs, seeds, checkpoints, and a pinned environment is not promoted to an empirical reproduction.

The raw-run gate uses `fractions.Fraction`. It checks the mean inside the printed rounding interval and compares the exact variance against the squared printed-standard-deviation interval, avoiding floating-point acceptance. The paper does not specify sample (`n-1`) versus population (`n`) standard deviation, which must be resolved in a faithful rerun.

## Commands executed

```bash
pdfinfo repro/evidence/claim3/2605.21552v1.pdf
shasum -a 256 repro/evidence/claim3/2605.21552v1.pdf
pdftotext -layout repro/evidence/claim3/2605.21552v1.pdf /tmp/gfp-claim5-paper.txt
pdftoppm -f 8 -l 8 -png -r 170 repro/evidence/claim3/2605.21552v1.pdf /tmp/pdfs/gfp-claim5/table2

git clone https://github.com/NeuroDong/ECL.git /tmp/gfp-claim5-official.kVuQyB/repo
git -C /tmp/gfp-claim5-official.kVuQyB/repo checkout aae77f890f1e4ebc13dad135b5e29758d98d318d
git -C /tmp/gfp-claim5-official.kVuQyB/repo ls-tree -r --long HEAD
git -C /tmp/gfp-claim5-official.kVuQyB/repo log --all --stat
git ls-remote --heads --tags https://github.com/NeuroDong/ECL.git
gh api repos/NeuroDong/ECL/releases
gh api repos/NeuroDong/ECL/actions/artifacts

git clone https://github.com/Anonymous-user-code/ECL.git /tmp/gfp-claim5-anon.GllkzJ/repo
git -C /tmp/gfp-claim5-anon.GllkzJ/repo ls-tree -r --long HEAD
git -C /tmp/gfp-claim5-anon.GllkzJ/repo log --all --stat
python3 -m py_compile /tmp/gfp-claim5-anon.GllkzJ/repo/*.py

hf papers search 'Expectation Consistency Loss Rethink Confidence Calibration' --limit 10 --format json
hf models list --search 'Expectation Consistency Loss' --limit 20 --format json
hf datasets list --search 'Expectation Consistency Loss' --limit 20 --format json
hf spaces search 'Expectation Consistency Loss confidence calibration covariate shift' --limit 20 --include-non-running --format json
```

Final deterministic audit and tests:

```bash
.venv/bin/python repro/src/claim5_table2_audit.py \
  --paper repro/evidence/claim3/2605.21552v1.pdf \
  --official-root upstream \
  --predecessor-root /tmp/gfp-claim5-anon.GllkzJ/repo \
  --output outputs/claim5_table2_audit.json

.venv/bin/python -m pytest -q repro/tests/test_claim5_table2_audit.py
```

## Verdict recommendation and next distinct attempt

Recommendation: **inconclusive**, not verified, falsified, or toy. The quoted values are authentic paper transcriptions and their reductions are arithmetically correct, but neither is an independent empirical reproduction.

This is a deterministic source/provenance audit, not a substantive empirical attempt. The next materially distinct approach is a faithful reconstruction from the older public digit script under an authorized GPU ceiling: pin a compatible environment, record the two minimal code repairs, parameterize the three architectures on target SVHN, declare ten seeds, archive all per-run logits/ECE/accuracy/checkpoints, and cross-check 15-bin top-label ECE with an independent evaluator. No training, GPU, dependency install, Hub resource, paid compute, publication, or commit was used in this attempt.
