# Claim 5 route 3 limitations and deviations

- The `1e-12` square-root stabilization is explicit but is not stated in the
  paper or current official source.
- The paper omits the digit ECL cross-entropy coefficient; route 3 uses 0.5
  from the official top-label demonstration.
- One deterministic seed cannot verify a reported ten-run mean and standard
  deviation.
- The current official repository supplies no digit pipeline, seed schedule,
  checkpoints, or raw per-run results.
- Only the target-SVHN LeNet-5 cell is tested.
