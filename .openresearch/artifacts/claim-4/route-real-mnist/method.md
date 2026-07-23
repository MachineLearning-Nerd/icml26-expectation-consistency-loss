# Claim 4 real-MNIST method

The fixed campaign command downloads official MNIST through torchvision into a
shared cache and verifies all four decompressed IDX files against preregistered
SHA-256 values. It trains a 10-class multinomial logistic model on images
0-29,999 and an independent posterior head on images 30,000-59,999. Both use
the same deterministic 77-feature representation, zero initialization, and
L-BFGS configuration. All 10,000 official test images are held out.

Positive source and target finite-pool weights are constructed from mean ink and
horizontal ink center, without labels. Since the same image atoms and posterior
head are used in both domains, only the marginal over X changes. Probability
weights are multiplied by 10,000 so Appendix F's `1e-5` stabilizer retains
sample-count semantics.

The primary implementation is PyTorch float64 autograd. A scalar calibration
temperature changes the complete trained score vector. The three ECL variants
are evaluated at temperature 1.0. A separate checker, written in NumPy and
without importing the primary implementation, recomputes each loss and a
centered finite-difference derivative with step `1e-4`.

Raw numerical inputs are written to `inputs.json`; primary metrics and controls
to `raw_results.json`; and the independent result to
`independent_checker.json`. Every executable exits nonzero on a failed contract.
