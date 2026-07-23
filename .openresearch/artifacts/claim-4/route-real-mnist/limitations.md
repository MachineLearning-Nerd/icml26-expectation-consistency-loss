# Claim 4 limitations and deviations

- The trained models are multinomial logistic classifiers, not the paper's
  LeNet-5, ResNet20, or DenseNet40 architectures.
- The independent posterior head is an estimate of `P(Y|X)`, not an oracle for
  the natural-image population conditional.
- The source and target distributions are deterministic reweightings of the
  complete held-out finite pool. This is a faithful X-only covariate-shift
  construction but not the MNIST-to-SVHN benchmark.
- The route verifies real-data formula execution and differentiation. It does
  not support any Table 2 performance number.
- Canonical `B=15` follows the official shifted simplex-grid rule and therefore
  produces 55 actual ten-class anchors. This distinction is recorded rather
  than silently treating requested and actual bin counts as equal.
