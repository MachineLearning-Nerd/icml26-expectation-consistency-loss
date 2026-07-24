# Claim 6 route 1 limitations

- The paper calls the network three-layer; the released `SimpleNet` has two
  learned affine layers and one ReLU (the input may have been counted).
- The released notebook conflicts with the paper's distribution, sample-size,
  learning-rate, and epoch statements. The paper-text choices are frozen here.
- A `1e-12` radicand floor repairs the released Soft-ECE zero-gradient
  singularity already observed in Claim 5 route 1.
- The released `ECLossMiniBatch` is used despite Theorem 3.3's already-proven
  gradient-bias counterexamples; this route tests empirical behavior, not that
  theorem.
- Figure 2 is only one conjunct of Claim 6. This route cannot establish the
  Table 1 or PACS/Table 3 components.
