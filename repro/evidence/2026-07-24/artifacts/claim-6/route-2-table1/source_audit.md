# Claim 6 route 2 source audit

Primary source: arXiv `2605.21552v1`, PDF SHA-256
`fb1d1a634d55132694349d40d56731cc5c7401571bc8c1a9f6eee1b5849950ab`.

- Table 1 (PDF page 3, HTML anchor `S1.T1`) marks ECL true for all
  five columns, including “Mini-batch Trainable.”
- The paragraph below Table 1 calls this property “theoretically mini-batch
  trainable” and says ECL satisfies all dimensions simultaneously.
- Section 3.5 states the paper's operational requirement: the expected
  mini-batch gradient must equal the full-dataset ECL gradient.
- Theorem 3.3 asserts that equality for the auxiliary-variable formulation.
- Appendix H attempts the corresponding proof.

Thus “trainable” cannot be weakened here to “the Python loop executes.” The
paper explicitly assigns it an unbiased-gradient meaning.
