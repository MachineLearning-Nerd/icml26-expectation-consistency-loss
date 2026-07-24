# Claim 5 route 1 source audit

Table 2 reports top-label ECE percentages as mean and standard deviation over
ten runs. For target SVHN and LeNet-5, the paper prints `61.9 +/- 6.16` for
Uncal and `21.5 +/- 1.51` for ECL.

Appendix J states a batch size of 100, Adam with learning rate 0.001, 100
classifier epochs, three-channel 28x28 digit images, and source calibration of
the correctness/posterior head using Soft-ECE. Section 4.2 says one domain is
the target and the other two domains are merged as source. Algorithm 2 defines
top-label mini-batch ECL.

The source release is not sufficient to identify a unique executable:

- current official commit `aae77f890f1e4ebc13dad135b5e29758d98d318d`
  contains only a synthetic notebook and no digit pipeline;
- predecessor commit `944d492b9d542ebbc0d0396fc57a187b2ce6b293`
  contains the three digit datasets and architectures but imports
  `ECLoss_hd`, which is absent from its `losses.py`;
- the predecessor uses batch 256, not Appendix J's 100;
- its named ECL path is post-hoc positive temperature scaling, which cannot
  change predictions, while Table 2 reports nonzero accuracy changes;
- no ten-run seeds, raw observations, checkpoints, or standard-deviation
  convention are released.

Route 1 therefore preregisters the Algorithm 2 in-training interpretation. The
ambiguity remains a limitation even if the observed values align.
