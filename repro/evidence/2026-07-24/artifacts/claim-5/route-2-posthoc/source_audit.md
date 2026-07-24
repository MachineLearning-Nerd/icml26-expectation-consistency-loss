# Claim 5 route 2 source audit

Table 2 reports target-SVHN LeNet-5 top-label ECE of `61.9 +/- 6.16` percent
for Uncal and `21.5 +/- 1.51` percent for ECL, summarized over ten runs.

The current official source commit
`aae77f890f1e4ebc13dad135b5e29758d98d318d` has no digit pipeline. Public
predecessor commit `944d492b9d542ebbc0d0396fc57a187b2ce6b293` does contain
`Cali_in_Digit.py`, the digit model topologies, and the dataset construction.
It trains the classifier and a binary correctness head jointly, then searches
positive integer temperatures 1 through 50. Its named ECL path cannot execute:
it imports `ECLoss_hd`, which is absent from `losses.py`.

The predecessor also uses batch 256 whereas Appendix J specifies batch 100,
and its head uses Adam learning rate 0.01 whereas Appendix J says the head uses
the classifier hyperparameters. Route 2 follows the paper's batch size and the
predecessor's head optimizer, and substitutes the Appendix F top-label
soft-binning formula only for the missing `ECLoss_hd`.

Positive temperature scaling preserves every argmax and therefore cannot yield
the nonzero ECL accuracy change printed in Table 2. This is an implementation
ambiguity, not a counterexample to the empirical ECE cell.
