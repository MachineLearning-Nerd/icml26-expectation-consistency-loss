# Claim 5 route 2 limitations and deviations

- Only one deterministic seed is run; the paper reports ten.
- The predecessor's required `ECLoss_hd` implementation is absent. Route 2
  explicitly substitutes the Appendix F top-label soft-bin formula.
- The cumulative adaptive weight makes the selected temperature path-dependent
  on evaluating candidates in ascending order.
- Positive temperature scaling cannot change argmax accuracy, whereas Table 2
  prints a nonzero ECL accuracy change.
- The head learning rate follows the predecessor and conflicts with Appendix J.
- This route covers only the LeNet-5 target-SVHN comparison.
