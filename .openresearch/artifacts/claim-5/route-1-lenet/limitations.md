# Claim 5 route 1 limitations and deviations

- The current release does not uniquely specify an executable Table 2
  protocol. Route 1 explicitly selects the paper's in-training Algorithm 2
  interpretation.
- Only one deterministic seed is run here; the paper summarizes ten.
- The digit-specific cross-entropy coefficient is not stated. The value 0.5
  comes from the official top-label demonstration.
- The predecessor trains the correctness head jointly at learning rate 0.01;
  Appendix J instead says the head uses the classifier's hyperparameters.
  Route 1 follows Appendix J: frozen backbone, learning rate 0.001, and
  Soft-ECE calibration.
- This route covers only the LeNet-5 comparison. ResNet20/PseudoCal and
  DenseNet40/Uncal require separate evidence.
- A numerical mismatch cannot falsify the paper cell because the missing seeds,
  incomplete source, and conflicting source/paper settings leave multiple
  admissible implementations.
