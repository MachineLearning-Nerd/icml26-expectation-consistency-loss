# Claim 5 route 2 method

The route downloads and hash-checks all official MNIST, USPS, and SVHN files
and constructs the predecessor's full domains: 79,298 MNIST+USPS source rows
and 99,289 SVHN target rows, each resized to three-channel 28x28.

A predecessor-topology LeNet-5 and its binary correctness head are trained
jointly for 100 epochs. The primary classifier uses Adam at learning rate
0.001. The head uses the predecessor's batch-weighted correctness
cross-entropy and Adam at learning rate 0.01. The experiment then evaluates
temperatures 1 through 50 in ascending order, using source ECE and the
predecessor's cumulative adaptive weighting. The missing `ECLoss_hd` is
replaced by the paper's Appendix F top-label soft-bin ECL. Target labels are
never passed to training or temperature selection.

The primary program saves all 99,289 target labels, predictions, and
confidences. A separate standard-library checker reimplements 15-bin ECE and
accuracy, checks the CSV digest and row count, verifies positive-temperature
prediction invariance, and rotates labels as a negative control. Both programs
exit nonzero when evidence integrity fails.
