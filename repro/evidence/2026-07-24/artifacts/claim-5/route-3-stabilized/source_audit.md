# Claim 5 route 3 source audit

Appendix J states that digit experiments use batch 100, Adam at learning rate
0.001, and 100 classifier epochs. It says the binary correctness head is
calibrated on source using Soft-ECE, has the same structure and hyperparameters
as the classifier head, and is trained with the backbone frozen. Algorithm 2
specifies the top-label mini-batch ECL update.

The current official `soft_binning_ece` computes the square root of a weighted
squared discrepancy without adding an epsilon. At the valid zero-discrepancy
boundary its derivative is undefined in floating-point autodiff; attempt 1
reached that boundary and poisoned the head and ECL stages with NaNs.

Route 3 keeps the formula, Appendix-J optimizer settings, frozen backbone, and
Algorithm-2 transcription, but preregisters one numerical repair:
`sqrt(weighted_squared_discrepancy + 1e-12)`. A boundary fixture must show the
literal gradient is non-finite and the repaired gradient finite before the
full run proceeds.

The paper still does not provide the ten seeds, digit ECL cross-entropy
coefficient, checkpoints, raw observations, or executable digit pipeline.
