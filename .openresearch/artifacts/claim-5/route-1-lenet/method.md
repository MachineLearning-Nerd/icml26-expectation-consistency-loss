# Claim 5 route 1 method

The experiment downloads and hash-checks all official MNIST, USPS, and SVHN
files. It concatenates train and test splits exactly as the predecessor does:
79,298 MNIST+USPS source images and 99,289 SVHN target images. All images are
three-channel float tensors resized to 28x28.

A predecessor-topology LeNet-5 is trained on source labels for 100 epochs using
Adam, batch 100, and learning rate 0.001. The complete target-domain predictions
are saved as the uncalibrated baseline. With the backbone frozen, a binary
correctness head is trained on source for 100 epochs with cross-entropy plus
Soft-ECE. Finally the classifier MLP is fine-tuned for 100 epochs with source
cross-entropy and the current official top-label proximal/EMA mini-batch ECL.
Target labels are never passed to training.

The primary program writes all 99,289 target labels, predictions, and
confidences to CSV. A stdlib-only checker recomputes 15-bin ECE and accuracy
with explicit loops, validates the CSV digest and row count, and rotates labels
as a negative control. Both programs exit nonzero on failed integrity checks.
