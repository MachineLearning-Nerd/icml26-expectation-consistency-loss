# Claim 5 route 3 method

The route uses every official MNIST and USPS train/test image as source and
every SVHN train/test image as unlabeled target, with exact hash and count
checks. It trains the predecessor-topology LeNet-5 for 100 epochs, freezes the
backbone and trains the same-topology binary correctness head for 100 epochs
with cross-entropy plus stabilized Soft-ECE, then fine-tunes the classifier
head for 100 epochs with the current official Algorithm-2 top-label ECL
transcription.

Target labels are withheld from every training stage. The primary program
saves all 99,289 target rows and complete curves. A standard-library checker
recomputes 15-bin ECE and accuracy, validates the digest and row count, rotates
labels, rejects nonfinite values, and checks that the declared Soft-ECE
boundary repair was exercised. Both programs exit nonzero on a failed gate.
