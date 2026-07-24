# Claim 6 route 1 method

Five fixed seeds generate the full 400+400 normal-shift construction. For each
seed and each of the three calibration paradigms, the released 2→100→3
`SimpleNet` is trained on source labels for 100 epochs using Adam at 0.001.
A separate model uses the stabilized released Soft-ECE objective. The
auxiliary expectation head is then trained on source data, and the released
mini-batch ECL implementation fine-tunes the classifier output layer for 100
epochs. Target labels are withheld from every training stage.

All target labels and probability vectors are stored. A separate checker,
which does not import the released metrics, recomputes calibration errors,
accuracy, per-seed differences, uncertainty, and negative controls.
