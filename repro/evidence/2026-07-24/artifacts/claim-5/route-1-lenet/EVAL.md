# Claim 5 route 1 evaluation

Attempt status: **BLOCKED_BY_NONFINITE_OFFICIAL_OBJECTIVE**.

Run `9a905b04-719b-490b-a0c3-0b130b16a534` executed the full-domain,
single-seed Algorithm 2 interpretation at Git SHA
`4ad74044747ba66a5bdde9bb5532ec6a613b3b0c`. All dataset hashes and exact
sample counts passed. The uncalibrated target-SVHN result was 57.2321% ECE at
19.6678% accuracy, within the paper's 61.9 +/- 2(6.16) percentage-point band.

The correctness head's square-root SoftECE objective became non-finite after
epoch 10; its loss was NaN from epoch 20 onward, and the ECL stage consequently
produced only NaNs. The primary program's vacuous zero-ECE output was rejected
by the independent checker because confidences were non-finite, label rotation
did not change ECE, and the recomputation did not match.

This failed-closed attempt is neither evidence for nor evidence against the
Table 2 ECL value. It supplies no Claim 5 verdict.
