# Claim 5 route 2 evaluation

Attempt status: **SUBSTANTIVE_DIVERGENCE_UNDER_INTERPRETATION**.

HF CPU Upgrade run `b43a864b-d804-4a62-97b5-c079eea92a95` completed at Git SHA
`e4f89e9fa0d620a2a39ba350d55938fc267c3b8c`. All 16 cumulative steps and 139
tests passed. The independent standard-library checker recomputed:

- uncalibrated target-SVHN ECE: 54.3882875%;
- post-hoc ECL target-SVHN ECE: 10.1818572%;
- target accuracy: 24.3873944%;
- rotated-label negative-control ECE: 5.7232279%.

All 99,289 rows were finite, contiguous, digest-matched, and prediction
invariance under positive temperature held. The uncalibrated result is inside
the paper mean +/- 2 printed standard deviations. The ECL result is outside
that band and substantially lower than the reported 21.5%.

This is not FALSIFICATION. The predecessor's required `ECLoss_hd` is missing,
this route records an Appendix F repair, and one seed cannot adjudicate the
paper's ten-run summary. Claim 5 remains LOW confidence after attempt 2.
