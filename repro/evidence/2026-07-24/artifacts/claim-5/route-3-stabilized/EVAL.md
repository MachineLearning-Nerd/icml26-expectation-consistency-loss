# Claim 5 route 3 evaluation

Attempt status: **SUBSTANTIVE_DIVERGENCE_UNDER_INTERPRETATION**.

HF CPU Upgrade run `18a771f4-e6c6-4222-9dcc-3d27dd5b638c` completed at Git SHA
`02a3f60f691dd069585c8ec7bf859afc672ecae4`. All 16 cumulative steps and 141
tests passed. The boundary fixture reproduced the literal zero-loss non-finite
gradient and confirmed the stabilized gradient was finite. All 300 epoch rows
were finite.

The independent checker recomputed uncalibrated ECE 41.9553451%, ECL ECE
68.4513757%, uncalibrated accuracy 25.9656155%, and ECL accuracy 23.3892979%.
Thus ECL increased ECE by 26.4960 percentage points and lowered accuracy by
2.5763 points under this declared reconstruction.

This is not FALSIFICATION: the stabilization and loss coefficient are
unreleased choices, only one seed and one architecture were run, and the
historical ten-run observations are unavailable. Claim 5 remains LOW after
three materially different attempts, so route 4 is mandatory.
