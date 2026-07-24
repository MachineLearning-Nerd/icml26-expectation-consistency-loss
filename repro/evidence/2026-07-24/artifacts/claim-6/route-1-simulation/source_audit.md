# Claim 6 route 1 source audit

The paper's Section 4.1 and Figure 2 specify 400 source samples from
`N([0,0], diag(5,5))`, 400 target samples from `N([2,2], diag(5,5))`, a
three-layer backpropagation network, Adam at learning rate 0.001, 100 epochs,
and 15 bins. Figure 2 presents uncalibrated, Soft-ECE, and ECL reliability
results for top-label, class-wise, and canonical calibration.

The released notebook is not an exact executable specification of that text.
Its saved default uses uniform distributions, 400/600/800 samples depending on
paradigm, Adam at 0.01 for the first stages, and 200 classifier epochs. This
route gives the paper text priority while retaining the released `SimpleNet`,
deterministic label rule, `ECLossMiniBatch`, and metric definitions.

The compound judged Claim 6 also contains Table 1 and PACS/Table 3 assertions.
Those are not silently folded into this simulation route.
