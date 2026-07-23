# Claim 6 route 2 limitations

- The result does not imply that the released training loop cannot execute or
  cannot sometimes improve calibration.
- It falsifies the narrower and explicit Table 1 property: *theoretically*
  mini-batch trainable under the Section 3.5 unbiased-gradient definition.
- PACS and Figure 2 are not independently reproduced by this route. They are
  not needed to falsify a compound conjunction once a required Table 1 cell is
  false.
- A corrected fixed-direction estimator with independent frozen state remains
  possible and is retained as a negative control; it is not the claimed Eq. 10
  estimator.
