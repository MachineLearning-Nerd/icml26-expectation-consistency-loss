# Limitations and deviations

- This route proves fixed hard-bin Eq. 5, not unrestricted soft Eq. 8.
- The order contains the theorem's displayed logarithmic factor; the paper's
  shorthand `O(B/epsilon^2)` suppresses it.
- Positive source counts are required wherever the empirical target weight is
  positive; otherwise the estimator is undefined.
- A posterior head learned on the evaluation sample is outside the proof
  unless a separate stability argument is supplied.
