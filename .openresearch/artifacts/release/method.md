# Release method

Every experiment inherited this fixed command:

```text
uv run --frozen --python 3.12 python repro/src/run_campaign.py
```

The environment is the single repository `.venv`, resolved by `uv.lock` for
Python 3.12. Each child reruns every previously accepted check. Stochastic
routes use committed seeds and machine-readable outputs; exact routes use
rational arithmetic where possible. Each decisive result has a separately
implemented checker and a negative control. Any failing verifier exits
nonzero.

The release gate validates the six terminal verdicts, forecast arithmetic,
claim EVAL state, report images, notebook validation, and the fixed command.
The protected judged Space tree is assembled and checked outside this branch
because it is immutable external evidence.
