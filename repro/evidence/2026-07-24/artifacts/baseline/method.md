# Frozen baseline method

The baseline runs every previously accepted exact certificate, both existing
sample-complexity routes, both source-only empirical audits, and the complete
test suite. It writes machine-readable outputs beneath
`.openresearch/artifacts/`, prints every result into the OpenResearch run log,
and exits nonzero on the first failed step.

The baseline is a regression reference, not evidence that upgrades any claim.
All later child nodes must inherit the same command and locked environment and
retain these checks.
