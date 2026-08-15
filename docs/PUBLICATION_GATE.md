# Publication gate

Run the fast gate with:

```bash
uv run python repro/src/publication_gate.py --skip-producers
```

The gate is fail-closed and checks:

- the final project slug and paper identity;
- citation and author thank-you text;
- six terminal claim statuses and the claim summary values;
- the required report, source audit, claim map, and branch map;
- the official source pin and paper PDF hash;
- absence of tracked root `logbook.json`, `.trackio`, old `master`, and live
  `orx/*` branch refs;
- the exact MachineLearning-Nerd author/committer identity across all reachable
  final branch history;
- the release report's six-claim summary and the existing cumulative artifact
  checks.

Without `--skip-producers`, the gate invokes the existing `release_gate.py` for
the report/notebook release surface before checking the final repository. It
does not silently rerun the long campaign or download unavailable benchmark
assets.

The machine-readable results are written to
`outputs/verification.json` and `outputs/publication_gate.json` after the gate
is run. A `SCOPED_PASS` means the repository surface is trustworthy and its
limits are explicit; it does not turn blocked or falsified claims into verified
ones.
