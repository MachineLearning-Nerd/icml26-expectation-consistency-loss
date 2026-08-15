#!/usr/bin/env python3
"""Run the fast ECL publication gate and write its result."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-producers", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    producer_results: dict[str, str] = {}

    if not args.skip_producers:
        completed = subprocess.run(
            [sys.executable, "repro/src/release_gate.py"],
            cwd=root,
            text=True,
            capture_output=True,
        )
        producer_results["release_gate"] = "PASS" if completed.returncode == 0 else "FAIL"
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

    verifier = subprocess.run(
        [sys.executable, "repro/src/verify_results.py"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if verifier.stdout:
        print(verifier.stdout, end="")
    if verifier.stderr:
        print(verifier.stderr, end="", file=sys.stderr)
    verification = json.loads(verifier.stdout)
    producers_ok = all(value == "PASS" for value in producer_results.values())
    gate_ok = verifier.returncode == 0 and producers_ok
    result = {
        "status": "SCOPED_PASS" if gate_ok else "FAIL",
        "producers": producer_results,
        "verification_status": verification["status"],
        "scope": "Six-claim evidence and repository publication surface; not an official score.",
    }
    (root / "outputs/publication_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
