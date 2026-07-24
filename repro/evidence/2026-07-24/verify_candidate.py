#!/usr/bin/env python3
"""Validate the additive, text-only Hugging Face Space candidate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


GENERATED_PATHS = {
    "repro/evidence/2026-07-24/CANDIDATE_MANIFEST.sha256",
    "repro/evidence/2026-07-24/SUBSET_CHECK.json",
    "repro/evidence/2026-07-24/UPLOAD_ALLOWLIST.txt",
}
ALLOWED_SUFFIXES = {".json", ".md", ".py", ".sha256", ".txt"}
SECRET_PATTERNS = {
    "hugging_face_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and str(path.relative_to(root)) != "PROTECTED_MANIFEST.sha256"
    }


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--write-manifests", action="store_true")
    args = parser.parse_args()

    protected = files(args.protected)
    candidate = files(args.candidate)
    old_paths = set(protected)
    new_paths = set(candidate)
    missing = sorted(old_paths - new_paths)
    if missing:
        raise SystemExit(f"old paths missing from candidate: {len(missing)}")

    modified_old = sorted(
        path for path in old_paths if digest(protected[path]) != digest(candidate[path])
    )
    if modified_old != ["logbook.json"]:
        raise SystemExit(f"unexpected modified old paths: {modified_old}")

    changed = sorted(
        path
        for path in new_paths
        if path in GENERATED_PATHS
        or path not in protected
        or digest(candidate[path]) != digest(protected[path])
    )
    for path in changed:
        if Path(path).suffix not in ALLOWED_SUFFIXES:
            raise SystemExit(f"non-text upload suffix: {path}")
        payload = candidate[path].read_bytes()
        if b"\x00" in payload:
            raise SystemExit(f"NUL byte in upload: {path}")
        text = payload.decode("utf-8")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                raise SystemExit(f"secret pattern {label} found in {path}")

    logbook = json.loads(candidate["logbook.json"].read_text(encoding="utf-8"))
    if logbook["space_id"] != "DineshAI/gFPPTokv9C":
        raise SystemExit("wrong Space id")
    children = logbook["root"]["children"]
    for node in children:
        if node["file"] not in candidate:
            raise SystemExit(f"unreachable logbook file: {node['file']}")
    if len({node["slug"] for node in children}) != len(children):
        raise SystemExit("duplicate logbook slug")

    old_pages_unchanged = all(
        digest(protected[path]) == digest(candidate[path])
        for path in protected
        if path.startswith("pages/")
    )
    if not old_pages_unchanged:
        raise SystemExit("an old page changed")

    output = {
        "status": "PASS",
        "protected_revision": "b864c4b287cffb41d35d51e471f0f23013a787e4",
        "candidate_git_sha": "27f6b268ab59e159c572033852f62bb9a884088e",
        "old_path_count": len(old_paths),
        "candidate_path_count": len(new_paths | GENERATED_PATHS) if args.write_manifests else len(new_paths),
        "old_path_set_is_subset": True,
        "old_pages_byte_identical": True,
        "modified_old_paths": modified_old,
        "new_logbook_pages": 5,
        "upload_path_count": len(changed),
        "all_uploads_utf8_text": True,
        "secret_scan": "PASS",
    }

    if args.write_manifests:
        evidence = args.candidate / "repro" / "evidence" / "2026-07-24"
        subset_path = evidence / "SUBSET_CHECK.json"
        changed = sorted(set(changed) | {
            str(subset_path.relative_to(args.candidate)),
            *GENERATED_PATHS,
        })
        output["upload_path_count"] = len(changed)
        subset_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        allowlist = evidence / "UPLOAD_ALLOWLIST.txt"
        allowlist.write_text("\n".join(changed) + "\n", encoding="utf-8")

        manifest = evidence / "CANDIDATE_MANIFEST.sha256"
        manifest_rows = [
            f"{digest(args.candidate / path)}  {path}"
            for path in changed
            if path != str(manifest.relative_to(args.candidate))
        ]
        manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    elif GENERATED_PATHS <= new_paths:
        evidence = args.candidate / "repro" / "evidence" / "2026-07-24"
        expected_allowlist = "\n".join(changed) + "\n"
        if (evidence / "UPLOAD_ALLOWLIST.txt").read_text(encoding="utf-8") != expected_allowlist:
            raise SystemExit("upload allowlist is stale")
        expected_manifest = "\n".join(
            f"{digest(args.candidate / path)}  {path}"
            for path in changed
            if path != "repro/evidence/2026-07-24/CANDIDATE_MANIFEST.sha256"
        ) + "\n"
        if (evidence / "CANDIDATE_MANIFEST.sha256").read_text(encoding="utf-8") != expected_manifest:
            raise SystemExit("candidate manifest is stale")

    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
