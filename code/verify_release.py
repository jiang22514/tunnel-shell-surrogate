#!/usr/bin/env python3
"""Validate the public release metadata and immutable public artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from reproduce_statistics import validate_analysis_tree


REQUIRED_CFF = {
    "cff-version": "1.2.0",
    "message": "If you use this software, please cite it as below.",
    "title": "Shell-reduced surrogate for frost-season tunnel lining internal forces",
    "version": "1.0.0",
    "repository-code": "https://github.com/jiang22514/tunnel-shell-surrogate",
    "license": "MIT",
}
REQUIRED_AUTHORS = [
    "Hongyue Jiang",
    "Shunyuan Zhang",
    "Peizhong Yu",
    "Fan Wang",
    "Zhenggui Hu",
]
MANIFEST_HEADER = ["path", "bytes", "sha256", "category", "description"]
FORBIDDEN_SUFFIXES = {".cae", ".odb", ".pkl", ".pickle", ".rpy"}
FORBIDDEN_FILENAMES = {"genieshan", "full_field", "full-field"}
PRIVATE_PATH_PATTERN = re.compile(r"(?:/home/|/mnt/[a-z]/|[A-Za-z]:\\\\Users\\\\)")
CREDENTIAL_PATTERN = re.compile(
    r"(?i)\\b(?:api[_-]?key|access[_-]?token|secret|password)\\b\\s*[:=]\\s*['\"]?[A-Za-z0-9_./+=-]{8,}"
)


def _public_artifact_paths(root: Path) -> list[Path]:
    """Return regular data and model artifacts in portable lexical order."""

    return sorted(
        path
        for directory in (root / "data", root / "models")
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path, output_path: Path) -> None:
    """Write deterministic SHA-256 lines for every public data/model file."""

    root = Path(root).resolve()
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in _public_artifact_paths(root)
    ]
    Path(output_path).write_text("".join(lines), encoding="utf-8")


def _parse_cff(path: Path) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    authors: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- family-names: "):
            authors.append(line.removeprefix("- family-names: ").strip('"'))
        elif line.startswith("  given-names: ") and authors:
            authors[-1] = line.removeprefix("  given-names: ").strip('"') + " " + authors[-1]
        elif ": " in line and not line.startswith((" ", "-")):
            key, value = line.split(": ", 1)
            fields[key] = value.strip('"')
    return fields, authors


def _parse_checksums(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    entries: list[tuple[str, str]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return entries, [f"SHA256SUMS: {error}"]
    for line_number, line in enumerate(lines, start=1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            errors.append(f"SHA256SUMS:{line_number}: malformed checksum line")
        else:
            entries.append((parts[0], parts[1]))
    return entries, errors


def _manifest_errors(root: Path) -> list[str]:
    path = root / "MANIFEST.tsv"
    try:
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as error:
        return [f"MANIFEST.tsv: {error}"]
    if not rows or rows[0] != MANIFEST_HEADER:
        return ["MANIFEST.tsv: invalid header"]
    errors: list[str] = []
    paths: list[str] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(MANIFEST_HEADER):
            errors.append(f"MANIFEST.tsv:{line_number}: expected five columns")
            continue
        relative, byte_count, digest, category, description = row
        paths.append(relative)
        candidate = root / relative
        if not relative.startswith(("data/", "models/")) or not candidate.is_file():
            errors.append(f"MANIFEST.tsv:{line_number}: missing public artifact {relative}")
            continue
        if not byte_count.isdecimal() or int(byte_count) != candidate.stat().st_size:
            errors.append(f"MANIFEST.tsv:{line_number}: byte count mismatch for {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != _sha256(candidate):
            errors.append(f"MANIFEST.tsv:{line_number}: checksum mismatch for {relative}")
        if not category or not description:
            errors.append(f"MANIFEST.tsv:{line_number}: category and description are required")
    expected = [path.relative_to(root).as_posix() for path in _public_artifact_paths(root)]
    if paths != expected:
        errors.append("MANIFEST.tsv: paths must cover public artifacts in lexical order")
    return errors


def _core_array_errors(root: Path) -> list[str]:
    expected = {
        "ct_params.npy": ((100, 5), "float64"),
        "profiles_NM.npz": None,
        "tprofiles_T.npz": None,
    }
    errors: list[str] = []
    for name, schema in expected.items():
        path = root / "data" / name
        if not path.is_file():
            errors.append(f"missing data file: {name}")
            continue
        try:
            loaded = np.load(path, allow_pickle=False)
            if schema is not None:
                if (loaded.shape, str(loaded.dtype)) != schema:
                    errors.append(f"{name}: unexpected shape or dtype")
                continue
            with loaded:
                if name == "profiles_NM.npz" and (
                    loaded["N"].shape != (100, 200) or loaded["valid"].dtype != np.dtype(bool)
                ):
                    errors.append(f"{name}: unexpected profile schema")
                if name == "tprofiles_T.npz" and (
                    loaded["T_c0"].shape != (100, 200) or loaded["T_slope"].shape != (100, 200)
                ):
                    errors.append(f"{name}: unexpected temperature schema")
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"{name}: {error}")
    return errors


def _release_text_paths(root: Path) -> list[Path]:
    paths = [
        root / name
        for name in ("README.md", "CITATION.cff", "DATA_DICTIONARY.md", "LICENSE-DATA", "MANIFEST.tsv", "SHA256SUMS", "requirements.txt")
    ]
    paths.extend(path for path in (root / "code").glob("*.py") if path.name != "verify_release.py")
    return [path for path in paths if path.is_file()]


def _safety_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _public_artifact_paths(root):
        name = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(item in name for item in FORBIDDEN_FILENAMES):
            errors.append(f"forbidden release artifact: {path.relative_to(root)}")
    for path in _release_text_paths(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        if PRIVATE_PATH_PATTERN.search(text):
            errors.append(f"private absolute path in {path.relative_to(root)}")
        if CREDENTIAL_PATTERN.search(text):
            errors.append(f"credential-like text in {path.relative_to(root)}")
    return errors


def audit_release(root: Path) -> list[str]:
    """Return release-integrity findings without modifying *root*."""

    root = Path(root).resolve()
    errors: list[str] = []
    required = ("README.md", "CITATION.cff", "DATA_DICTIONARY.md", "LICENSE-DATA", "MANIFEST.tsv", "SHA256SUMS")
    errors.extend(f"missing required file: {name}" for name in required if not (root / name).is_file())
    if errors:
        return errors

    fields, authors = _parse_cff(root / "CITATION.cff")
    if fields != REQUIRED_CFF:
        errors.append("CITATION.cff: required release fields do not match")
    if authors != REQUIRED_AUTHORS:
        errors.append("CITATION.cff: author order does not match")
    if "doi" in (root / "CITATION.cff").read_text(encoding="utf-8").lower() or "orcid" in (root / "CITATION.cff").read_text(encoding="utf-8").lower():
        errors.append("CITATION.cff: DOI and ORCID must not be invented")

    entries, checksum_errors = _parse_checksums(root / "SHA256SUMS")
    errors.extend(checksum_errors)
    expected_paths = [path.relative_to(root).as_posix() for path in _public_artifact_paths(root)]
    if [relative for _, relative in entries] != expected_paths:
        errors.append("SHA256SUMS: paths must cover public artifacts in lexical order")
    for digest, relative in entries:
        candidate = root / relative
        if not candidate.is_file() or _sha256(candidate) != digest:
            errors.append(f"SHA256SUMS: checksum mismatch for {relative}")

    errors.extend(_manifest_errors(root))
    errors.extend(_core_array_errors(root))
    errors.extend(validate_analysis_tree(root / "data"))
    errors.extend(_safety_errors(root))
    return errors


def main() -> None:
    findings = audit_release(Path(__file__).resolve().parents[1])
    if findings:
        print("Release integrity findings:")
        print("\n".join(f"- {finding}" for finding in findings))
        raise SystemExit(1)
    print("Release integrity: OK")


if __name__ == "__main__":
    main()
