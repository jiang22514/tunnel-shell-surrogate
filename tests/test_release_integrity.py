from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from verify_release import audit_release, write_checksums


REQUIRED_CFF_FIELDS = {
    "cff-version": "1.2.0",
    "message": "If you use this software, please cite it as below.",
    "title": "Shell-reduced surrogate for frost-season tunnel lining internal forces",
    "version": "1.0.0",
    "repository-code": "https://github.com/jiang22514/tunnel-shell-surrogate",
    "license": "MIT",
}
EXPECTED_AUTHORS = [
    "Hongyue Jiang",
    "Shunyuan Zhang",
    "Peizhong Yu",
    "Fan Wang",
    "Zhenggui Hu",
]


class ReleaseIntegrityTest(unittest.TestCase):
    def _copy_release_tree(self, temporary_directory: str) -> Path:
        destination = Path(temporary_directory) / "release"
        shutil.copytree(
            ROOT,
            destination,
            ignore=shutil.ignore_patterns(
                ".git", ".private_release_audit", ".superpowers", "__pycache__",
                "source_figures", "*.pyc",
            ),
        )
        return destination

    def _assert_finding(self, root: Path, text: str) -> None:
        self.assertTrue(
            any(text in finding for finding in audit_release(root)),
            f"expected finding containing {text!r}",
        )

    def test_audit_accepts_the_real_release_tree_without_modifying_it(self) -> None:
        before = {
            path.relative_to(ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (ROOT / "data", ROOT / "models")
            for path in directory.rglob("*")
            if path.is_file()
        }

        self.assertEqual(audit_release(ROOT), [])

        after = {
            path.relative_to(ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (ROOT / "data", ROOT / "models")
            for path in directory.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_sha256sums_matches_every_public_data_and_model_file(self) -> None:
        lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        parsed = [line.split(maxsplit=1) for line in lines if line]
        self.assertTrue(parsed)
        self.assertEqual([parts[1] for parts in parsed], sorted(parts[1] for parts in parsed))

        expected_paths = sorted(
            path.relative_to(ROOT).as_posix()
            for directory in (ROOT / "data", ROOT / "models")
            for path in directory.rglob("*")
            if path.is_file()
        )
        self.assertEqual([parts[1] for parts in parsed], expected_paths)
        for digest, relative_path in parsed:
            actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(digest, actual, relative_path)

    def test_citation_has_required_release_fields_and_author_order(self) -> None:
        lines = (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines()
        fields = {}
        authors = []
        for line in lines:
            if line.startswith("- family-names: "):
                authors.append(line.removeprefix("- family-names: ").strip('"'))
            elif line.startswith("  given-names: "):
                authors[-1] = line.removeprefix("  given-names: ").strip('"') + " " + authors[-1]
            elif ": " in line and not line.startswith((" ", "-")):
                key, value = line.split(": ", 1)
                fields[key] = value.strip('"')

        self.assertEqual(fields, REQUIRED_CFF_FIELDS)
        self.assertEqual(authors, EXPECTED_AUTHORS)

    def test_private_audit_directory_is_ignored(self) -> None:
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".private_release_audit/", ignored)

    def test_write_checksums_reproduces_the_checked_in_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "SHA256SUMS"
            write_checksums(ROOT, output_path)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                (ROOT / "SHA256SUMS").read_text(encoding="utf-8"),
            )


    def test_safety_scan_detects_credentials_and_private_paths_in_any_release_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = self._copy_release_tree(temporary_directory)
            (release / "notes.txt").write_text(
                "api_" + "key = abcdefgh1234\n" + "C:" + "\\Users\\alice\\work\n",
                encoding="utf-8",
            )
            findings = audit_release(release)
            self.assertIn("credential-like text in notes.txt", findings)
            self.assertIn("private absolute path in notes.txt", findings)

    def test_safety_scan_rejects_forbidden_files_anywhere_in_release_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = self._copy_release_tree(temporary_directory)
            paths = (
                release / "_t3.csv",
                release / "code" / "geometry.step",
                release / "tests" / "model.odb",
                release / "docs" / "job.inp",
            )
            for candidate in paths:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_text("fixture", encoding="utf-8")
            findings = audit_release(release)
            for candidate in paths:
                self.assertTrue(
                    any(candidate.relative_to(release).as_posix() in finding for finding in findings),
                    candidate,
                )

    def test_core_archives_require_exact_schema_dtype_and_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = self._copy_release_tree(temporary_directory)
            path = release / "data" / "profiles_NM.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["N"] = arrays["N"].astype(np.float32)
            arrays["M"][0, 0] = np.nan
            arrays["unexpected"] = np.array([1.0])
            np.savez_compressed(path, **arrays)
            findings = audit_release(release)
            self.assertTrue(any("profiles_NM.npz: schema mismatch" in item for item in findings))
            self.assertTrue(any("profiles_NM.npz:M contains non-finite" in item for item in findings))

    def test_analysis_archives_require_exact_dtype_indices_and_station_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = self._copy_release_tree(temporary_directory)
            primary_path = release / "data" / "analysis" / "primary_test_predictions.npz"
            with np.load(primary_path, allow_pickle=False) as archive:
                primary = {key: archive[key] for key in archive.files}
            primary["test_indices"] = primary["test_indices"].astype(np.int32)
            np.savez_compressed(primary_path, **primary)

            shell_path = release / "data" / "analysis" / "split_predictions" / "split_1" / "mechanics_seed_1.npz"
            with np.load(shell_path, allow_pickle=False) as archive:
                shell = {key: archive[key] for key in archive.files}
            shell["test_indices"][1] = shell["test_indices"][0]
            shell["station_coordinates"][0] += 0.125
            np.savez_compressed(shell_path, **shell)

            findings = audit_release(release)
            self.assertTrue(any("primary_test_predictions.npz:test_indices has dtype" in item for item in findings))
            self.assertTrue(any("mechanics_seed_1.npz: test_indices differ" in item for item in findings))
            self.assertTrue(any("mechanics_seed_1.npz: station_coordinates differ" in item for item in findings))

    def test_analysis_json_records_require_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release = self._copy_release_tree(temporary_directory)
            path = release / "data" / "analysis" / "baseline_per_case.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["cases"][0]["axial_force_relative_l2_pct"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self._assert_finding(release, "baseline_per_case.json: invalid schema")



if __name__ == "__main__":
    unittest.main()
