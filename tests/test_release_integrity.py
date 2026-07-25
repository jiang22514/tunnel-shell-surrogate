from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


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
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "SHA256SUMS"
            write_checksums(ROOT, output_path)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                (ROOT / "SHA256SUMS").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
