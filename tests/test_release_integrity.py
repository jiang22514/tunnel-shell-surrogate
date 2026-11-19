import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import verify_release  # noqa: E402


class ReleaseIntegrityTest(unittest.TestCase):
    def test_checksums_and_manifest_cover_the_release(self):
        self.assertEqual(verify_release.audit(ROOT), [])

    def test_citation_metadata(self):
        text = (ROOT/'CITATION.cff').read_text(encoding='utf-8')
        for line in ('version: "2.0.0"', 'doi: "10.5281/zenodo.21589351"', 'license: "MIT"',
                     'repository-code: "https://github.com/jiang22514/tunnel-shell-surrogate"'):
            self.assertIn(line, text)
        self.assertEqual(text.count('family-names'), 5)


if __name__ == '__main__':
    unittest.main()
