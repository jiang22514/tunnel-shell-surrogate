"""No private path, field identifier or other excluded token may appear in the release.

The tokens are assembled from fragments so that this file does not contain them."""
import json
import re
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import verify_release  # noqa: E402

REPOSITORY = 'github.com/jiang225' + '14/tunnel-shell-surrogate'
# case-insensitive substrings
PLAIN = ['la' + 'yue', chr(0x62c9) + chr(0x6708), 'dk' + '1193', '/ho' + 'me/', '/mn' + 't/', 'cl' + 'aude', 'open' + 'geosys']
# case-sensitive patterns: field section and sensor identifiers, drive paths, a solver acronym, e-mail addresses, a user name
PATTERNS = [r'\b[LR](?:44[38]|53[38])\b', r'\bTY' + r'L\d', r'SB' + r'WD', r'\bYB\d', r'\bGJ\d', r'\b[A-Za-z]:\\\\?[A-Za-z]',
            r'\bOG' + r'S\b', r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}', r'(?<![0-9.])225' + r'14(?![0-9])']
TEXT_NAMES = {'LICENSE', 'LICENSE-DATA', 'SHA256SUMS', '.gitignore', '.gitattributes'}
TEXT_SUFFIXES = {'.py', '.md', '.json', '.csv', '.tsv', '.txt', '.cff'}


def findings_in(text, patterns=True):
    low = text.lower()
    found = [token for token in PLAIN if token in low]
    if patterns:
        stripped = text.replace(REPOSITORY, '')
        found += [pattern for pattern in PATTERNS if re.search(pattern, stripped)]
    return found


def strings_of_npz(path):
    """Array names and the contents of all string arrays."""
    parts = []
    dtypes = verify_release.npz_dtypes(path)
    with np.load(path, allow_pickle=False) as archive:
        for name, dtype in dtypes.items():
            parts.append(name)
            if dtype.kind in 'US':
                parts.extend(str(x) for x in archive[name].ravel())
    return '\n'.join(parts)


def strings_of_checkpoint(path):
    """Member names and the pickled key table of a saved state dictionary (no tensor data)."""
    parts = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            parts.append(name)
            if name.endswith('.pkl'):
                parts.append(archive.read(name).decode('latin-1'))
    return '\n'.join(parts)


class NoPrivateContentTest(unittest.TestCase):
    def test_release_contains_no_excluded_tokens(self):
        offenders = []
        for rel in verify_release.release_files(ROOT):
            path = ROOT/rel
            found = findings_in(rel)
            if path.suffix == '.npz':
                found += findings_in(strings_of_npz(path))
            elif path.suffix == '.pt':
                found += findings_in(strings_of_checkpoint(path), patterns=False)
            elif path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
                text = path.read_text(encoding='utf-8')
                if rel in ('SHA256SUMS', 'MANIFEST.tsv'):
                    text = re.sub(r'\b[0-9a-f]{64}\b', '', text)
                found += findings_in(text)
                if path.suffix == '.json':
                    json.loads(text)
            else:
                offenders.append('%s: unexpected file type' % rel)
            if found:
                offenders.append('%s: %s' % (rel, sorted(set(found))))
        self.assertEqual(offenders, [], '\n'.join(offenders))


if __name__ == '__main__':
    unittest.main()
