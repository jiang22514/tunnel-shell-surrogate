"""Check every file of the release against SHA256SUMS and MANIFEST.tsv.

    python code/verify_release.py

SHA256SUMS lists every file of the release except itself. MANIFEST.tsv lists every file except itself and
SHA256SUMS, with its size, checksum, category and description. The check also confirms that every array archive
loads without pickled objects.
"""
from pathlib import Path
import hashlib
import re
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', 'runs', '.pytest_cache'}
MANIFEST_HEADER = ['path', 'bytes', 'sha256', 'category', 'description']


def release_files(root):
    """Relative posix paths of all release files in lexical order."""
    root = Path(root)
    files = []
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if path.is_file() and not (set(rel.parts[:-1]) & SKIP_DIRS) and path.suffix != '.pyc':
            files.append(rel.as_posix())
    return sorted(files)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def npz_dtypes(path):
    """Array names and dtypes of an .npz archive, read from the headers only."""
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                header = np.lib.format.read_array_header_1_0(handle) if version == (1, 0) else np.lib.format.read_array_header_2_0(handle)
            out[name[:-4] if name.endswith('.npy') else name] = header[2]
    return out


def npz_object_arrays(path):
    """Names of arrays in an .npz archive whose dtype holds Python objects."""
    return [name for name, dtype in npz_dtypes(path).items() if dtype.hasobject]


def audit(root=ROOT):
    """List of findings; empty if the release is intact."""
    root = Path(root)
    errors = []
    for name in ('README.md', 'DATA_DICTIONARY.md', 'CITATION.cff', 'LICENSE', 'LICENSE-DATA', 'MANIFEST.tsv', 'SHA256SUMS'):
        if not (root/name).is_file():
            errors.append('missing file: ' + name)
    if errors:
        return errors
    files = release_files(root)
    digests = {}
    listed = []
    for number, line in enumerate((root/'SHA256SUMS').read_text(encoding='utf-8').splitlines(), start=1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r'[0-9a-f]{64}', parts[0]):
            errors.append('SHA256SUMS:%d: malformed line' % number)
            continue
        listed.append(parts[1])
        digests[parts[1]] = parts[0]
    expected = [f for f in files if f != 'SHA256SUMS']
    for f in sorted(set(expected) - set(listed)):
        errors.append('not listed in SHA256SUMS: ' + f)
    for f in sorted(set(listed) - set(expected)):
        errors.append('listed in SHA256SUMS but absent: ' + f)
    actual = {}
    for f in expected:
        actual[f] = sha256(root/f)
        if f in digests and digests[f] != actual[f]:
            errors.append('checksum mismatch: ' + f)
    rows = [line.split('\t') for line in (root/'MANIFEST.tsv').read_text(encoding='utf-8').splitlines()]
    if not rows or rows[0] != MANIFEST_HEADER:
        errors.append('MANIFEST.tsv: invalid header')
    else:
        in_manifest = [f for f in expected if f != 'MANIFEST.tsv']
        paths = [row[0] for row in rows[1:]]
        if paths != in_manifest:
            errors.append('MANIFEST.tsv: paths do not cover the release files in lexical order')
        for number, row in enumerate(rows[1:], start=2):
            if len(row) != 5:
                errors.append('MANIFEST.tsv:%d: expected five columns' % number)
                continue
            rel, size, digest, category, description = row
            if rel in actual and (digest != actual[rel] or not size.isdecimal() or int(size) != (root/rel).stat().st_size):
                errors.append('MANIFEST.tsv:%d: size or checksum mismatch for %s' % (number, rel))
            if not category or not description:
                errors.append('MANIFEST.tsv:%d: category and description are required' % number)
    for f in expected:
        if f.endswith('.npz'):
            for name in npz_object_arrays(root/f):
                errors.append('object array %s in %s' % (name, f))
    return errors


def main():
    findings = audit()
    if findings:
        print('Release integrity findings:')
        print('\n'.join('- ' + finding for finding in findings))
        sys.exit(1)
    print('Release integrity: OK (%d files)' % len(release_files(ROOT)))


if __name__ == '__main__':
    main()
