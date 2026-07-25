# Shell-reduced surrogate for frost-season tunnel lining internal forces

This repository is the compact public release for *A shell-reduced surrogate for frost-season thermal internal forces in cold-region tunnel linings across section geometries*. It contains portable Python code, six released model checkpoints, synthetic section/profile arrays for 100 designs, and processed analysis arrays for the fixed 20-section test split and supplementary experiments.

## Public contents

- `code/`: training, headline evaluation, statistical analysis, and release integrity scripts.
- `data/`: synthetic geometry, profile, temperature, and processed analysis arrays. Field names, shapes, units, and provenance are in `DATA_DICTIONARY.md`.
- `models/`: three mechanics and three temperature PyTorch checkpoints.
- `CITATION.cff`, `MANIFEST.tsv`, and `SHA256SUMS`: citation and immutable artifact metadata.

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run the released evaluations

Run the portable headline evaluation on CPU:

```bash
python code/reproduce_headline.py --device cpu
```

Compute the public statistical summaries and write four source-data figures to `source_figures/`:

```bash
python code/reproduce_statistics.py
```

Check the metadata and all released data/model bytes before use:

```bash
python code/verify_release.py
```

## Scope and limitations

The approximately 500 MB full-field mesh arrays are intentionally omitted. This compact package does not include Abaqus output databases, author-local working directories, or any Genieshan restricted data. The released synthetic and processed arrays support the commands above, but are not a substitute for the omitted full-field source collection.

## Licences

Code and model checkpoints are released under the MIT License; see `LICENSE`. Public synthetic and processed numeric data are released under CC BY 4.0; see `LICENSE-DATA`.