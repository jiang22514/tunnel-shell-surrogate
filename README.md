# Shell-reduced surrogate for frost-season tunnel lining internal forces

Reproducibility package for:

> *A shell-reduced surrogate for frost-season thermal internal forces in
> cold-region tunnel linings across section geometries*

The compact release is self-contained for reproducing the paper's primary
20-section test-set results. It does not require Abaqus output databases,
the authors' local directories, or the approximately 500 MB full-field mesh
dataset.

## Contents

- `data/ct_params.npy`: four section parameters for 100 designs (the stored
  fifth column is the fixed thermal boundary value).
- `data/profiles_NM.npz`: 200-station reference mechanics/resultant profiles
  and validity masks.
- `data/tprofiles_T.npz`: 200-station temperature-profile coefficients.
- `models/`: three mechanics checkpoints and three temperature checkpoints.
- `code/reproduce_headline.py`: portable evaluation and C40 cracking check.
- `code/train_mechanics.py`: joint training of the six mechanics heads
  (`eps_m`, `slope_ss`, `enn_c0`, `enn_slope`, `N`, and `M`).
- `code/train_temperature.py`: a separate joint training run for the two
  temperature heads (`T_c0` and `T_slope`).
- `code/legacy/`: scope notes for extended analyses whose raw inputs are not
  contained in this compact package.

All eight output networks have independent weights and no shared trunk.
"Joint training" refers to the summed loss within each run: six mechanics
networks in one run and two temperature networks in a separate run.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

The released results were produced with PyTorch 2.11, NumPy 2.2,
scikit-learn 1.7, and SciPy 1.15. The compact headline reproduction itself
uses only NumPy and PyTorch.

## Reproduce the headline test-set results

From the repository root:

```bash
python code/reproduce_headline.py --device cpu
```

Expected output (minor last-digit variation between platforms is acceptable):

```text
N profile median relative L2: 3.19%
M profile median relative L2: 9.45%
sigma_t profile median relative L2: 3.40%
sigma_t peak median relative error: 3.71%
cracking-extent median absolute error: 1.00 pp
```

The cracking-extent calculation uses the C40 characteristic axial tensile
strength `f_tk = 2.39 MPa`, matching the manuscript. The bending moment uses
the paper's primary strain/temperature constitutive route; axial force uses
the direct `N` head.

To write a machine-readable result file:

```bash
python code/reproduce_headline.py --device cpu --output-json results.json
```

## Retrain the released profile operators

The two training commands use the same fixed 80/20 split (split seed 0),
32-harmonic periodic trunk, 120,000 full-batch epochs, and learning-rate
schedule described in the paper:

```bash
python code/train_mechanics.py --seed 0 --device cuda
python code/train_temperature.py --seed 0 --device cuda
```

Repeat with `--seed 1` and `--seed 2` for the three-seed ensemble. On the
reported RTX 5090, the six mechanics heads took about 16 minutes per seed and
the two temperature heads about 6 minutes per seed when run sequentially.

## Run regression tests

```bash
python -m unittest discover -s tests -v
```

License: MIT (see `LICENSE`).
