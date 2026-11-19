# Shell-reduced profile operator for thermal internal forces of tunnel linings: code, data and trained models

Version 2.0.0. This repository accompanies the manuscript *Thermal internal forces in the insulated lining of a high-geothermal railway tunnel: Field calibration and surrogate modelling* (under review). It contains the finite element reference data of three families of tunnel cross-sections, the trained operators that predict the axial force N(s) and the bending moment M(s) around the lining, and the code that evaluates and retrains them.

The operator does not learn two-dimensional fields. At 200 stations of a landmark-aligned arc-length coordinate it predicts the through-thickness profiles of hoop strain, transverse strain and temperature together with N, and it recovers M from the profiles by the plane-strain constitutive law and integration through the thickness.

## Contents

| Folder | Content |
| --- | --- |
| `data/closed_ring/` | 164 closed linings in rock under a cold-region thermal load, four geometric inputs. Split into 96 training, 28 validation and 40 test sections. |
| `data/filled/` | Double-track sections with an invert fill and floor voids, eight inputs. 1399 training, 100 validation and four test sets of 100 sections. |
| `data/insulated/` | Single-track sections with an insulation layer, generated on the monitored design section of a high-geothermal railway tunnel in southwest China: nine geometric and four thermal inputs, half of the sections with local thickness deviations. 1400 training and 150 validation sections, three test sets (100 without deviations, 100 with one or two, 50 with three), the monitored design section, and the finite element checks of the design chart, the casting temperature and the insulation stiffness. |
| `data/expected_numbers.json` | The values printed in the manuscript that `code/reproduce_tables.py` recomputes. |
| `models/` | Trained weights: closed ring (three seeds), filled family (two stages, five seeds), insulated family (design-parameter operator A and shape-input operator B, two stages, five seeds each). |
| `predictions/` | Stored predictions on the test sets: the operators of the closed-ring and insulated families, the two baselines of all three families, and the per-section errors of the filled family. |
| `code/` | Networks, operators, training scripts, baselines, reproduction and integrity scripts. |
| `tests/` | Integrity, content and reproduction tests. |

Array names, shapes and units are listed in `DATA_DICTIONARY.md`. `MANIFEST.tsv` describes every file, and `SHA256SUMS` holds the checksums.

## Use

Python with NumPy and PyTorch (tested with Python 3.10, NumPy 2.2 and PyTorch 2.11):

```bash
python -m pip install -r requirements.txt
```

Check the files, recompute the headline errors, and run the tests:

```bash
python code/verify_release.py
python code/reproduce_tables.py --device cpu
python -m unittest discover -s tests
```

`reproduce_tables.py` evaluates the released weights on the CPU in less than a minute and compares 216 numbers with the values printed in the manuscript: the closed-ring errors of Section 4.9, the errors of both operators and of the baselines on the three test sets of the insulated family and on the held-out monitored design section (Sections 5.2 and 5.3), and the test errors of the filled family (appendix). It exits with status 1 if a number does not agree with the printed value to the printed digits.

Predict a closed-ring section from its four inputs (crown radius, knee elevation, invert-to-crown radius ratio, thickness):

```bash
python code/closed_ring.py --params 6.2 -3.1 1.4 0.4 --out prediction.csv
```

The operators of the filled and insulated families are evaluated on the released sections from the station inputs stored with each data set; see the docstrings of `code/filled.py` and `code/insulated.py`:

```python
import sys; sys.path.insert(0, "code")
import insulated
d = insulated.load("test_dev")            # inputs, station inputs and reference profiles
p = insulated.Operator("B").predict(d)    # shape-input operator: p["N"], p["M"], p["T_c0"], p["T_slope"]
```

Retraining writes to a new folder and never changes the released weights:

```bash
python code/train_closed_ring.py --out runs/closed_ring
python code/train_filled.py first --out runs/filled_first
python code/train_filled.py second --first runs/filled_first --out runs/filled_second
python code/train_insulated.py first B --out runs/insulated_B_first
python code/train_insulated.py second B --first runs/insulated_B_first --out runs/insulated_B_second
python code/baselines.py closed_ring --out runs/baselines_closed_ring.npz    # needs scikit-learn
```

A retraining reproduces the reported errors statistically, not bit for bit. The numbers of the manuscript are those of the released weights.

## Not included

The following material is not part of this release:

- field records of the monitored tunnel (temperatures, strains, rebar stresses, pressures) and anything else measured in the tunnel;
- construction drawings;
- ABAQUS input files and output databases, and the scripts that generate the finite element models;
- the geometry routines that turn a new design or an as-built outline of the filled and insulated families into station inputs (the station inputs of all released sections are in `data/`);
- two-dimensional field arrays (displacement, strain and temperature on the meshes), which the field-operator experiments of the manuscript use;
- the three-dimensional model of the Genieshan portal section and the section rebuilt from it.

The field records and drawings were supplied by the construction organisation and are confidential. The other items are available from the corresponding author where their owner permits.

## Changes from v1.0.0

Version 1.0.0 accompanied an earlier version of the manuscript and contained the frost-season closed-ring data of 100 designs. Version 2.0.0 replaces it:

- The closed-ring reference data are replaced by a corrected set of 164 sections. In v1.0.0 the convective boundary of the tunnel air was assigned in the finite element models by a distance tolerance that, for some sections, also selected element edges on the interface between lining and rock. All sections were solved again with the film condition on the tunnel surface only; geometry, mesh, materials and the other boundary conditions are unchanged. The results of v2.0.0 supersede those of v1.0.0.
- 64 closed-ring sections were added, and the split is now 96 training, 28 validation and 40 test sections.
- The filled family and the insulated family, their trained operators and their stored predictions are new.
- The code is reorganised by family, and the reproduction script covers all three families.

## Licences and citation

Code and trained weights are released under the MIT License (`LICENSE`). The data in `data/` are released under CC BY 4.0 (`LICENSE-DATA`), and the stored predictions in `predictions/` under the same terms. Please cite the release as given in `CITATION.cff`; the DOI 10.5281/zenodo.21589351 resolves to the latest version. The branch-trunk network in `code/opnn.py` follows the DIMON code of Yin et al. (2024).
