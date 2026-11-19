# Data dictionary

All arrays are NumPy archives (`.npz`) that load with `allow_pickle=False`. `n` is the number of sections of a set and 200 the number of stations around the ring. Unless stated otherwise the arrays are `float64`.

## Conventions

- **Stations.** The lining is described along its intrados by a normalised arc-length coordinate. A piecewise-linear map pins the structural landmarks of every section to fixed positions (`lm_tgt`), so that station `sc[j]` denotes the same structural location on every section. `sc` holds the 200 aligned station centres, (j + 1/2)/200.
- **Thickness coordinate.** `n` runs from 0 at the intrados to 1 at the extrados.
- **Profiles.** At each station the hoop strain, the transverse normal strain and the temperature are fitted linearly in `n` about the mid-surface:
  hoop strain = `eps_m` + (n - 1/2) `slope_ss`, transverse strain = `enn_c0` + (n - 1/2) `enn_slope`, temperature = `T_c0` + (n - 1/2) `T_slope`.
  Strains are dimensionless, `T_c0` is in deg C and `T_slope` in K (extrados minus intrados).
- **Resultants.** `N` is the axial force in N/m and `M` the bending moment in N m/m, per unit length of tunnel, about the mid-surface, without the curved-beam correction. `N` is positive in tension and `M` is positive for tension at the extrados.
- **Recovery.** With the Lame constants of the lining and local thickness t, M = t^2/12 [(lambda + 2 mu) `slope_ss` + lambda `enn_slope` - E alpha/(1 - 2 nu) `T_slope`], and N = t [(lambda + 2 mu) `eps_m` + lambda `enn_c0` - E alpha/(1 - 2 nu) (`T_c0` - T_ref)], where T_ref is the stress-free temperature. The operators take N from its own network.
- **Errors.** All errors are relative L2 errors of a profile in percent. The closed-ring family uses the valid stations with equal weights; the other families weight each station by its arc length.

## `data/closed_ring/`

Steady plane-strain thermoelasticity of a closed concrete lining (E = 32.5 GPa, nu = 0.2, alpha = 1e-5 1/K) in rock; tunnel air at -19.2 deg C with a film coefficient of 15 W/(m2 K), ground and stress-free temperature 5 deg C.

`profiles_train.npz` (n = 96), `profiles_validation.npz` (n = 28), `profiles_test.npz` (n = 40):

| Array | Shape | Unit | Meaning |
| --- | --- | --- | --- |
| `case` | (n,) int64 | | Section number. 0 to 99: first sample; 1000 to 1063: added sections. |
| `params` | (n, 4) | m, m, -, m | Crown radius R_top, knee elevation y_knee, invert-to-crown radius ratio, lining thickness t. |
| `valid` | (n, 200) bool | | Stations with a reference profile. |
| `N`, `M` | (n, 200) | N/m, N m/m | Reference resultants. |
| `eps_m`, `slope_ss`, `enn_c0`, `enn_slope` | (n, 200) | - | Strain profile coefficients. |
| `T_c0`, `T_slope` | (n, 200) | deg C, K | Temperature profile coefficients. |
| `perim` | (n,) | m | Perimeter of the intrados. |
| `temperature_range` | (n, 2) | deg C | Lowest and highest nodal temperature of the finite element solution. |
| `sc` | (200,) | - | Aligned station coordinate. |
| `lm_tgt` | (4,) | - | Aligned positions of the landmarks: invert bottom, knee, crown, knee. The origin is the knee tangent point on the invert side. |

`design.csv` lists the four inputs of all 164 sections with `split` (train, validation, test), `sample` (first, added) and, for added sections, `added_group` (`full_range`: drawn over the full parameter ranges; `low_radius_ratio`: drawn near the lower bound of the radius ratio). `split.json` lists the section numbers of the split; the 40 test sections are 20 of the first sample and 20 added ones.

## `data/filled/`

Double-track sections with an invert fill and floor voids, same materials and thermal load as the closed rings. The fill and the voids follow one floor layout that is scaled with the section.

`design_<set>.csv` for `train`, `validation`, `test1` to `test4`: `case`, `group` and the eight inputs R_top [m], y_knee [m], R_inv_ratio, arch thickness t [m], knee radius r_k [m], invert-to-arch thickness ratio rho, relative junction height phi and floor scale s. In the training set `group` is `first_sample` (999 sections) or `thin_added` (400 added sections with t < 0.45 m).

`profiles_<set>.npz`:

| Array | Shape | Unit | Meaning |
| --- | --- | --- | --- |
| `case` | (n,) int64 | | Section number. |
| `family` | (n,) str | | Always `filled`. |
| `params` | (n, 8) | | The eight inputs in the order of the design table. |
| `valid`, `widened` | (n, 200) bool | | Stations with a profile; stations whose fitting window had to be widened. |
| `sc`, `lm_tgt` | (200,), (5,) | - | Aligned station coordinate; aligned positions of the landmarks invert bottom, left knee end, left fill junction, crown, right fill junction. The origin is the right knee end. |
| `lm` | (n, 5) | - | Arc-length fractions of the landmarks on each section. |
| `perim` | (n,) | m | Perimeter of the intrados. |
| `t_st` | (n, 200) | m | Local thickness of the structural ring at the stations. |
| `n_lining` | (n,) int64 | | Number of lining elements of the finite element model. |
| `N`, `M`, `eps_m`, `slope_ss`, `enn_c0`, `enn_slope`, `T_c0`, `T_slope` | (n, 200) | as above | Reference resultants and profile coefficients. |

`stations_<set>.npz` (geometric station inputs of the second stage, computed from the eight inputs alone):

| Array | Shape | Unit | Meaning |
| --- | --- | --- | --- |
| `case`, `params` | (n,), (n, 8) | | As in the profile file. |
| `desc` | (n, 200, 5) | - | exp(-d/0.05 m), exp(-d/0.30 m), exp(-a/0.5 m), kappa t, t_loc/t. |
| `d` | (n, 200) | m | Distance from the intrados point of the station to the nearest floor void. |
| `a` | (n, 200) | m | Arc-length distance to the nearest knee end. |
| `kappa` | (n, 200) | 1/m | Curvature of the intrados. |
| `t_geo` | (n, 200) | m | Local thickness from the geometry (equal to `t_st`). |
| `xy` | (n, 200, 2) | m | Coordinates of the intrados points of the stations. |
| `lm`, `perim` | (n, 5), (n,) | -, m | Landmarks and perimeter. |

## `data/insulated/`

Sections generated on the monitored design section: shotcrete primary support, polyurethane insulation over the arch and walls, concrete secondary lining (E = 31.5 GPa, nu = 0.2, alpha = 1e-5 1/K) and an invert fill. Each section is one steady coupled temperature-displacement solution; the lining and the fill are stress-free at the casting temperature. A thickness deviation moves the extrados of the arch and walls by A exp[-((s - s_c L)/w)^2], with L the length of the arch-and-wall extrados.

`design_<set>.json`: a list with one entry per section: `tag`; `x` with crown radius `r1` [m], crown half-angle `a1` [deg], wall-to-crown radius ratio `k2`, wall arc angle `a2` [deg], invert radius `r3` [m], lining thickness `t` [m], extrados departure angle `b` [deg], insulation thickness `t_pu` [m] and fill height `hf` [m]; `th` with rock temperature `T_far` [deg C], air temperature `T_air` [deg C], heat transfer coefficient `h` [W/(m2 K)] and casting temperature `T_cast` [deg C]; `dev`, a list of deviations [s_c, w, A] with the position s_c as a fraction of L, the width w [m] and the amplitude A [m] (positive: thicker). The entries of `design_check_insulation_stiffness.json` also give the modulus and Poisson ratio of the insulation (`insulation_E_Pa`, `insulation_nu`); all other sections use 1 MPa and 0.3.

Sets: `train` (1400), `validation` (150), `test_nom` (100, no deviation), `test_dev` (100, one or two deviations), `test_dev3` (50, three deviations), `monitored_design` (1, the monitored design section in its calibrated thermal state: finite element model output, no measurement), `check_chart_tc45` (12), `check_chart_tc38` (6), `check_chart_tc54` (8) (finite element checks of the design chart at three casting temperatures), `check_casting` (2, the monitored design section cast at 38 and 54 deg C) and `check_insulation_stiffness` (3, the monitored design section with stiffer insulation).

`profiles_<set>.npz`:

| Array | Shape | Unit | Meaning |
| --- | --- | --- | --- |
| `tag` | (n,) str | | Section name, as in the design file. |
| `params` | (n, 13) | | r1, a1, k2, a2, r3, t, b, t_pu, hf, T_far, T_air, h, T_cast. |
| `dev_fc` | (n, 8) | m | Fourier coefficients of the thickness deviation t_st - t_nom on the aligned coordinate: cosine terms 1 to 4, then sine terms 1 to 4. |
| `n_dev` | (n,) int64 | | Number of deviations. |
| `sc`, `lm_tgt` | (200,), (5,) | - | Aligned station coordinate; aligned positions of the landmarks invert bottom, left knee, left fill junction, crown, right fill junction. The origin is the right knee. |
| `lm`, `lm_nom` | (n, 5) | - | Arc-length fractions of the landmarks on the as-built and on the design outline. |
| `perim` | (n,) | m | Perimeter of the intrados. |
| `t_st`, `t_nom` | (n, 200) | m | Local lining thickness at the stations: as built, and of the design. |
| `desc_act`, `desc_nom` | (n, 200, 6) | - | Station inputs of the second stage from the as-built and from the design outline: t_loc/t, intrados curvature times t, exp(-a_K/0.5 m) with a_K the arc length to the nearest knee, exp(-a_F/0.3 m) with a_F the arc length to the nearest fill junction, local insulation thickness over t, extrados curvature times t. |
| `valid`, `widened` | (n, 200) bool | | Stations with a profile; stations whose fitting window had to be widened. |
| `n_lining` | (n,) int64 | | Number of lining elements of the finite element model. |
| `N`, `M`, `eps_m`, `slope_ss`, `enn_c0`, `enn_slope`, `T_c0`, `T_slope` | (n, 200) | as above | Reference resultants and profile coefficients. T_ref of the recovery is the casting temperature. |

Operator A (design parameters) uses `params`, `t_nom` and `desc_nom`. Operator B (shape input) uses `params`, `dev_fc`, `t_st` and `desc_act`.

## `data/expected_numbers.json`

The values printed in the manuscript, as strings, in three groups (`closed_ring`, `filled`, `insulated`). The names are those of `code/reproduce_tables.py`:

- closed ring: `GenN`, `GenM` medians; `max`, `pn` (90th percentile); `ret`/`add` for the 20 test sections of the first sample and the 20 added ones; `GenMdir` direct moment network; `GenSeedMlo`/`GenSeedMhi` single seeds; `GenFloor*` recovery applied to the reference profiles; `GenSig*` extreme-fibre tension; `GenLin*`, `GenGp*` baselines; `GenBetter` number of sections on which the operator has the lower moment error than the Gaussian process.
- filled family: `FlTest*` two-stage operator on test set 4, `FlTestthree*`, `FlTesttwo*`, `FlTestone*` on the other test sets, `FlD*` first stage alone, `FlGp*`, `FlLin*` baselines; `Thin`, `Mid`, `Thick` refer to t < 0.45 m, 0.45 to 0.6 m and t >= 0.6 m.
- insulated family: `Ins<Set><Model><Quantity>` with sets `Nom`, `Dev`, `Three`, `Mon`, models `A`, `B`, `Gp`, `Lin`, and quantities `Nmed`, `Nmean`, `Mmed`, `Mmean`, `Mpn`, `Mmax`, `Peak` (mean relative error of the peak extreme-fibre tension, percent) and `GtTen` (sections with a moment error above 10 percent); `Better` counts the sections on which operator B has the lower moment error than operator A; `PeakFE` and `InsMonPeakMPa*` are peak tensions in MPa.

## `models/`

Each operator folder holds `config.json` (architecture, training settings), `preprocessing.npz` (normalisation of inputs and outputs) and one folder per seed. First-stage and closed-ring weights are in `best_models.pt`, a dictionary of eight state dictionaries keyed by the profile names; second-stage weights are in `best.pt`. `run.json`, `training_complete.json` and `history.json` record the selected update and the validation error.

| Folder | Operator |
| --- | --- |
| `models/closed_ring/` | Three seeds. |
| `models/filled/first_stage/`, `models/filled/second_stage/` | Five seeds each. |
| `models/insulated/A/first_stage/`, `.../A/second_stage/` | Design-parameter operator, five seeds each. |
| `models/insulated/B/first_stage/`, `.../B/second_stage/` | Shape-input operator, five seeds each. |

## `predictions/`

| File | Content |
| --- | --- |
| `closed_ring/test_ensemble.npz`, `closed_ring/test_seed_<seed>.npz` | Operator on the 40 test sections: `case`, `params`, `sc`, `valid`, the eight predicted profiles, and `N_recovered`, `M_recovered` from the predicted profiles. The reported N is `N`, the reported M is `M_recovered`. |
| `closed_ring/baselines_test.npz` | `case`, `N_lin`, `M_lin` (linear regression), `N_GP`, `M_GP` (Gaussian process on 12 principal components). |
| `filled/test_errors.json` | Per-section errors on test sets 1 to 4 of the two-stage operator (`two_stage_<set>`) and of the first stage (`first_stage_<set>`), and of the baselines on test set 4. |
| `filled/baselines_test4.npz` | `case`, `N_lin`, `M_lin`, `N_GP`, `M_GP` on test set 4. |
| `insulated/pred_<set>.npz` | `tag`; for operators A and B: `N_A`, `M_A`, `N_B`, `M_B` (final), `M_DA`, `M_DB` (first stage), `T_c0_*`, `T_slope_*`; baselines `N_lin`, `M_lin`, `N_GP`, `M_GP`. Sets: `test_nom`, `test_dev`, `test_dev3`, `monitored_design`. |
