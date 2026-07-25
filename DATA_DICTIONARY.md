# Public data dictionary

All arrays are synthetic or processed numeric release artifacts. `n=100` is
the synthetic design count, `m=200` is the normalised perimeter-station count,
and `q=20` is the fixed public test split. Numeric entries are finite; boolean
entries are masks. Units labelled 鈥渘ormalised鈥?or 鈥渄imensionless鈥?have no
physical unit. Source/provenance classes are **synthetic reference**,
**model prediction**, or **processed analysis**.

## Core synthetic reference arrays

| File and field | Shape / dtype | Units | Meaning and provenance |
| --- | --- | --- | --- |
| `ct_params.npy[:, 0:4]` | `(100, 4)` / `float64` | m | Four geometric input parameters for each synthetic section; synthetic reference. |
| `ct_params.npy[:, 4]` | `(100,)` / `float64` | deg C | Fixed thermal boundary value (`-19.2`); synthetic reference. |
| `profiles_NM.npz:N`, `N_hooke`, `N_grid` | each `(100, 200)` / `float64` | MPa m | Axial-resultant reference and two constitutive/grid comparison routes; synthetic reference. |
| `profiles_NM.npz:M`, `M_hooke`, `M_grid` | each `(100, 200)` / `float64` | MPa m虏 | Bending-moment reference and two comparison routes; synthetic reference. |
| `profiles_NM.npz:Q` | `(100, 200)` / `float64` | MPa m | Shear resultant; synthetic reference. |
| `profiles_NM.npz:eps_m`, `slope_ss`, `enn_c0`, `enn_slope` | each `(100, 200)` / `float64` | dimensionless | Shell strain/strain-gradient target fields; synthetic reference. |
| `profiles_NM.npz:valid` | `(100, 200)` / `bool` | mask | Valid perimeter stations; synthetic reference. |
| `profiles_NM.npz:sc` | `(200,)` / `float64` | normalised perimeter | Shared station centres; synthetic reference. |
| `profiles_NM.npz:perim` | `(100,)` / `float64` | m | Section perimeters; synthetic reference. |
| `profiles_NM.npz:lm_tgt` | `(4,)` / `float64` | normalised perimeter | Target landmark locations for perimeter reparameterisation; synthetic reference. |
| `tprofiles_T.npz:T_c0`, `T_slope` | each `(100, 200)` / `float64` | deg C | Thermal profile coefficients; synthetic reference. |

## Processed test and analysis NPZ arrays

Every archive below uses `test_indices` (`(20,)`, `int64`, zero-based design
index, dimensionless) and `station_coordinates` (`(200,)`, `float64`,
normalised perimeter). The primary archive and all ablation archives share one fixed
20-section test split. Each `split_{1..5}` alternative-split family has its own
20-section indices; its mechanics and temperature archives share those indices and
station coordinates only within that family.

| Archive family and field | Shape / dtype | Units | Provenance |
| --- | --- | --- | --- |
| `analysis/primary_test_predictions.npz:valid_mask`, `comparison_mask` | each `(20, 200)` / `bool` | mask | Valid/evaluation mask; processed analysis. |
| `primary_test_predictions:axial_force_truth`, `axial_force_prediction`, `axial_force_prediction_strain_route` | `(20, 200)` / `float64`, `float32`, `float64` | MPa m | Reference, direct model, and strain-route axial resultants; synthetic reference/model prediction/processed analysis. |
| `primary_test_predictions:bending_moment_truth`, `bending_moment_prediction_direct`, `bending_moment_prediction_strain_route` | `(20, 200)` / `float64`, `float32`, `float64` | MPa m虏 | Reference, direct model, and strain-route bending resultants; synthetic reference/model prediction/processed analysis. |
| `primary_test_predictions:midplane_strain_prediction`, `slope_ss_prediction`, `enn_c0_prediction`, `enn_slope_prediction` | each `(20, 200)` / `float32` | dimensionless | Primary mechanics model predictions. |
| `primary_test_predictions:axial_force_relative_l2_pct`, `bending_moment_relative_l2_pct`, `slope_ss_relative_l2_pct`, `baseline_slope_ss_relative_l2_pct` | each `(20,)` / `float64` | % | Per-section relative-L2 errors; processed analysis. |
| `analysis/ablation_predictions/{primary_k32,parameters_pca_k32,parameters_pca_k16,pca_only_k32}/seed_{0,1,2}.npz:predicted_midplane_strain`, `predicted_slope_ss`, `predicted_enn_c0`, `predicted_enn_slope` | each `(20, 200)` / `float32` | dimensionless | Ablation mechanics predictions. |
| Same ablation archives: `predicted_axial_force` | `(20, 200)` / `float32` | MPa m | Ablation axial-resultant prediction. |
| Same ablation archives: `predicted_bending_moment_direct` | `(20, 200)` / `float32` | MPa m虏 | Ablation bending-moment prediction. |
| `analysis/split_predictions/split_{1..5}/mechanics_seed_{0..2}.npz`: the six `predicted_*` mechanics fields above | each `(20, 200)` / `float32` | as above | Alternative-split mechanics predictions. |
| `analysis/split_predictions/split_{1..5}/temperature_seed_{0..2}.npz:predicted_temperature_c0`, `predicted_temperature_slope` | each `(20, 200)` / `float32` | deg C | Alternative-split temperature predictions. |

## Processed JSON records

All JSON numerical values are `float` unless stated otherwise. Percent suffixes
mean percent; `pp` means percentage points; indices and training-case keys are
integers. These are processed analysis records, not raw field data.

| File and JSON field | Type / units | Meaning |
| --- | --- | --- |
| `analysis/baseline_per_case.json:axial_force_median_mean_absolute_error_pct`, `axial_force_median_relative_l2_pct`, `bending_moment_median_mean_absolute_error_pct`, `bending_moment_median_relative_l2_pct` | float / % | Summary baseline errors. |
| `baseline_per_case.json:comparison_scope` | string | Scope label for the baseline comparison. |
| `baseline_per_case.json:cases` | list of 20 objects | One record per public test case. |
| `cases[].case_index` | int / dimensionless | Zero-based synthetic-design index. |
| `cases[].axial_force_mean_absolute_error_pct`, `axial_force_relative_l2_pct`, `bending_moment_mean_absolute_error_pct`, `bending_moment_relative_l2_pct` | float / % | Per-case baseline errors. |
| `analysis/expected_statistics.json:baseline` | object | Baseline summary returned by the statistics script: `N_median_pct`, `M_median_pct`, `N_wilcoxon_p`, `M_wilcoxon_p`, `primary_N_median_pct`, `primary_M_median_pct` (float; `%` except p-values), and `N/M_median_bootstrap_interval_pct`, `primary_N/M_median_bootstrap_interval_pct` (two-float `%` lists). |
| `expected_statistics.json:ablations.{primary_k32,parameters_pca_k32,parameters_pca_k16,pca_only_k32}` | object | `slope_mean_pct`, `slope_std_pct`, `N_mean_pct`, `N_std_pct`, `M_mean_pct`, `M_std_pct`; float / %. |
| `expected_statistics.json:alternative_splits.per_split.split_{1..5}` | object | `N_median_pct`, `M_median_pct`, `sigma_profile_median_pct`, `sigma_peak_median_pct` (float / %) and `cracking_extent_median_pp` (float / pp). |
| `expected_statistics.json:alternative_splits.ranges` | object | Min/max forms of `N`, `M`, `sigma_profile`, and `sigma_peak` in `%`, plus `cracking_extent` in `pp`; float. |
| `analysis/learning_curves/seed_{0,1,2}.json:training_cases` | object keyed by `20`, `40`, `60`, `80` | Training-count records for each seed. Each value has `axial_force_relative_l2_pct`, `bending_moment_relative_l2_pct`, `midplane_strain_relative_l2_pct`, `slope_ss_relative_l2_pct`, `enn_c0_relative_l2_pct`, and `enn_slope_relative_l2_pct` (float / %). |
