#!/usr/bin/env python3
"""Compute compact-release statistics and regenerate four source figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from materials import ALPHA_CONCRETE, E_CONCRETE, F_TK_C40, NU_CONCRETE


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIRNAME = "analysis"
CONFIGURATIONS = (
    "primary_k32",
    "parameters_pca_k32",
    "parameters_pca_k16",
    "pca_only_k32",
)
PRIMARY_KEYS = {
    "test_indices", "station_coordinates", "valid_mask", "axial_force_truth",
    "bending_moment_truth", "axial_force_prediction",
    "bending_moment_prediction_direct", "axial_force_prediction_strain_route",
    "bending_moment_prediction_strain_route", "comparison_mask",
    "midplane_strain_prediction", "slope_ss_prediction", "enn_c0_prediction",
    "enn_slope_prediction", "axial_force_relative_l2_pct",
    "bending_moment_relative_l2_pct", "slope_ss_relative_l2_pct",
    "baseline_slope_ss_relative_l2_pct",
}
SHELL_KEYS = {
    "test_indices", "station_coordinates", "predicted_midplane_strain",
    "predicted_slope_ss", "predicted_enn_c0", "predicted_enn_slope",
    "predicted_axial_force", "predicted_bending_moment_direct",
}
TEMPERATURE_KEYS = {
    "test_indices", "station_coordinates", "predicted_temperature_c0",
    "predicted_temperature_slope",
}


PRIMARY_DTYPES = {
    "test_indices": "int64", "station_coordinates": "float64",
    "valid_mask": "bool", "comparison_mask": "bool",
    "axial_force_truth": "float64", "bending_moment_truth": "float64",
    "axial_force_prediction": "float32", "bending_moment_prediction_direct": "float32",
    "axial_force_prediction_strain_route": "float64",
    "bending_moment_prediction_strain_route": "float64",
    "midplane_strain_prediction": "float32", "slope_ss_prediction": "float32",
    "enn_c0_prediction": "float32", "enn_slope_prediction": "float32",
    "axial_force_relative_l2_pct": "float64",
    "bending_moment_relative_l2_pct": "float64",
    "slope_ss_relative_l2_pct": "float64",
    "baseline_slope_ss_relative_l2_pct": "float64",
}
SHELL_DTYPES = {
    "test_indices": "int64", "station_coordinates": "float64",
    "predicted_midplane_strain": "float32", "predicted_slope_ss": "float32",
    "predicted_enn_c0": "float32", "predicted_enn_slope": "float32",
    "predicted_axial_force": "float32", "predicted_bending_moment_direct": "float32",
}
TEMPERATURE_DTYPES = {
    "test_indices": "int64", "station_coordinates": "float64",
    "predicted_temperature_c0": "float32", "predicted_temperature_slope": "float32",
}


def _schema_for_keys(required_keys: set[str]) -> dict[str, str]:
    if required_keys == PRIMARY_KEYS:
        return PRIMARY_DTYPES
    if required_keys == SHELL_KEYS:
        return SHELL_DTYPES
    if required_keys == TEMPERATURE_KEYS:
        return TEMPERATURE_DTYPES
    raise ValueError("unknown analysis archive schema")


def _load_npz(path: Path, required_keys: set[str]) -> dict[str, np.ndarray]:
    expected_dtypes = _schema_for_keys(required_keys)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required_keys:
            raise ValueError(f"{path}: schema mismatch (unexpected keys)")
        arrays = {key: archive[key] for key in archive.files}
    for key, array in arrays.items():
        if str(array.dtype) != expected_dtypes[key]:
            raise TypeError(f"{path}:{key} has dtype {array.dtype}, expected {expected_dtypes[key]}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"{path}:{key} contains non-finite values")
    return arrays

def _relative_l2_percent(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> float:
    numerator = np.sum((prediction[mask] - truth[mask]) ** 2)
    denominator = np.sum(truth[mask] ** 2) + 1.0e-30
    return float(100.0 * np.sqrt(numerator / denominator))


def _extreme_fibre_tension(
    axial_force: np.ndarray, bending_moment: np.ndarray, thickness: float
) -> np.ndarray:
    return axial_force / thickness + 6.0 * np.abs(bending_moment) / thickness**2


def _bootstrap_median_interval(values: np.ndarray, random_state: np.random.RandomState) -> list[float]:
    samples = random_state.choice(values, size=(20_000, len(values)), replace=True)
    lower, upper = np.percentile(np.median(samples, axis=1), (2.5, 97.5))
    return [float(lower), float(upper)]


def _analysis_root(data_dir: Path) -> Path:
    return data_dir / ANALYSIS_DIRNAME


def _expected_analysis_paths(analysis_dir: Path) -> dict[Path, set[str]]:
    expected: dict[Path, set[str]] = {
        analysis_dir / "primary_test_predictions.npz": PRIMARY_KEYS,
    }
    for configuration in CONFIGURATIONS:
        for seed in range(3):
            expected[
                analysis_dir / "ablation_predictions" / configuration / f"seed_{seed}.npz"
            ] = SHELL_KEYS
    for split in range(1, 6):
        for seed in range(3):
            split_dir = analysis_dir / "split_predictions" / f"split_{split}"
            expected[split_dir / f"mechanics_seed_{seed}.npz"] = SHELL_KEYS
            expected[split_dir / f"temperature_seed_{seed}.npz"] = TEMPERATURE_KEYS
    return expected


def _array_structure_errors(
    path: Path, arrays: dict[str, np.ndarray], required_keys: set[str]
) -> list[str]:
    errors: list[str] = []
    test_indices = arrays["test_indices"]
    stations = arrays["station_coordinates"]
    if test_indices.shape != (20,):
        errors.append(f"{path}: test_indices must have shape (20,)")
    if np.any(test_indices < 0) or np.any(test_indices >= 100):
        errors.append(f"{path}: test_indices must be in [0, 100)")
    if len(np.unique(test_indices)) != len(test_indices):
        errors.append(f"{path}: test_indices must be unique")
    if stations.shape != (200,):
        errors.append(f"{path}: station_coordinates must have shape (200,)")
    if len(stations) > 1 and not np.all(np.diff(stations) > 0):
        errors.append(f"{path}: station_coordinates must be strictly increasing")
    profile_shape = (len(test_indices), len(stations))
    if required_keys == PRIMARY_KEYS:
        profile_keys = PRIMARY_KEYS - {
            "test_indices", "station_coordinates", "axial_force_relative_l2_pct",
            "bending_moment_relative_l2_pct", "slope_ss_relative_l2_pct",
            "baseline_slope_ss_relative_l2_pct",
        }
        vector_keys = {
            "axial_force_relative_l2_pct", "bending_moment_relative_l2_pct",
            "slope_ss_relative_l2_pct", "baseline_slope_ss_relative_l2_pct",
        }
    else:
        profile_keys = required_keys - {"test_indices", "station_coordinates"}
        vector_keys = set()
    for key in profile_keys:
        if arrays[key].shape != profile_shape:
            errors.append(f"{path}: {key} has shape {arrays[key].shape}, expected {profile_shape}")
    for key in vector_keys:
        if arrays[key].shape != (len(test_indices),):
            errors.append(f"{path}: {key} has shape {arrays[key].shape}, expected {(len(test_indices),)}")
    return errors


def _archive_group_errors(
    archives: dict[Path, dict[str, np.ndarray]], paths: list[Path]
) -> list[str]:
    errors: list[str] = []
    reference_path = paths[0]
    reference = archives.get(reference_path)
    if reference is None:
        return errors
    for path in paths[1:]:
        arrays = archives.get(path)
        if arrays is None:
            continue
        if not np.array_equal(arrays["test_indices"], reference["test_indices"]):
            errors.append(f"{path}: test_indices differ from {reference_path}")
        if not np.array_equal(arrays["station_coordinates"], reference["station_coordinates"]):
            errors.append(f"{path}: station_coordinates differ from {reference_path}")
    return errors


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _mapping_has_exact_keys(payload: object, keys: set[str]) -> bool:
    return isinstance(payload, dict) and set(payload) == keys


def _analysis_json_errors(path: Path, payload: object) -> list[str]:
    name = path.name
    invalid = False
    if name == "baseline_per_case.json":
        top_keys = {
            "axial_force_median_mean_absolute_error_pct", "axial_force_median_relative_l2_pct",
            "bending_moment_median_mean_absolute_error_pct", "bending_moment_median_relative_l2_pct",
            "cases", "comparison_scope",
        }
        case_keys = {
            "case_index", "axial_force_mean_absolute_error_pct", "axial_force_relative_l2_pct",
            "bending_moment_mean_absolute_error_pct", "bending_moment_relative_l2_pct",
        }
        invalid = not _mapping_has_exact_keys(payload, top_keys)
        if not invalid:
            invalid = not all(_is_finite_number(payload[key]) for key in top_keys - {"cases", "comparison_scope"})
            invalid |= not isinstance(payload["comparison_scope"], str)
            cases = payload["cases"]
            invalid |= not isinstance(cases, list) or len(cases) != 20
            if not invalid:
                invalid |= any(
                    not _mapping_has_exact_keys(case, case_keys)
                    or not isinstance(case["case_index"], int)
                    or not 0 <= case["case_index"] < 100
                    or not all(_is_finite_number(case[key]) for key in case_keys - {"case_index"})
                    for case in cases
                )
                invalid |= len({case["case_index"] for case in cases}) != len(cases)
    elif name == "expected_statistics.json":
        ablation_keys = {"slope_mean_pct", "slope_std_pct", "N_mean_pct", "N_std_pct", "M_mean_pct", "M_std_pct"}
        split_keys = {"N_median_pct", "M_median_pct", "sigma_profile_median_pct", "sigma_peak_median_pct", "cracking_extent_median_pp"}
        range_keys = {"N_min_pct", "N_max_pct", "M_min_pct", "M_max_pct", "sigma_profile_min_pct", "sigma_profile_max_pct", "sigma_peak_min_pct", "sigma_peak_max_pct", "cracking_extent_min_pp", "cracking_extent_max_pp"}
        baseline_keys = {"N_median_pct", "M_median_pct", "N_median_bootstrap_interval_pct", "M_median_bootstrap_interval_pct", "N_wilcoxon_p", "M_wilcoxon_p", "primary_N_median_pct", "primary_M_median_pct", "primary_N_median_bootstrap_interval_pct", "primary_M_median_bootstrap_interval_pct"}
        invalid = not _mapping_has_exact_keys(payload, {"baseline", "ablations", "alternative_splits"})
        if not invalid:
            baseline = payload["baseline"]
            invalid = not _mapping_has_exact_keys(baseline, baseline_keys)
            if not invalid:
                intervals = [baseline[key] for key in baseline_keys if key.endswith("interval_pct")]
                invalid |= not all(isinstance(value, list) and len(value) == 2 and all(_is_finite_number(item) for item in value) for value in intervals)
                invalid |= not all(_is_finite_number(baseline[key]) for key in baseline_keys if not key.endswith("interval_pct"))
            ablations = payload["ablations"]
            invalid |= not _mapping_has_exact_keys(ablations, set(CONFIGURATIONS))
            invalid |= any(not _mapping_has_exact_keys(item, ablation_keys) or not all(_is_finite_number(value) for value in item.values()) for item in ablations.values()) if isinstance(ablations, dict) else True
            alternative = payload["alternative_splits"]
            invalid |= not _mapping_has_exact_keys(alternative, {"per_split", "ranges"})
            if not invalid:
                per_split = alternative["per_split"]
                invalid |= not _mapping_has_exact_keys(per_split, {f"split_{index}" for index in range(1, 6)})
                invalid |= any(not _mapping_has_exact_keys(item, split_keys) or not all(_is_finite_number(value) for value in item.values()) for item in per_split.values()) if isinstance(per_split, dict) else True
                ranges = alternative["ranges"]
                invalid |= not _mapping_has_exact_keys(ranges, range_keys) or not all(_is_finite_number(value) for value in ranges.values())
    elif name.startswith("seed_"):
        metric_keys = {"axial_force_relative_l2_pct", "bending_moment_relative_l2_pct", "enn_c0_relative_l2_pct", "enn_slope_relative_l2_pct", "midplane_strain_relative_l2_pct", "slope_ss_relative_l2_pct"}
        invalid = not _mapping_has_exact_keys(payload, {"training_cases"})
        if not invalid:
            cases = payload["training_cases"]
            invalid = not _mapping_has_exact_keys(cases, {"20", "40", "60", "80"})
            invalid |= any(not _mapping_has_exact_keys(item, metric_keys) or not all(_is_finite_number(value) for value in item.values()) for item in cases.values()) if isinstance(cases, dict) else True
    return [f"{path}: invalid schema"] if invalid else []

def validate_analysis_tree(data_dir: Path) -> list[str]:
    """Return exact-schema, safety, and cross-archive consistency errors."""

    data_dir = Path(data_dir)
    analysis_dir = _analysis_root(data_dir)
    expected = _expected_analysis_paths(analysis_dir)
    errors: list[str] = []
    archives: dict[Path, dict[str, np.ndarray]] = {}
    for path, required_keys in expected.items():
        if not path.is_file():
            errors.append(f"missing: {path.relative_to(data_dir)}")
            continue
        try:
            arrays = _load_npz(path, required_keys)
        except (OSError, TypeError, ValueError) as error:
            errors.append(str(error))
            continue
        archives[path] = arrays
        errors.extend(_array_structure_errors(path, arrays, required_keys))

    actual_paths = set(analysis_dir.rglob("*.npz")) if analysis_dir.is_dir() else set()
    for path in sorted(actual_paths - set(expected)):
        errors.append(f"unexpected analysis archive: {path.relative_to(data_dir)}")

    primary_path = analysis_dir / "primary_test_predictions.npz"
    for configuration in CONFIGURATIONS:
        paths = [
            analysis_dir / "ablation_predictions" / configuration / f"seed_{seed}.npz"
            for seed in range(3)
        ]
        errors.extend(_archive_group_errors(archives, [primary_path, *paths]))
    for split in range(1, 6):
        split_dir = analysis_dir / "split_predictions" / f"split_{split}"
        paths = [
            split_dir / f"{kind}_seed_{seed}.npz"
            for kind in ("mechanics", "temperature")
            for seed in range(3)
        ]
        errors.extend(_archive_group_errors(archives, paths))

    json_paths = [
        analysis_dir / "baseline_per_case.json",
        analysis_dir / "expected_statistics.json",
        *(analysis_dir / "learning_curves" / f"seed_{seed}.json" for seed in range(3)),
    ]
    for path in json_paths:
        try:
            payload = json.loads(path.read_text())
            json.dumps(payload, allow_nan=False)
            errors.extend(_analysis_json_errors(path, payload))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
    return errors

def _load_references(data_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(data_dir / "profiles_NM.npz", allow_pickle=False) as archive:
        profiles = {key: archive[key] for key in archive.files}
    parameters = np.load(data_dir / "ct_params.npy", allow_pickle=False)
    return profiles, parameters


def _case_errors(
    prediction: np.ndarray,
    truth: np.ndarray,
    test_indices: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _relative_l2_percent(prediction[row], truth[case], valid[case])
            for row, case in enumerate(test_indices)
        ]
    )


def _moment_from_strain_route(
    slope_ss: np.ndarray,
    enn_slope: np.ndarray,
    temperature_slope: np.ndarray,
    thickness: np.ndarray,
) -> np.ndarray:
    lame_lambda = E_CONCRETE * NU_CONCRETE / ((1.0 + NU_CONCRETE) * (1.0 - 2.0 * NU_CONCRETE))
    lame_mu = E_CONCRETE / (2.0 * (1.0 + NU_CONCRETE))
    thermal_coefficient = ALPHA_CONCRETE * (1.0 + NU_CONCRETE)
    stress_slope = lame_lambda * (
        slope_ss + enn_slope - 2.0 * thermal_coefficient * temperature_slope
    ) + 2.0 * lame_mu * (slope_ss - thermal_coefficient * temperature_slope)
    return thickness[:, None] ** 2 * stress_slope / 12.0


def _baseline_statistics(
    analysis_dir: Path, primary: dict[str, np.ndarray]
) -> dict[str, object]:
    baseline = json.loads((analysis_dir / "baseline_per_case.json").read_text())
    cases = baseline["cases"]
    case_indices = np.asarray([case["case_index"] for case in cases])
    if not np.array_equal(case_indices, primary["test_indices"]):
        raise ValueError("baseline case order does not match primary test_indices")
    baseline_n = np.asarray([case["axial_force_relative_l2_pct"] for case in cases])
    baseline_m = np.asarray([case["bending_moment_relative_l2_pct"] for case in cases])
    primary_n = primary["axial_force_relative_l2_pct"]
    primary_m = primary["bending_moment_relative_l2_pct"]
    random_state = np.random.RandomState(0)
    return {
        "N_median_pct": float(np.median(baseline_n)),
        "M_median_pct": float(np.median(baseline_m)),
        "N_median_bootstrap_interval_pct": _bootstrap_median_interval(baseline_n, random_state),
        "M_median_bootstrap_interval_pct": _bootstrap_median_interval(baseline_m, random_state),
        "N_wilcoxon_p": float(wilcoxon(baseline_n, primary_n).pvalue),
        "M_wilcoxon_p": float(wilcoxon(baseline_m, primary_m).pvalue),
        "primary_N_median_pct": float(np.median(primary_n)),
        "primary_M_median_pct": float(np.median(primary_m)),
        "primary_N_median_bootstrap_interval_pct": _bootstrap_median_interval(primary_n, random_state),
        "primary_M_median_bootstrap_interval_pct": _bootstrap_median_interval(primary_m, random_state),
    }


def _ablation_statistics(
    analysis_dir: Path, profiles: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    valid = profiles["valid"].astype(bool)
    results: dict[str, dict[str, float]] = {}
    for configuration in CONFIGURATIONS:
        per_seed: dict[str, list[float]] = {"slope": [], "N": [], "M": []}
        reference_indices: np.ndarray | None = None
        for seed in range(3):
            prediction = _load_npz(
                analysis_dir / "ablation_predictions" / configuration / f"seed_{seed}.npz",
                SHELL_KEYS,
            )
            indices = prediction["test_indices"]
            if reference_indices is None:
                reference_indices = indices
            elif not np.array_equal(reference_indices, indices):
                raise ValueError(f"{configuration} seed test_indices differ")
            per_seed["slope"].append(float(np.median(_case_errors(
                prediction["predicted_slope_ss"], profiles["slope_ss"], indices, valid
            ))))
            per_seed["N"].append(float(np.median(_case_errors(
                prediction["predicted_axial_force"], profiles["N"], indices, valid
            ))))
            per_seed["M"].append(float(np.median(_case_errors(
                prediction["predicted_bending_moment_direct"], profiles["M"], indices, valid
            ))))
        results[configuration] = {
            "slope_mean_pct": float(np.mean(per_seed["slope"])),
            "slope_std_pct": float(np.std(per_seed["slope"], ddof=0)),
            "N_mean_pct": float(np.mean(per_seed["N"])),
            "N_std_pct": float(np.std(per_seed["N"], ddof=0)),
            "M_mean_pct": float(np.mean(per_seed["M"])),
            "M_std_pct": float(np.std(per_seed["M"], ddof=0)),
        }
    return results


def _alternative_split_statistics(
    analysis_dir: Path, profiles: dict[str, np.ndarray], parameters: np.ndarray
) -> dict[str, object]:
    valid = profiles["valid"].astype(bool)
    per_split: dict[str, dict[str, float]] = {}
    for split in range(1, 6):
        split_dir = analysis_dir / "split_predictions" / f"split_{split}"
        mechanics = [_load_npz(split_dir / f"mechanics_seed_{seed}.npz", SHELL_KEYS) for seed in range(3)]
        temperatures = [_load_npz(split_dir / f"temperature_seed_{seed}.npz", TEMPERATURE_KEYS) for seed in range(3)]
        indices = mechanics[0]["test_indices"]
        if not all(np.array_equal(item["test_indices"], indices) for item in mechanics + temperatures):
            raise ValueError(f"split_{split} has inconsistent test_indices")
        axial = np.mean([item["predicted_axial_force"] for item in mechanics], axis=0)
        moment = _moment_from_strain_route(
            np.mean([item["predicted_slope_ss"] for item in mechanics], axis=0),
            np.mean([item["predicted_enn_slope"] for item in mechanics], axis=0),
            np.mean([item["predicted_temperature_slope"] for item in temperatures], axis=0),
            parameters[indices, 3],
        )
        n_errors = _case_errors(axial, profiles["N"], indices, valid)
        m_errors = _case_errors(moment, profiles["M"], indices, valid)
        stress_errors: list[float] = []
        peak_errors: list[float] = []
        cracking_errors: list[float] = []
        for row, case in enumerate(indices):
            thickness = parameters[case, 3]
            mask = valid[case]
            truth_stress = _extreme_fibre_tension(profiles["N"][case], profiles["M"][case], thickness)
            predicted_stress = _extreme_fibre_tension(axial[row], moment[row], thickness)
            stress_errors.append(_relative_l2_percent(predicted_stress, truth_stress, mask))
            peak_errors.append(100.0 * abs(float(predicted_stress[mask].max()) - float(truth_stress[mask].max())) / float(truth_stress[mask].max()))
            cracking_errors.append(100.0 * abs(float(np.mean(predicted_stress[mask] > F_TK_C40)) - float(np.mean(truth_stress[mask] > F_TK_C40))))
        per_split[f"split_{split}"] = {
            "N_median_pct": float(np.median(n_errors)),
            "M_median_pct": float(np.median(m_errors)),
            "sigma_profile_median_pct": float(np.median(stress_errors)),
            "sigma_peak_median_pct": float(np.median(peak_errors)),
            "cracking_extent_median_pp": float(np.median(cracking_errors)),
        }
    metric_names = next(iter(per_split.values())).keys()
    ranges = {
        f"{metric[:-11]}_min_pct" if metric.endswith("_median_pct") else f"{metric[:-10]}_min_pp": float(min(values[metric] for values in per_split.values()))
        for metric in metric_names
    }
    ranges.update({
        f"{metric[:-11]}_max_pct" if metric.endswith("_median_pct") else f"{metric[:-10]}_max_pp": float(max(values[metric] for values in per_split.values()))
        for metric in metric_names
    })
    return {"per_split": per_split, "ranges": ranges}


def compute_statistics(data_dir: Path) -> dict[str, object]:
    """Compute all release statistics from compact public arrays only."""

    data_dir = Path(data_dir)
    errors = validate_analysis_tree(data_dir)
    if errors:
        raise ValueError("invalid analysis tree:\n" + "\n".join(errors))
    analysis_dir = _analysis_root(data_dir)
    primary = _load_npz(analysis_dir / "primary_test_predictions.npz", PRIMARY_KEYS)
    profiles, parameters = _load_references(data_dir)
    return {
        "baseline": _baseline_statistics(analysis_dir, primary),
        "ablations": _ablation_statistics(analysis_dir, profiles),
        "alternative_splits": _alternative_split_statistics(analysis_dir, profiles, parameters),
    }


def write_source_figures(data_dir: Path, output_dir: Path) -> list[Path]:
    """Write the four compact, source-data figures and return their paths."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = _analysis_root(data_dir)
    primary = _load_npz(analysis_dir / "primary_test_predictions.npz", PRIMARY_KEYS)
    baseline = json.loads((analysis_dir / "baseline_per_case.json").read_text())

    paths: list[Path] = []
    path = output_dir / "error_distributions.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    axes[0].boxplot([
        [case["axial_force_relative_l2_pct"] for case in baseline["cases"]],
        primary["axial_force_relative_l2_pct"],
    ], tick_labels=["baseline", "primary"])
    axes[0].set_title("Axial-force relative L2")
    axes[0].set_ylabel("error (%)")
    axes[1].boxplot([
        [case["bending_moment_relative_l2_pct"] for case in baseline["cases"]],
        primary["bending_moment_relative_l2_pct"],
    ], tick_labels=["baseline", "primary"])
    axes[1].set_title("Bending-moment relative L2")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "selected_profiles.png"
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
    station = primary["station_coordinates"]
    selected = (0, min(1, len(primary["test_indices"]) - 1))
    for row in selected:
        axes[0].plot(station, primary["axial_force_truth"][row], alpha=0.75)
        axes[0].plot(station, primary["axial_force_prediction"][row], "--", alpha=0.75)
        axes[1].plot(station, primary["bending_moment_truth"][row], alpha=0.75)
        axes[1].plot(station, primary["bending_moment_prediction_direct"][row], "--", alpha=0.75)
    axes[0].set_ylabel("N")
    axes[1].set_ylabel("M")
    axes[1].set_xlabel("normalised station")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "stationwise_errors.png"
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
    axes[0].plot(station, np.mean(np.abs(primary["axial_force_prediction"] - primary["axial_force_truth"]), axis=0))
    axes[0].set_ylabel("mean |N error|")
    axes[1].plot(station, np.mean(np.abs(primary["bending_moment_prediction_direct"] - primary["bending_moment_truth"]), axis=0))
    axes[1].set_ylabel("mean |M error|")
    axes[1].set_xlabel("normalised station")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = output_dir / "learning_curve.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    curves = [json.loads((analysis_dir / "learning_curves" / f"seed_{seed}.json").read_text())["training_cases"] for seed in range(3)]
    counts = np.asarray(sorted(int(count) for count in curves[0]))
    for curve in curves:
        axes[0].plot(counts, [curve[str(count)]["axial_force_relative_l2_pct"] for count in counts], marker="o", alpha=0.75)
        axes[1].plot(counts, [curve[str(count)]["bending_moment_relative_l2_pct"] for count in counts], marker="o", alpha=0.75)
    axes[0].set_ylabel("N relative L2 (%)")
    axes[1].set_ylabel("M relative L2 (%)")
    for axis in axes:
        axis.set_xlabel("training cases")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "source_figures")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    statistics = compute_statistics(args.data_dir)
    write_source_figures(args.data_dir, args.output_dir)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
