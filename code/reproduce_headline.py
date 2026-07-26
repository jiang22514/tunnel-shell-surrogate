#!/usr/bin/env python3
"""Reproduce the paper's primary 20-section test-set metrics.

Only the compact arrays in ``data/`` and pretrained state dictionaries in
``models/`` are required.  No Abaqus result database or full-field mesh data
is used by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from materials import (
    ALPHA_CONCRETE,
    E_CONCRETE,
    F_TK_C40,
    NU_CONCRETE,
)
from opnn import opnn


ROOT = Path(__file__).resolve().parents[1]
SHELL_FIELDS = ("N", "slope_ss", "enn_slope")
TEMPERATURE_FIELDS = ("T_c0", "T_slope")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def _fixed_split(n_cases: int) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.random.RandomState(0).permutation(n_cases)
    return permutation[20:], permutation[:20]


def _periodic_trunk(stations: np.ndarray, harmonics: int = 32) -> np.ndarray:
    wave_numbers = np.arange(1, harmonics + 1)
    angles = 2.0 * np.pi * stations[:, None] * wave_numbers[None, :]
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1)


def _load_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibility with torch releases predating the ``weights_only`` flag.
        return torch.load(path, map_location=device)


def _predict_ensemble(
    model_paths: list[Path],
    fields: tuple[str, ...],
    targets: dict[str, np.ndarray],
    valid: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    branch1: np.ndarray,
    branch2: np.ndarray,
    trunk: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    branch1_tensor = torch.as_tensor(
        branch1[test_indices], dtype=torch.float32, device=device
    )
    branch2_tensor = torch.as_tensor(
        branch2[test_indices], dtype=torch.float32, device=device
    )
    trunk_tensor = torch.as_tensor(trunk, dtype=torch.float32, device=device)

    architecture = (
        [branch1.shape[1], 100, 100, 100],
        [branch2.shape[1], 150, 150, 150, 100],
        [trunk.shape[1], 100, 100, 100, 100, 100],
    )
    normalisation: dict[str, tuple[float, float]] = {}
    for field in fields:
        training_values = targets[field][train_indices][valid[train_indices]]
        normalisation[field] = (
            float(training_values.mean()),
            float(training_values.std()),
        )

    predictions = {field: [] for field in fields}
    for model_path in model_paths:
        checkpoint = _load_state_dict(model_path, device)
        for field in fields:
            network = opnn(*architecture).to(device).float()
            network.load_state_dict(checkpoint[field])
            network.eval()
            with torch.no_grad():
                standardised = network(
                    branch1_tensor, branch2_tensor, trunk_tensor
                ).cpu().numpy()
            mean, standard_deviation = normalisation[field]
            predictions[field].append(
                standardised * standard_deviation + mean
            )

    return {
        field: np.mean(np.stack(per_seed, axis=0), axis=0)
        for field, per_seed in predictions.items()
    }


def _relative_l2_percent(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> float:
    numerator = np.sum((prediction[mask] - truth[mask]) ** 2)
    denominator = np.sum(truth[mask] ** 2) + 1.0e-30
    return float(100.0 * np.sqrt(numerator / denominator))


def _extreme_fibre_tension(
    axial_force: np.ndarray,
    bending_moment: np.ndarray,
    thickness: float,
) -> np.ndarray:
    return axial_force / thickness + 6.0 * np.abs(bending_moment) / thickness**2


def compute_headline_metrics(
    data_dir: str | Path,
    model_dir: str | Path,
    device: str = "auto",
) -> dict[str, float]:
    """Return the paper's test-set headline and engineering-check metrics."""

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    torch_device = _resolve_device(device)

    profiles = np.load(data_dir / "profiles_NM.npz", allow_pickle=False)
    temperatures = np.load(data_dir / "tprofiles_T.npz", allow_pickle=False)
    parameters = np.load(data_dir / "ct_params.npy", allow_pickle=False)[:, :4]
    valid = profiles["valid"].astype(bool)
    train_indices, test_indices = _fixed_split(len(parameters))

    parameter_mean = parameters[train_indices].mean(axis=0)
    parameter_std = parameters[train_indices].std(axis=0)
    parameter_branch = (parameters - parameter_mean) / parameter_std
    # The primary configuration is parameter-only.  The unused shape branch is
    # therefore the same 20-component vector of ones used during training.
    shape_branch = np.ones((len(parameters), 20), dtype=np.float64)
    trunk = _periodic_trunk(profiles["sc"])

    shell_targets = {field: profiles[field] for field in SHELL_FIELDS}
    temperature_targets = {
        field: temperatures[field] for field in TEMPERATURE_FIELDS
    }
    shell_predictions = _predict_ensemble(
        sorted(model_dir.glob("_stage3_shell_model_ncs*_nobc_nopca.pt")),
        SHELL_FIELDS,
        shell_targets,
        valid,
        train_indices,
        test_indices,
        shape_branch,
        parameter_branch,
        trunk,
        torch_device,
    )
    temperature_predictions = _predict_ensemble(
        sorted(model_dir.glob("_stage3_T_model_nc*_nobc_nopca.pt")),
        TEMPERATURE_FIELDS,
        temperature_targets,
        valid,
        train_indices,
        test_indices,
        shape_branch,
        parameter_branch,
        trunk,
        torch_device,
    )

    if len(list(model_dir.glob("_stage3_shell_model_ncs*_nobc_nopca.pt"))) != 3:
        raise FileNotFoundError("Expected three released shell-model checkpoints")
    if len(list(model_dir.glob("_stage3_T_model_nc*_nobc_nopca.pt"))) != 3:
        raise FileNotFoundError("Expected three released temperature checkpoints")

    lame_lambda = (
        E_CONCRETE
        * NU_CONCRETE
        / ((1.0 + NU_CONCRETE) * (1.0 - 2.0 * NU_CONCRETE))
    )
    lame_mu = E_CONCRETE / (2.0 * (1.0 + NU_CONCRETE))
    thermal_coefficient = ALPHA_CONCRETE * (1.0 + NU_CONCRETE)
    stress_slope = lame_lambda * (
        shell_predictions["slope_ss"]
        + shell_predictions["enn_slope"]
        - 2.0 * thermal_coefficient * temperature_predictions["T_slope"]
    ) + 2.0 * lame_mu * (
        shell_predictions["slope_ss"]
        - thermal_coefficient * temperature_predictions["T_slope"]
    )
    thickness = parameters[test_indices, 3]
    moment_prediction = thickness[:, None] ** 2 * stress_slope / 12.0
    axial_prediction = shell_predictions["N"]

    n_errors: list[float] = []
    m_errors: list[float] = []
    stress_errors: list[float] = []
    stress_peak_errors: list[float] = []
    cracking_extent_errors: list[float] = []
    for local_index, case_index in enumerate(test_indices):
        mask = valid[case_index]
        n_truth = profiles["N"][case_index]
        m_truth = profiles["M"][case_index]
        n_prediction = axial_prediction[local_index]
        m_prediction = moment_prediction[local_index]

        n_errors.append(
            _relative_l2_percent(n_prediction, n_truth, mask)
        )
        m_errors.append(
            _relative_l2_percent(m_prediction, m_truth, mask)
        )

        stress_truth = _extreme_fibre_tension(
            n_truth, m_truth, thickness[local_index]
        )
        stress_prediction = _extreme_fibre_tension(
            n_prediction, m_prediction, thickness[local_index]
        )
        stress_errors.append(
            _relative_l2_percent(stress_prediction, stress_truth, mask)
        )
        truth_peak = float(stress_truth[mask].max())
        prediction_peak = float(stress_prediction[mask].max())
        stress_peak_errors.append(
            100.0 * abs(prediction_peak - truth_peak) / truth_peak
        )
        truth_fraction = float(np.mean(stress_truth[mask] > F_TK_C40))
        prediction_fraction = float(
            np.mean(stress_prediction[mask] > F_TK_C40)
        )
        cracking_extent_errors.append(
            100.0 * abs(prediction_fraction - truth_fraction)
        )

    return {
        "n_profile_median_rel_l2_pct": float(np.median(n_errors)),
        "m_profile_median_rel_l2_pct": float(np.median(m_errors)),
        "sigma_t_profile_median_rel_l2_pct": float(np.median(stress_errors)),
        "sigma_t_peak_median_rel_error_pct": float(
            np.median(stress_peak_errors)
        ),
        "cracking_extent_median_abs_error_percentage_points": float(
            np.median(cracking_extent_errors)
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for a machine-readable copy of the metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = compute_headline_metrics(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        device=args.device,
    )
    labels = {
        "n_profile_median_rel_l2_pct": "N profile median relative L2",
        "m_profile_median_rel_l2_pct": "M profile median relative L2",
        "sigma_t_profile_median_rel_l2_pct": (
            "sigma_t profile median relative L2"
        ),
        "sigma_t_peak_median_rel_error_pct": (
            "sigma_t peak median relative error"
        ),
        "cracking_extent_median_abs_error_percentage_points": (
            "cracking-extent median absolute error"
        ),
    }
    print("Fixed 80/20 split (seed 0), three-seed ensemble, C40 check")
    for key, label in labels.items():
        unit = " pp" if "percentage_points" in key else "%"
        print(f"{label}: {metrics[key]:.2f}{unit}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
