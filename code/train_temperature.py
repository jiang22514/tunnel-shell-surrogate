#!/usr/bin/env python3
"""Train the two temperature-profile heads in a separate joint run."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from opnn import opnn


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("T_c0", "T_slope")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120_000)
    parser.add_argument("--harmonics", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--log-every", type=int, default=20_000)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return torch.device(name)


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    profiles = np.load(args.data_dir / "profiles_NM.npz", allow_pickle=False)
    temperatures = np.load(args.data_dir / "tprofiles_T.npz", allow_pickle=False)
    parameters = np.load(args.data_dir / "ct_params.npy", allow_pickle=False)[:, :4]
    valid = profiles["valid"].astype(bool)
    stations = profiles["sc"]
    permutation = np.random.RandomState(0).permutation(len(parameters))
    test_indices, train_indices = permutation[:20], permutation[20:]

    parameter_mean = parameters[train_indices].mean(axis=0)
    parameter_std = parameters[train_indices].std(axis=0)
    parameter_branch = (parameters - parameter_mean) / parameter_std
    shape_branch = np.ones((len(parameters), 20), dtype=np.float64)

    wave_numbers = np.arange(1, args.harmonics + 1)
    angles = 2.0 * np.pi * stations[:, None] * wave_numbers[None, :]
    trunk = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)

    branch1_train = torch.as_tensor(
        shape_branch[train_indices], dtype=torch.float32, device=device
    )
    branch1_test = torch.as_tensor(
        shape_branch[test_indices], dtype=torch.float32, device=device
    )
    branch2_train = torch.as_tensor(
        parameter_branch[train_indices], dtype=torch.float32, device=device
    )
    branch2_test = torch.as_tensor(
        parameter_branch[test_indices], dtype=torch.float32, device=device
    )
    trunk_tensor = torch.as_tensor(trunk, dtype=torch.float32, device=device)
    mask = torch.as_tensor(
        valid[train_indices].astype(np.float32), device=device
    )
    mask_sum = mask.sum()

    normalisation: dict[str, tuple[float, float]] = {}
    targets: dict[str, torch.Tensor] = {}
    for field in FIELDS:
        values = temperatures[field]
        training_values = values[train_indices][valid[train_indices]]
        mean = float(training_values.mean())
        standard_deviation = float(training_values.std())
        normalisation[field] = (mean, standard_deviation)
        targets[field] = torch.as_tensor(
            (values[train_indices] - mean) / standard_deviation,
            dtype=torch.float32,
            device=device,
        )

    branch1_widths = [20, 100, 100, 100]
    branch2_widths = [4, 150, 150, 150, 100]
    trunk_widths = [2 * args.harmonics, 100, 100, 100, 100, 100]
    networks = {
        field: opnn(branch1_widths, branch2_widths, trunk_widths)
        .to(device)
        .float()
        for field in FIELDS
    }
    parameters_to_optimise = [
        parameter
        for network in networks.values()
        for parameter in network.parameters()
    ]
    optimiser = torch.optim.Adam(parameters_to_optimise, lr=1.0e-3)
    decay_epochs = {int(0.6 * args.epochs), int(0.8 * args.epochs)}

    started = time.time()
    for epoch in range(args.epochs):
        if epoch in decay_epochs:
            for group in optimiser.param_groups:
                group["lr"] *= 0.1
        optimiser.zero_grad()
        loss = sum(
            (
                (
                    networks[field](
                        branch1_train, branch2_train, trunk_tensor
                    )
                    - targets[field]
                )
                ** 2
                * mask
            ).sum()
            / mask_sum
            for field in FIELDS
        )
        loss.backward()
        optimiser.step()
        if epoch % args.log_every == 0:
            elapsed = time.time() - started
            print(
                f"epoch {epoch} loss {float(loss.detach()):.3e} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    print(
        f"two-head temperature training finished in "
        f"{(time.time() - started) / 60.0:.1f} min",
        flush=True,
    )
    output: dict[str, np.ndarray] = {
        "te": test_indices,
        "sc": stations,
    }
    for field in FIELDS:
        networks[field].eval()
        with torch.no_grad():
            standardised = networks[field](
                branch1_test, branch2_test, trunk_tensor
            ).cpu().numpy()
        mean, standard_deviation = normalisation[field]
        output[field] = standardised * standard_deviation + mean

        rmse = []
        for local_index, case_index in enumerate(test_indices):
            case_mask = valid[case_index]
            rmse.append(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                output[field][local_index][case_mask]
                                - temperatures[field][case_index][case_mask]
                            )
                            ** 2
                        )
                    )
                )
            )
        print(
            f"{field:8s} RMSE: median {np.median(rmse):.4f} "
            f"mean {np.mean(rmse):.4f} max {np.max(rmse):.4f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"nc{args.seed}_nobc_nopca"
    torch.save(
        {field: network.state_dict() for field, network in networks.items()},
        args.output_dir / f"_stage3_T_model_{stem}.pt",
    )
    np.savez(args.output_dir / f"_stage3_T_pred_{stem}.npz", **output)


if __name__ == "__main__":
    main()
