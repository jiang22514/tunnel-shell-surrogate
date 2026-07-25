"""DeepONet branch--trunk network used by the released profile operators.

The architecture is reproduced from the training code by Minglang Yin
(`minglang_yin@brown.edu`) and kept intentionally small so that the released
state dictionaries can be loaded without the authors' local source tree.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(widths: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_features, out_features in zip(widths[:-1], widths[1:]):
        layers.append(nn.Sequential(nn.Linear(in_features, out_features), nn.Tanh()))
    return nn.Sequential(*layers)


class opnn(nn.Module):
    """Two-branch DeepONet with an elementwise branch fusion."""

    def __init__(
        self,
        branch1_dim: list[int],
        branch2_dim: list[int],
        trunk_dim: list[int],
    ) -> None:
        super().__init__()
        self.z_dim = trunk_dim[-1]
        self._branch1 = _mlp(branch1_dim)
        self._branch2 = _mlp(branch2_dim)
        self._trunk = _mlp(trunk_dim)

    def forward(
        self,
        branch1: torch.Tensor,
        branch2: torch.Tensor,
        trunk: torch.Tensor,
    ) -> torch.Tensor:
        branch_embedding = self._branch1(branch1) * self._branch2(branch2)
        trunk_embedding = self._trunk(trunk)
        return torch.einsum("ij,kj->ik", branch_embedding, trunk_embedding)

    def loss(
        self,
        branch1: torch.Tensor,
        branch2: torch.Tensor,
        trunk: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return ((self(branch1, branch2, trunk) - target) ** 2).mean()
