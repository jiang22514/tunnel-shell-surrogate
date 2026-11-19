"""DeepONet branch-trunk network used by the released profile operators.

The architecture follows the DIMON training code by Minglang Yin and co-workers
(Yin et al., 2024, "A scalable framework for learning the geometry-dependent
solution operators of partial differential equations"). It is kept small so
that the released state dictionaries load without any other source tree.
"""
import torch
import torch.nn as nn


def _stack(dims):
    """Fully connected tanh layers; dims[0] is the input width."""
    modules = []
    in_channels = dims[0]
    for h_dim in dims[1:]:
        modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), nn.Tanh()))
        in_channels = h_dim
    return nn.Sequential(*modules)


class opnn(nn.Module):
    """Two branch networks (multiplied) and one trunk network; the output is their inner product."""

    def __init__(self, branch1_dim, branch2_dim, trunk_dim):
        super().__init__()
        self.z_dim = trunk_dim[-1]
        self._branch1 = _stack(branch1_dim)
        self._branch2 = _stack(branch2_dim)
        self._trunk = _stack(trunk_dim)

    def forward(self, f, f_bc, x):
        """f: (M, dim_f), f_bc: (M, dim_b), x: (N, dim_x); returns (M, N)."""
        y_br = self._branch1(f)*self._branch2(f_bc)
        y_tr = self._trunk(x)
        return torch.einsum("ij,kj->ik", y_br, y_tr)

    def loss(self, f, f_bc, x, y):
        return ((self.forward(f, f_bc, x) - y)**2).mean()
