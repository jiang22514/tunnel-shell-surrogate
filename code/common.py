"""Helpers shared by the three section families."""
import numpy as np
import torch.nn as nn

FIELDS = ['eps_m', 'slope_ss', 'enn_c0', 'enn_slope', 'N', 'M', 'T_c0', 'T_slope']
KH = 32          # harmonics of the station coordinate in the trunk input
NB = 200         # stations around the ring
FTK_C40 = 2.39e6  # characteristic tensile strength of C40 concrete [Pa]


def fourier(sc, kh=KH):
    """Periodic embedding of the aligned station coordinate: cos and sin of the harmonics 1..kh."""
    ang = 2*np.pi*np.asarray(sc)[:, None]*np.arange(1, kh + 1)[None, :]
    return np.c_[np.cos(ang), np.sin(ang)]


def plane_strain_constants(E, nu, alpha):
    """Coefficients of the hoop strain, the transverse strain and the temperature in the hoop stress
    (plane strain): lambda + 2 mu, lambda, -E alpha / (1 - 2 nu)."""
    lam = E*nu/((1 + nu)*(1 - 2*nu))
    mu = E/(2*(1 + nu))
    return lam + 2*mu, lam, -E*alpha/(1 - 2*nu)


def fibre(N, M, t):
    """Extreme-fibre stresses (extrados, intrados). n increases outwards and M > 0 is tension at the extrados."""
    return N/t + 6*M/t**2, N/t - 6*M/t**2


def weighted_rel_l2(pred, truth, w):
    """Relative L2 error [%] of each profile with station weights w (arc length of each station)."""
    return 100*np.sqrt((w*(pred - truth)**2).sum(1)/(w*truth**2).sum(1))


def stats(e):
    e = np.asarray(e)
    return dict(median=float(np.median(e)), mean=float(np.mean(e)), p90=float(np.quantile(e, .9)), max=float(np.max(e)))


class Res(nn.Module):
    """Pointwise network of the second stage: `depth` SiLU layers of `width` units and a zero-initialised output."""

    def __init__(self, nin, width=256, depth=4):
        super().__init__()
        layers = []
        d = nin
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        return self.out(self.body(z)).squeeze(-1)
