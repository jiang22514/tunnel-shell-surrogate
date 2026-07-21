#!/usr/bin/env python3
"""Per-(case, shat-bin) polynomial fit in n + analytic thickness integration
(Task 1.3 machinery, shared by truth-side and prediction-side).

Quantity definitions (docstring per plan):
  N(s) = t * int_0^1 sigma_ss dn          [N/m per unit longitudinal length]
  M(s) = t^2 * int_0^1 sigma_ss (n-1/2) dn ; M > 0 <=> tension at EXTRADOS
  Q(s) = t * int_0^1 sigma_sn dn
Flat resultants about n=1/2 (no curved-beam (1+z/R) metric) -- consistent
between truth and prediction; note the omission when comparing to Abaqus SF/SM.
Fits use interior points n in [NLO, NHI] only (griddata face-flattening guard),
then integrate the polynomial analytically over the FULL [0,1]."""
import numpy as np

NBINS = 200
NLO, NHI = 0.07, 0.93
MIN_PTS, MIN_DISTINCT_N = 8, 3

# integrals over [0,1] of n^k and n^k*(n-1/2), k=0..2
_INT = np.array([1.0, 0.5, 1.0 / 3.0])
_INT_M = np.array([0.0, 1.0 / 12.0, 1.0 / 12.0])


def bin_profiles(shat, n, fields, nbins=NBINS, deg=2, n_range=(NLO, NHI)):
    """Fit each field (dict name->values) as poly(deg) in n per shat bin.
    Returns dict: 'sc' bin centers, 'valid' [nbins] bool, and for each field
    'coef_<name>' [nbins, deg+1] (c0 + c1*n + c2*n^2 ...)."""
    inside = (n >= n_range[0]) & (n <= n_range[1])
    edges = np.linspace(0, 1, nbins + 1)
    sc = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(shat, edges) - 1, 0, nbins - 1)
    out = {'sc': sc, 'valid': np.zeros(nbins, bool)}
    for name in fields:
        out['coef_' + name] = np.zeros((nbins, deg + 1))
    for b in range(nbins):
        m = inside & (which == b)
        if m.sum() < MIN_PTS or len(np.unique(np.round(n[m], 3))) < MIN_DISTINCT_N:
            continue
        A = np.vander(n[m], deg + 1, increasing=True)
        AtA = A.T @ A
        out['valid'][b] = True
        for name, val in fields.items():
            c = np.linalg.solve(AtA, A.T @ val[m])
            out['coef_' + name][b] = c
    return out


def integrate_NMQ(prof, t_lin, key_ss='sig_ss', key_sn='sig_sn'):
    """N, M, Q from fitted coefficients (analytic over [0,1])."""
    css = prof['coef_' + key_ss]; deg1 = css.shape[1]
    N = t_lin * css @ _INT[:deg1]
    M = t_lin ** 2 * css @ _INT_M[:deg1]
    Q = None
    if 'coef_' + key_sn in prof:
        Q = t_lin * prof['coef_' + key_sn] @ _INT[:deg1]
    return N, M, Q


def strain_labels(prof, key='eps_ss'):
    """eps_m (value at n=1/2) and dimensionless slope (=kappa*t convention)
    from a deg-1 or deg-2 fit of eps_ss."""
    c = prof['coef_' + key]
    nh = np.array([1.0, 0.5, 0.25])[:c.shape[1]]
    eps_m = c @ nh                      # value at n = 1/2
    slope = c[:, 1] + (c[:, 2] if c.shape[1] > 2 else 0.0)  # d/dn at n=1/2
    return eps_m, slope


def profile_rl2(pred, truth, valid):
    """rel-L2 between two profiles on valid bins."""
    p, t = pred[valid], truth[valid]
    return 100 * np.sqrt(np.sum((p - t) ** 2) / (np.sum(t ** 2) + 1e-30))
