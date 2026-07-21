#!/usr/bin/env python3
"""(s, n) unroll parametrization of the E_CT lining (2026-07 shell route, Task 1.1).

Conventions (documented per plan review):
- Inner cavity outline from make_ct_outline_filleted(R_top, y_knee, R_inv_ratio,
  r_fillet=t_lining, N): CW order, native start = invert-side right-knee tangent
  point T2R (a consistent landmark across cases -- do NOT re-anchor).
- s = arc length along that outline from the native start; shat = s / perimeter.
- Outward normal (cavity -> lining) for the CW traversal is N = (-Ty, Tx);
  orientation is asserted per case via the polygon signed area (must be CW).
- n = (signed distance along N) / t_lining: n=0 intrados (cavity face),
  n=1 extrados (rock interface).
- Bending moment computed downstream with kernel (n - 1/2): M > 0 <=> tension
  at the EXTRADOS (rock side). Note: opposite of one common tunnel convention
  (inner-fiber tension positive) -- flip at figure time if needed.
- Landmark reparametrization: raw shat of the four analytic landmarks
  (invert bottom IB, left-knee tangent T1L, crown, right-knee tangent T1R)
  drifts up to ~10% of perimeter across cases; map them to fleet-fixed target
  values with a periodic piecewise-linear warp so that "the same shat means
  the same structural location" across cases.
"""
import numpy as np
from scipy.spatial import cKDTree
from ct_outline import make_ct_outline_filleted, ct_fillet_center

N_OUTLINE = 4000
LINING = 1


def outline_frame(R_top, y_knee, R_inv_ratio, t_lining, n_outline=N_OUTLINE):
    """Dense inner-cavity outline with arc length, tangent, outward normal,
    curvature. Returns dict(ol, s, perim, T, N, curv)."""
    ol = np.asarray(make_ct_outline_filleted(R_top, y_knee, R_inv_ratio,
                                             t_lining, n_outline), float)
    d = np.vstack([ol[1:] - ol[:-1], ol[:1] - ol[-1:]])
    seg = np.linalg.norm(d, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    perim = s[-1] + seg[-1]
    T = d / (seg[:, None] + 1e-15)
    area2 = float(np.sum(ol[:, 0] * np.roll(ol[:, 1], -1)
                         - np.roll(ol[:, 0], -1) * ol[:, 1]))
    if area2 >= 0:
        raise AssertionError('outline is not CW (signed area %.3g >= 0)' % area2)
    N = np.column_stack([-T[:, 1], T[:, 0]])          # outward for CW
    th = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    dth = np.empty_like(th)
    dth[1:] = th[1:] - th[:-1]
    dth[0] = th[0] - (th[-1] - 2 * np.pi * round((th[-1] - th[0]) / (2 * np.pi)))
    ds = 0.5 * (seg + np.roll(seg, 1))
    curv = np.abs(dth) / (ds + 1e-15)
    return dict(ol=ol, s=s, perim=perim, T=T, N=N, curv=curv)


def landmarks_raw(R_top, y_knee, R_inv_ratio, t_lining, frame):
    """Raw shat of [IB, T1L, crown, T1R] (analytic points projected on outline).
    Native start T2R is shat=0 by construction."""
    px, py, y_c, R_inv = ct_fillet_center(R_top, y_knee, R_inv_ratio, t_lining)
    a = R_top - t_lining
    T1R = (R_top * px / a, R_top * py / a)
    pts = np.array([[0.0, y_c - R_inv],       # invert bottom
                    [-T1R[0], T1R[1]],        # left-knee tangent T1L
                    [0.0, R_top],             # crown
                    T1R], float)              # right-knee tangent T1R
    _, idx = cKDTree(frame['ol']).query(pts)
    lm = frame['s'][idx] / frame['perim']
    if not np.all(np.diff(lm) > 0):
        raise AssertionError('landmark order violated: %s' % lm)
    return lm


def reparam(shat_raw, lm_raw, lm_tgt):
    """Periodic piecewise-linear warp: 0->0, lm_raw->lm_tgt, 1->1."""
    xs = np.concatenate([[0.0], lm_raw, [1.0]])
    ys = np.concatenate([[0.0], lm_tgt, [1.0]])
    if not (np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)):
        raise AssertionError('non-monotone landmark warp')
    return np.interp(shat_raw, xs, ys)


def unroll_points(R_top, y_knee, R_inv_ratio, t_lining, pts,
                  lm_tgt=None, frame=None, n_keep=(-0.05, 1.05)):
    """Project points into (shat, n) with local frame vectors.

    Returns dict: shat (reparametrized if lm_tgt given, else raw), shat_raw,
    n, That, Nhat, curv, keep (bool mask of points inside n_keep), perim,
    lm_raw."""
    if frame is None:
        frame = outline_frame(R_top, y_knee, R_inv_ratio, t_lining)
    pts = np.asarray(pts, float)
    _, idx = cKDTree(frame['ol']).query(pts)
    n = ((pts - frame['ol'][idx]) * frame['N'][idx]).sum(1) / t_lining
    shat_raw = frame['s'][idx] / frame['perim']
    lm = landmarks_raw(R_top, y_knee, R_inv_ratio, t_lining, frame)
    shat = reparam(shat_raw, lm, np.asarray(lm_tgt, float)) if lm_tgt is not None else shat_raw
    keep = (n > n_keep[0]) & (n < n_keep[1])
    return dict(shat=shat, shat_raw=shat_raw, n=n, That=frame['T'][idx],
                Nhat=frame['N'][idx], curv=frame['curv'][idx], keep=keep,
                perim=frame['perim'], lm_raw=lm)


def rotate_strain_to_frame(vx, vy, e11, e22, g12):
    """Normal strain along unit direction (vx,vy); g12 = ENGINEERING shear."""
    return vx * vx * e11 + vy * vy * e22 + vx * vy * g12


def rotate_stress_to_frame(vx, vy, s11, s22, s12):
    """Normal stress along unit direction (vx,vy); s12 = TENSOR shear (factor 2)."""
    return vx * vx * s11 + vy * vy * s22 + 2.0 * vx * vy * s12


def shear_stress_in_frame(tx, ty, nx, ny, s11, s22, s12):
    """sigma_sn = t^T sigma n."""
    return tx * nx * s11 + ty * ny * s22 + (tx * ny + ty * nx) * s12
