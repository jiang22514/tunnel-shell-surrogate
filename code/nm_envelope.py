#!/usr/bin/env python3
"""JTG 3370.1 / JTG D70-style plain-concrete eccentric-compression capacity for a
rectangular lining section (per metre run), and per-station safety factors.

Convention: N > 0 = COMPRESSION here (tunnel practice); the caller converts from
the dataset convention if needed. M in N*m per metre, N in N per metre.

Plain concrete rectangular section b x h (b = 1 m):
  small eccentricity (e0 <= 0.2 h): compressive capacity with eccentricity factor
      N_u = phi * alpha_e * R_a * b * h,  alpha_e = 1 - 2 e0 / h   (uniaxial block)
  larger eccentricity: capacity limited by flexural tension
      N_u = phi * 1.75 * R_l * b * h / (6 e0 / h - 1)
  (highway-tunnel code form for plain concrete linings; R_a = axial compressive
   design strength, R_l = flexural tensile design strength.)
Safety factor at a station: K = N_u(e0) / N with e0 = M / N.
C30: R_a = 13.8 MPa, R_l = 1.35 MPa (design values, JTG D70 Table); phi = 1
(section check, buckling not relevant for a continuously bedded ring).
NOTE for the paper: verify strength values against the governing code edition
before submission."""
import numpy as np

R_A = 13.8e6   # Pa, C30 axial compressive design strength
R_L = 1.35e6   # Pa, C30 flexural tensile design strength
PHI = 1.0


def capacity_N(e0, h, b=1.0, Ra=R_A, Rl=R_L, phi=PHI):
    """Ultimate compressive capacity N_u [N/m] at eccentricity e0 [m]."""
    e0 = np.abs(np.asarray(e0, float))
    small = e0 <= 0.2 * h
    Nu = np.empty_like(e0)
    Nu[small] = phi * (1.0 - 2.0 * e0[small] / h) * Ra * b * h
    ratio = 6.0 * e0[~small] / h - 1.0
    Nu[~small] = phi * 1.75 * Rl * b * h / np.maximum(ratio, 1e-9)
    return Nu


def envelope_curve(h, b=1.0, n=400):
    """(N_u, M_u) pairs tracing the capacity envelope for plotting."""
    e0 = np.concatenate([np.linspace(1e-4, 0.2 * h, n // 2),
                         np.linspace(0.2 * h, 3.0 * h, n // 2)])
    Nu = capacity_N(e0, h, b)
    return Nu, Nu * e0


def safety_factors(N, M, h, b=1.0, compression_positive=True):
    """Per-station K = N_u/N for compressive stations; tensile stations get K
    from the flexural-tension branch on |N| (reported separately)."""
    N = np.asarray(N, float); M = np.asarray(M, float)
    sgn = 1.0 if compression_positive else -1.0
    Nc = sgn * N
    K = np.full(N.shape, np.nan)
    comp = Nc > 1e3          # > 1 kN/m compression
    e0 = np.abs(M[comp] / Nc[comp])
    K[comp] = capacity_N(e0, h, b) / Nc[comp]
    return K, comp


if __name__ == '__main__':
    # self-test: pure axial C30, h=0.4 -> N_u = 13.8e6*0.4 = 5.52 MN/m
    assert abs(capacity_N(0.0, 0.4) - 5.52e6) < 1e3
    # e0 = 0.1h = 0.04 m -> alpha = 0.8 -> 4.416 MN/m
    assert abs(capacity_N(0.04, 0.4) - 4.416e6) < 1e3
    # e0 = h/2 = 0.2 (large ecc): 1.75*1.35e6*0.4/(6*0.5-1) = 472.5 kN/m
    assert abs(capacity_N(0.2, 0.4) - 1.75 * 1.35e6 * 0.4 / 2.0) < 1e3
    Nu, Mu = envelope_curve(0.4)
    print('self-test OK; envelope N_u range [%.0f, %.0f] kN/m, M_u max %.1f kNm/m'
          % (Nu.min() / 1e3, Nu.max() / 1e3, Mu.max() / 1e3))
