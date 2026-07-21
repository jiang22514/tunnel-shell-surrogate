"""E_CT (Circular Top + Invert) outline generator.

Used by:
  - prepare_E_CT.py / build_E_CT_dataset.py for sanity checks
  - run_lddmm_E_CT.m re-implements the same algorithm in MATLAB
  - Rock_DDB_E_CT_*.py ABAQUS driver via shared geometry helper
  - refit_genieshan_E_CT.py for fitting real genieshan cross-section

Geometry (2-segment closed CW polyline, no straight walls, no fillets):

  Segment A (TOP):
    Arc of the main circle (centre = origin, radius R_top) from
    (-x_knee, y_knee) over the crown (0, R_top) down to (x_knee, y_knee).
    Spans an angle pi + 2*alpha where alpha = atan2(-y_knee, x_knee) > 0.

  Segment B (INVERT, concave-up "bowl"):
    Arc of the invert circle (centre = (0, y_c), radius R_inv) from
    (x_knee, y_knee) through (0, y_bottom) up to (-x_knee, y_knee).
    Spans an angle 2*beta where beta = pi/2 - atan2(y_c - y_knee, x_knee).

  Where:
    x_knee   = sqrt(R_top^2 - y_knee^2)
    R_inv    = R_inv_ratio * R_top
    y_c      = y_knee + sqrt(R_inv^2 - x_knee^2)   (invert centre above knee)
    y_bottom = y_c - R_inv

Polyline starts at (x_knee, y_knee) and proceeds CLOCKWISE:
  start  -> (0, y_bottom)   (down-left along invert arc)
         -> (-x_knee, y_knee)
         -> (-R_top, 0)
         -> (0, R_top)
         -> (R_top, 0)
         -> (x_knee, y_knee) = start

The tangent direction is generally discontinuous at the knee point. The
half-angle of the corner is small for realistic tunnels (< 5 deg for the
genie-shan tunnel at R_inv_ratio ~ 1.47, R_top ~ 6.6 m).
"""
import math


def make_ct_outline(R_top, y_knee, R_inv_ratio, n_total=120):
    """E_CT 2-segment CW polyline (invert arc first, then top arc).

    Args:
        R_top: m, top circle radius. Must be > 0.
        y_knee: m, knee y-coord. Must satisfy -R_top < y_knee < 0.
        R_inv_ratio: dimensionless, R_inv = R_inv_ratio * R_top.
                     Must satisfy R_inv > x_knee.
        n_total: int, number of sample points (uniform along arc length).

    Returns:
        list of (x, y) tuples, length = n_total, CW order,
        starting from (x_knee, y_knee).
    """
    if R_top <= 0:
        raise ValueError("R_top must be positive")
    if not (-R_top < y_knee < 0):
        raise ValueError("y_knee must be in (-R_top, 0); got %r (R_top=%r)"
                         % (y_knee, R_top))
    R_inv = R_inv_ratio * R_top
    x_knee = math.sqrt(R_top * R_top - y_knee * y_knee)
    inv_disc = R_inv * R_inv - x_knee * x_knee
    if inv_disc <= 0:
        raise ValueError("R_inv too small: invert arc cannot span the knees "
                         "(R_inv=%r, x_knee=%r)" % (R_inv, x_knee))
    y_c = y_knee + math.sqrt(inv_disc)
    y_bottom = y_c - R_inv

    # ---------------- Invert arc ----------------
    # At centre (0, y_c). Angles theta_inv measured from +x axis at centre.
    # (x_knee, y_knee) -> theta_R_inv = atan2(y_knee - y_c, x_knee)  in (-pi/2, 0).
    # bottom (0, y_bottom) -> -pi/2.
    # (-x_knee, y_knee) -> theta_L_inv = atan2(y_knee - y_c, -x_knee)  in (-pi, -pi/2).
    theta_R_inv = math.atan2(y_knee - y_c, x_knee)
    theta_L_inv = math.atan2(y_knee - y_c, -x_knee)
    delta_inv = theta_R_inv - theta_L_inv  # > 0, CW = decreasing theta
    L_inv = R_inv * delta_inv

    # ---------------- Top arc -------------------
    # At origin. Going CW from (-x_knee, y_knee) over (0, R_top) to (x_knee, y_knee).
    # (-x_knee, y_knee) -> theta_top_start = atan2(y_knee, -x_knee) wrapped to be
    #                      pi + alpha so that decreasing -> 0 passes the crown.
    alpha = math.atan2(-y_knee, x_knee)  # > 0
    theta_top_start = math.pi + alpha
    theta_top_end = -alpha
    delta_top = theta_top_start - theta_top_end  # = pi + 2*alpha
    L_top = R_top * delta_top

    L_total = L_inv + L_top

    pts = []
    for k in range(n_total):
        s = k * L_total / float(n_total)
        if s < L_inv:
            theta = theta_R_inv - (s / R_inv)
            x = R_inv * math.cos(theta)
            y = y_c + R_inv * math.sin(theta)
        else:
            s_top = s - L_inv
            theta = theta_top_start - (s_top / R_top)
            x = R_top * math.cos(theta)
            y = R_top * math.sin(theta)
        pts.append((float(x), float(y)))
    return pts


def ct_fillet_center(R_top, y_knee, R_inv_ratio, r_fillet):
    """Center of a fillet (radius r_fillet) tangent internally to the top
    circle (O, R_top) and the invert circle ((0, y_c), R_inv) at the RIGHT
    knee of the legacy (centre-above) E_CT horseshoe. Returns (px, py,
    y_c, R_inv). Mirror (-px, py) is the left-knee center.

    Concentric-fillet property: with the offset (lining) outline
    (R_top+t, R_inv+t, same y_c) and fillet radius r_fillet+t, this SAME
    center is tangent to both offset circles -> lining thickness stays
    exactly t through the haunch.
    """
    R_inv = R_inv_ratio * R_top
    x_knee = math.sqrt(R_top * R_top - y_knee * y_knee)
    inv_disc = R_inv * R_inv - x_knee * x_knee
    if inv_disc <= 0:
        raise ValueError("R_inv too small for fillet center")
    y_c = y_knee + math.sqrt(inv_disc)          # legacy: centre ABOVE knee
    a = R_top - r_fillet
    b = R_inv - r_fillet
    d = y_c
    py = (a * a - b * b + d * d) / (2.0 * d)
    disc = a * a - py * py
    if disc <= 0:
        raise ValueError("fillet radius too large: no tangent center "
                         "(r=%r, R_top=%r)" % (r_fillet, R_top))
    px = math.sqrt(disc)
    return px, py, y_c, R_inv


def _resample_closed(xs, ys, n_total):
    """Resample a fine closed polyline to n_total points uniform in arc len."""
    xs = list(xs) + [xs[0]]
    ys = list(ys) + [ys[0]]
    seg = [0.0]
    for i in range(1, len(xs)):
        seg.append(seg[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
    total = seg[-1]
    out = []
    j = 0
    for k in range(n_total):
        target = k * total / float(n_total)
        while j < len(seg) - 1 and seg[j + 1] < target:
            j += 1
        if seg[j + 1] == seg[j]:
            f = 0.0
        else:
            f = (target - seg[j]) / (seg[j + 1] - seg[j])
        out.append((float(xs[j] + f * (xs[j + 1] - xs[j])),
                    float(ys[j] + f * (ys[j + 1] - ys[j]))))
    return out


def make_ct_outline_filleted(R_top, y_knee, R_inv_ratio, r_fillet,
                             n_total=160):
    """Legacy E_CT horseshoe with both knees rounded by a fillet of radius
    r_fillet (de-singularization, doc_16 2026-06-13). CW order, starting at
    the invert-side right-knee tangent point, sampled uniformly by arc
    length -- a drop-in geometric replacement for make_ct_outline.

    For the inner (cavity) outline use r_fillet = t_lining; for the outer
    (interface) outline use the offset params (R_top+t, ...) with
    r_fillet = 2*t_lining (concentric center -> constant lining thickness).
    """
    if R_top <= 0:
        raise ValueError("R_top must be positive")
    if not (-R_top < y_knee < 0):
        raise ValueError("y_knee must be in (-R_top, 0)")
    if r_fillet <= 0:
        return make_ct_outline(R_top, y_knee, R_inv_ratio, n_total)

    px, py, y_c, R_inv = ct_fillet_center(R_top, y_knee, R_inv_ratio,
                                          r_fillet)
    a = R_top - r_fillet
    b = R_inv - r_fillet
    # tangent points (right knee), and mirrored left knee
    T1R = (R_top * px / a, R_top * py / a)              # on top circle
    T2R = (R_inv * px / b, y_c + R_inv * (py - y_c) / b)   # on invert circle
    T1L = (-T1R[0], T1R[1])
    T2L = (-T2R[0], T2R[1])

    NS = 800

    def arc(cx, cy, R, th0, th1, n):
        return [(cx + R * math.cos(th0 + (th1 - th0) * i / (n - 1)),
                 cy + R * math.sin(th0 + (th1 - th0) * i / (n - 1)))
                for i in range(n)]

    def minor_arc(cx, cy, r, pa, pb, n):
        a0 = math.atan2(pa[1] - cy, pa[0] - cx)
        a1 = math.atan2(pb[1] - cy, pb[0] - cx)
        if a1 - a0 > math.pi:
            a1 -= 2 * math.pi
        elif a1 - a0 < -math.pi:
            a1 += 2 * math.pi
        return arc(cx, cy, r, a0, a1, n)

    # --- CW order starting at T2R: invert (R->bottom->L), left fillet,
    #     top (L->crown->R), right fillet ---
    # invert arc on circle (0,y_c): CW = decreasing theta
    phiR = math.atan2(T2R[1] - y_c, T2R[0])
    phiL = math.atan2(T2L[1] - y_c, T2L[0])
    if phiL > phiR:
        phiL -= 2 * math.pi
    inv = arc(0.0, y_c, R_inv, phiR, phiL, NS)
    filL = minor_arc(-px, py, r_fillet, T2L, T1L, 80)
    # top arc on circle O: CW from T1L over crown to T1R (decreasing,
    # passing +pi/2). Take psiL in (pi/2, pi], psiR in [-pi, pi/2).
    psiL = math.atan2(T1L[1], T1L[0])
    psiR = math.atan2(T1R[1], T1R[0])
    if psiR > psiL:
        psiR -= 2 * math.pi
    top = arc(0.0, 0.0, R_top, psiL, psiR, NS)
    filR = minor_arc(px, py, r_fillet, T1R, T2R, 80)

    fine = inv + filL + top + filR
    xs = [p[0] for p in fine]
    ys = [p[1] for p in fine]
    return _resample_closed(xs, ys, n_total)


def ct_outer_params(R_top, y_knee, R_inv_ratio, t_lining):
    """Outer (rock-lining interface) params from inner + t_lining.
    Mirrors compute_outer_E_CT / MATLAB compute_outer."""
    R_inv = R_inv_ratio * R_top
    x_knee = math.sqrt(R_top * R_top - y_knee * y_knee)
    y_c = y_knee + math.sqrt(R_inv * R_inv - x_knee * x_knee)
    R_top_o = R_top + t_lining
    R_inv_o = R_inv + t_lining
    y_knee_o = (R_top_o ** 2 - R_inv_o ** 2 + y_c ** 2) / (2.0 * y_c)
    return R_top_o, y_knee_o, R_inv_o / R_top_o


def _seg_dist_and_nearest(points, poly):
    """Vectorised min distance from each point to a closed polyline + nearest
    segment index. points (N,2), poly (M,2). Chunked to bound memory."""
    import numpy as _np
    A = poly
    B = _np.roll(poly, -1, axis=0)
    AB = B - A
    L2 = (AB ** 2).sum(1) + 1e-30
    N = len(points)
    dmin = _np.empty(N)
    jmin = _np.empty(N, dtype=_np.int64)
    step = 2000
    for s in range(0, N, step):
        P = points[s:s + step]
        pa = P[:, None, :] - A[None]
        t = _np.clip((pa * AB[None]).sum(2) / L2[None], 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        d = _np.sqrt(((P[:, None, :] - proj) ** 2).sum(2))
        jmin[s:s + step] = d.argmin(1)
        dmin[s:s + step] = d.min(1)
    return dmin, jmin


def signed_dist_to_filleted(points, R_top, y_knee, R_inv_ratio, r_fillet,
                            n_poly=1200):
    """Signed distance (numpy) to the filleted E_CT outline; negative inside.
    points (N,2). Uses point-in-polygon for sign."""
    import numpy as _np
    from matplotlib.path import Path as _Path
    poly = _np.asarray(make_ct_outline_filleted(R_top, y_knee, R_inv_ratio,
                                                r_fillet, n_poly))
    d, _ = _seg_dist_and_nearest(_np.asarray(points), poly)
    inside = _Path(poly).contains_points(_np.asarray(points))
    return _np.where(inside, -d, d)


def classify_region_filleted(points, R_top, y_knee, R_inv_ratio, t_lining,
                             tol=0.0):
    """Region code (0=rock, 1=lining, 2=void) using the FILLETED inner cavity
    (r=t_lining) and outer interface (r=2*t_lining) outlines. tol>0 widens the
    lining band on both boundaries (for ABAQUS-node side-sets)."""
    import numpy as _np
    Rt_o, yk_o, ro = ct_outer_params(R_top, y_knee, R_inv_ratio, t_lining)
    d_in = signed_dist_to_filleted(points, R_top, y_knee, R_inv_ratio,
                                   t_lining)          # neg inside cavity
    d_out = signed_dist_to_filleted(points, Rt_o, yk_o, ro, 2.0 * t_lining)
    ROCK, LINING, VOID = 0, 1, 2
    region = _np.full(len(points), ROCK, dtype=_np.int64)
    region[d_out <= tol] = LINING          # at/inside interface (within tol)
    region[d_in < -tol] = VOID             # strictly inside cavity
    return region


def interface_normals_filleted(points, R_top, y_knee, R_inv_ratio, t_lining,
                               n_poly=1600):
    """Outward (cavity->rock) unit normals at band points, via the nearest
    segment of the FILLETED outer interface (r=2*t_lining). Correct at the
    knee fillets (unlike radial-from-circle)."""
    import numpy as _np
    Rt_o, yk_o, ro = ct_outer_params(R_top, y_knee, R_inv_ratio, t_lining)
    poly = _np.asarray(make_ct_outline_filleted(Rt_o, yk_o, ro,
                                                2.0 * t_lining, n_poly))
    AB = _np.roll(poly, -1, axis=0) - poly
    _, j = _seg_dist_and_nearest(_np.asarray(points), poly)
    seg = AB[j]
    nrm = _np.column_stack([seg[:, 1], -seg[:, 0]])
    nrm /= (_np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)
    centroid = poly.mean(0)
    out = _np.asarray(points) - centroid
    flip = (nrm * out).sum(1) < 0
    nrm[flip] *= -1.0
    return nrm


def ct_geometry_summary(R_top, y_knee, R_inv_ratio):
    """Return derived parameters for human inspection / logging."""
    R_inv = R_inv_ratio * R_top
    x_knee = math.sqrt(R_top * R_top - y_knee * y_knee)
    inv_disc = R_inv * R_inv - x_knee * x_knee
    if inv_disc <= 0:
        raise ValueError("R_inv too small: invert arc cannot span the knees")
    y_c = y_knee + math.sqrt(inv_disc)
    y_bottom = y_c - R_inv
    alpha_deg = math.degrees(math.atan2(-y_knee, x_knee))
    beta_deg = math.degrees(math.pi / 2.0 - math.atan2(y_c - y_knee, x_knee))
    # knee tangent mismatch (top arc tangent vs invert arc tangent at knee)
    # top arc tangent direction at (x_knee, y_knee), going CW (towards
    # decreasing theta) is perpendicular to radius pointing to centre at
    # origin, i.e. (sin(theta_top_end), -cos(theta_top_end)) at theta=-alpha:
    # = (sin(-alpha), -cos(-alpha)) = (-sin(alpha), -cos(alpha))
    # invert arc tangent at (x_knee, y_knee), going CW (towards decreasing
    # theta_inv) is perpendicular to radius from centre (0, y_c):
    # tangent = (sin(theta_R_inv), -cos(theta_R_inv)) at theta=theta_R_inv:
    theta_R_inv = math.atan2(y_knee - y_c, x_knee)
    t_top = (-math.sin(alpha_deg * math.pi / 180.0),
             -math.cos(alpha_deg * math.pi / 180.0))
    t_inv = (math.sin(theta_R_inv), -math.cos(theta_R_inv))
    dot = t_top[0] * t_inv[0] + t_top[1] * t_inv[1]
    dot = max(-1.0, min(1.0, dot))
    corner_half_angle_deg = math.degrees(math.acos(dot))
    return {
        'R_top': R_top,
        'y_knee': y_knee,
        'R_inv_ratio': R_inv_ratio,
        'R_inv': R_inv,
        'x_knee': x_knee,
        'y_c': y_c,
        'y_bottom': y_bottom,
        'alpha_top_deg': alpha_deg,
        'beta_inv_deg': beta_deg,
        'corner_half_angle_deg': corner_half_angle_deg,
        'span_x': 2 * R_top,
        'total_height': R_top - y_bottom,
    }


# =============================================================================
# CONCAVE-UP / STANDARD HORSESHOE VARIANT  (added 2026-05-26)
# =============================================================================
# Background: ct_outline above was originally INTENDED to be "concave-up bowl"
# (per the docstring of Segment B at line 16), but the formula
#     y_c = y_knee + sqrt(R_inv^2 - x_knee^2)
# places the invert centre ABOVE the knee, producing a *concave-down*
# (inverted-bowl) invert that dips down at x=0.  This is NOT what a real
# road/rail horseshoe tunnel looks like (cf. Genie Shan engineering case).
#
# The variant below uses y_c BELOW the knee, so the invert is genuinely
# concave-up (rises at center).  Same signature, same return type --
# meant to be a drop-in replacement after the path-C verification confirms
# DIMON + LDDMM still work on the corrected geometry.
#
# DO NOT REMOVE the legacy `make_ct_outline` yet -- existing 100-case datasets
# and trained models are tied to that geometry.  See doc_15 for migration.
# =============================================================================


def make_ct_outline_concave_up(R_top, y_knee, R_inv_ratio, n_total=120):
    """E_CT 2-segment CW polyline with CONCAVE-UP invert (standard horseshoe).

    Differs from `make_ct_outline` only in the sign of the sqrt term in y_c:
        legacy:  y_c = y_knee + sqrt(R_inv^2 - x_knee^2)   (centre ABOVE knee)
        new:     y_c = y_knee - sqrt(R_inv^2 - x_knee^2)   (centre BELOW knee)

    Args:
        R_top:        m, top arc radius (> 0)
        y_knee:       m, knee y-coord, must be in (-R_top, 0)
        R_inv_ratio:  dimensionless, R_inv = R_inv_ratio * R_top
                      must satisfy R_inv > x_knee
        n_total:      number of points (uniform along arc length)

    Returns:
        list of (x, y) tuples, length = n_total, CW order,
        starting from (x_knee, y_knee).

    Raises:
        ValueError if parameters violate constraints (including the new
        constraint that invert sagitta < |y_knee| so the invert top doesn't
        intrude above y=0).
    """
    if R_top <= 0:
        raise ValueError("R_top must be positive")
    if not (-R_top < y_knee < 0):
        raise ValueError("y_knee must be in (-R_top, 0); got %r (R_top=%r)"
                         % (y_knee, R_top))
    R_inv = R_inv_ratio * R_top
    x_knee = math.sqrt(R_top * R_top - y_knee * y_knee)
    inv_disc = R_inv * R_inv - x_knee * x_knee
    if inv_disc <= 0:
        raise ValueError("R_inv too small: invert arc cannot span the knees "
                         "(R_inv=%r, x_knee=%r)" % (R_inv, x_knee))

    # *** KEY DIFFERENCE: centre BELOW knee for concave-up invert ***
    y_c = y_knee - math.sqrt(inv_disc)
    y_top_invert = y_c + R_inv  # highest point of invert arc (at x=0)

    # Validity constraint: invert top must not rise above y=0 (where top arc
    # would intersect itself) and ideally not above y_knee + small margin.
    if y_top_invert >= 0:
        raise ValueError(
            "Invert sagitta too large: y_top_invert=%.3f >= 0 means the "
            "concave-up invert intrudes into the top half. Reduce R_inv_ratio "
            "or move y_knee deeper. (R_top=%r, y_knee=%r, R_inv_ratio=%r)"
            % (y_top_invert, R_top, y_knee, R_inv_ratio))

    # ---------------- Invert arc (concave-up) ----------------
    # Centre at (0, y_c) which is BELOW knee.
    # (x_knee, y_knee)  -> theta = atan2(y_knee - y_c,  x_knee) in (0, pi/2)
    # (0, y_top_invert) -> theta = pi/2  (top of upper half of circle)
    # (-x_knee, y_knee) -> theta = atan2(y_knee - y_c, -x_knee) in (pi/2, pi)
    # CW from start to end means INCREASING theta (since we go from right knee
    # up over the top of the invert arc back to left knee).
    theta_R_inv = math.atan2(y_knee - y_c, x_knee)
    theta_L_inv = math.atan2(y_knee - y_c, -x_knee)
    # Both should be in (0, pi) since y_knee > y_c
    delta_inv = theta_L_inv - theta_R_inv  # > 0
    L_inv = R_inv * delta_inv

    # ---------------- Top arc -------------------
    # Same as legacy (independent of invert orientation).
    alpha = math.atan2(-y_knee, x_knee)  # > 0
    theta_top_start = math.pi + alpha
    theta_top_end = -alpha
    delta_top = theta_top_start - theta_top_end  # = pi + 2*alpha
    L_top = R_top * delta_top

    L_total = L_inv + L_top

    pts = []
    for k in range(n_total):
        s = k * L_total / float(n_total)
        if s < L_inv:
            # Walk CW from (x_knee, y_knee) -- but for concave-up that means
            # going UP and over the top of the invert.  In our CW convention
            # (the closed-curve enclosing the cavity goes CW around the cavity),
            # the invert is traversed from (x_knee, y_knee) -> (0, y_top_inv) ->
            # (-x_knee, y_knee).  Theta INCREASES from theta_R_inv to theta_L_inv.
            theta = theta_R_inv + (s / R_inv)
            x = R_inv * math.cos(theta)
            y = y_c + R_inv * math.sin(theta)
        else:
            # Top arc -- same as legacy
            s_top = s - L_inv
            theta = theta_top_start - (s_top / R_top)
            x = R_top * math.cos(theta)
            y = R_top * math.sin(theta)
        pts.append((float(x), float(y)))
    return pts


def ct_geometry_summary_concave_up(R_top, y_knee, R_inv_ratio):
    """Same as ct_geometry_summary but for the concave-up variant."""
    R_inv = R_inv_ratio * R_top
    x_knee = math.sqrt(R_top * R_top - y_knee * y_knee)
    inv_disc = R_inv * R_inv - x_knee * x_knee
    if inv_disc <= 0:
        raise ValueError("R_inv too small")
    y_c = y_knee - math.sqrt(inv_disc)
    y_top_invert = y_c + R_inv
    sagitta = R_inv - math.sqrt(inv_disc)  # how much invert curves UP
    return {
        'R_top': R_top,
        'y_knee': y_knee,
        'R_inv_ratio': R_inv_ratio,
        'R_inv': R_inv,
        'x_knee': x_knee,
        'y_c': y_c,                 # centre BELOW knee (negative, large)
        'y_top_invert': y_top_invert,  # highest point of invert (was 'y_bottom')
        'invert_sagitta_up': sagitta,  # vertical rise of invert above y_knee
        'span_x': 2 * R_top,
        'total_height': R_top - y_top_invert,
        'valid_for_horseshoe': y_top_invert < 0,
    }


if __name__ == '__main__':
    import numpy as np
    cases = [
        ("E_CT mid-LHS", 5.0, -2.0, 2.0),
        ("Genie-shan fit (Day 0)", 6.631, -2.45, 1.473),
        ("Single-track small", 3.5, -1.4, 1.8),
        ("Deep bowl", 6.0, -2.5, 1.05),
        ("Near-flat invert", 4.0, -1.5, 3.0),
    ]
    for name, R, yk, ratio in cases:
        print(f"== {name}:  R_top={R}, y_knee={yk}, R_inv_ratio={ratio}")
        # --- legacy (inverted bowl)
        try:
            info = ct_geometry_summary(R, yk, ratio)
            print(f"  [LEGACY inverted-bowl]")
            for k, v in info.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        except Exception as e:
            print(f"  [LEGACY] FAILED: {e}")
        # --- new (concave-up)
        try:
            info = ct_geometry_summary_concave_up(R, yk, ratio)
            print(f"  [NEW concave-up]")
            for k, v in info.items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
            pts = make_ct_outline_concave_up(R, yk, ratio, 240)
            arr = np.array(pts)
            print(f"    outline_x_range: [{arr[:,0].min():.3f}, {arr[:,0].max():.3f}]")
            print(f"    outline_y_range: [{arr[:,1].min():.3f}, {arr[:,1].max():.3f}]")
        except Exception as e:
            print(f"  [NEW] FAILED: {e}")
        print()
