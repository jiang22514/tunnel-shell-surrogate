#!/usr/bin/env python3
"""E3: non-neural baselines (GP/kriging + PCE/polynomial) vs the neural main
configuration, for the reviewer-requested comparison. Protocol is IDENTICAL to
the primary configuration:
  split   : rng=RandomState(0); perm=permutation(100); te=perm[:20]; tr=perm[20:]
  inputs  : ct_params[:,:4] standardized with tr mean/std
  outputs : 8 profiles (eps_m, slope_ss, enn_c0, enn_slope, N, M, T_c0, T_slope),
            each standardized by a[tr][valid[tr]].mean()/std()
  eval    : same full chain as _e4_split_eval.py (eps_ss field recon via unroll
            points, N direct head, M via strain route with predicted T,
            sigma_t = N/t + 6|M|/t^2, FTK=2.39e6), plus GenieShan zero-shot
            (N direct + M strain route, truth side from _t3_plotdata.npz).
Baselines are per-station regressions (200 stations x 8 fields, 4-D input,
80 train samples; per station only cases with valid[ci,station] are used):
  gp  : ConstantKernel()*Matern(nu=2.5, ls=[1,1,1,1]) + WhiteKernel(1e-6),
        normalize_y=False, n_restarts_optimizer=2 (downgraded to 0 if a field
        exceeds 600 s; recorded in json)
  pce : PolynomialFeatures(degree=3) + RidgeCV(alphas=logspace(-6,2,20))
Genie M strain route is done analytically at profile level: all reconstructed
profiles are linear in (n-1/2), so sigma_ss = c0(s) + c1(s)*(n-1/2) exactly and
N = t*c0, M = t^2*c1/12 (equivalent to the point-level bin_profiles route; the
equivalence is verified numerically on a test case below).
Writes _e3_baselines.json."""
import json
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore', category=ConvergenceWarning)

sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel')
from unroll_ct import rotate_strain_to_frame
from shell_profiles import bin_profiles, integrate_NMQ, profile_rl2, NBINS

D = '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel/data_E_CT_L_T_v2_L8'
T_REF = 5.0
E_L, NU_L, A_L = 32.5e9, 0.2, 1.0e-5
lam = E_L * NU_L / ((1 + NU_L) * (1 - 2 * NU_L))
mu = E_L / (2 * (1 + NU_L))
FTK = 2.39e6  # C40
FIELD_TIMEOUT = 600.0  # s; beyond this a GP field drops to n_restarts=0

d1 = np.load(f'{D}/u1_dataset.npz')
UN = np.load(f'{D}/unroll_L8.npz')
PR = np.load(f'{D}/profiles_NM.npz')
TT = np.load(f'{D}/tprofiles_T.npz')
par = np.load(f'{D}/ct_params.npy')
PAR4 = par[:, :4]
V = PR['valid']

Ncase = par.shape[0]
rng = np.random.RandomState(0)
perm = rng.permutation(Ncase)
te = perm[:20]
tr = perm[20:]
print('split0: te=%s' % te.tolist(), flush=True)

p4m, p4s = PAR4[tr].mean(0), PAR4[tr].std(0)
Xtr = (PAR4[tr] - p4m) / p4s
Xte = (PAR4[te] - p4m) / p4s
GENIE_PAR = np.array([6.6183473293, -3.3373288219, 1.9160066845, 0.40])
Xg = (GENIE_PAR[None] - p4m) / p4s
Xpred = np.vstack([Xte, Xg])  # rows 0..19 test, row 20 genie

FIELDS = {f: PR[f] for f in ('eps_m', 'slope_ss', 'enn_c0', 'enn_slope', 'N', 'M')}
FIELDS.update({f: TT[f] for f in ('T_c0', 'T_slope')})
tcount = V[tr].sum(0)
pred_valid = tcount >= 2  # stations with a trainable per-station model
print('per-station valid train count: min %d max %d; pred_valid %d/%d'
      % (tcount.min(), tcount.max(), pred_valid.sum(), NBINS), flush=True)

e11, e22, e12 = d1['e11_data'], d1['e22_data'], d1['e12_data']
edges = np.linspace(0, 1, NBINS + 1)
t3 = np.load('/home/jiang/_t3_plotdata.npz')


# ---------------- baseline fits (per station, per field) ----------------
def fit_gp():
    preds, secs, restarts_used = {}, {}, {}
    for name, a in FIELDS.items():
        t0 = time.time()
        m = float(a[tr][V[tr]].mean())
        s = float(a[tr][V[tr]].std())
        pz = np.zeros((len(Xpred), NBINS))  # standardized; 0 -> field mean
        nres = 2
        for j in range(NBINS):
            msk = V[tr, j]
            if msk.sum() < 2:
                continue
            if nres and time.time() - t0 > FIELD_TIMEOUT:
                nres = 0
                print('  [%s] exceeded %.0f s at station %d -> n_restarts=0'
                      % (name, FIELD_TIMEOUT, j), flush=True)
            ker = (ConstantKernel() * Matern(nu=2.5, length_scale=[1, 1, 1, 1])
                   + WhiteKernel(1e-6))
            g = GaussianProcessRegressor(kernel=ker, normalize_y=False,
                                         n_restarts_optimizer=nres)
            g.fit(Xtr[msk], (a[tr, j][msk] - m) / s)
            pz[:, j] = g.predict(Xpred)
        preds[name] = pz * s + m
        secs[name] = round(time.time() - t0, 1)
        restarts_used[name] = nres
        print('gp field %-10s %6.1f s (final n_restarts=%d)'
              % (name, secs[name], nres), flush=True)
    return preds, dict(fit_seconds_per_field=secs,
                       final_n_restarts_per_field=restarts_used)


def fit_pce():
    pf = PolynomialFeatures(degree=3)
    Ftr = pf.fit_transform(Xtr)
    Fpred = pf.transform(Xpred)
    preds, secs = {}, {}
    for name, a in FIELDS.items():
        t0 = time.time()
        m = float(a[tr][V[tr]].mean())
        s = float(a[tr][V[tr]].std())
        pz = np.zeros((len(Xpred), NBINS))
        for j in range(NBINS):
            msk = V[tr, j]
            if msk.sum() < 2:
                continue
            r = RidgeCV(alphas=np.logspace(-6, 2, 20))
            r.fit(Ftr[msk], (a[tr, j][msk] - m) / s)
            pz[:, j] = r.predict(Fpred)
        preds[name] = pz * s + m
        secs[name] = round(time.time() - t0, 1)
        print('pce field %-10s %6.1f s' % (name, secs[name]), flush=True)
    return preds, dict(fit_seconds_per_field=secs, n_poly_features=Ftr.shape[1])


# ---------------- test-set evaluation (identical to _e4_split_eval.py) ----
def stats(x):
    return dict(median=float(np.median(x)), mean=float(np.mean(x)),
                max=float(np.max(x)))


def eval_test(P, tag):
    r_ess, rl_N_dir, rl_M = [], [], []
    sigt_rl2, sigt_peak, crack_pp = [], [], []
    route_check = None
    for k, ci in enumerate(te):
        tl_ = par[ci][3]
        mrow = UN['case'] == ci
        pt = UN['pt'][mrow]
        sh, nn = UN['shat'][mrow], UN['n'][mrow]
        tx, ty = UN['tx'][mrow], UN['ty'][mrow]
        wb = np.clip(np.digitize(sh, edges) - 1, 0, NBINS - 1)
        v = PR['valid'][ci]
        ok = v[wb]
        truth_ess = rotate_strain_to_frame(tx, ty, e11[ci][pt], e22[ci][pt], e12[ci][pt])
        rec_ess = P['eps_m'][k][wb] + (nn - 0.5) * P['slope_ss'][k][wb]
        r_ess.append(100 * np.sqrt(np.sum((rec_ess[ok] - truth_ess[ok]) ** 2)
                                   / np.sum(truth_ess[ok] ** 2)))
        rec_enn = P['enn_c0'][k][wb] + (nn - 0.5) * P['enn_slope'][k][wb]
        rec_T = P['T_c0'][k][wb] + (nn - 0.5) * P['T_slope'][k][wb]
        eth = A_L * (1 + NU_L) * (rec_T - T_REF)
        rec_sss = lam * (rec_ess + rec_enn - 2 * eth) + 2 * mu * (rec_ess - eth)
        prof = bin_profiles(sh[ok], nn[ok], dict(sig_ss=rec_sss[ok]))
        Np_st, Mp, _ = integrate_NMQ(prof, tl_)
        vv = prof['valid'] & v
        rl_M.append(profile_rl2(Mp, PR['M'][ci], vv))
        rl_N_dir.append(profile_rl2(P['N'][k], PR['N'][ci], v))
        # sigma_t: route policy N-direct + M-strain
        st_p = P['N'][k] / tl_ + 6 * np.abs(Mp) / tl_ ** 2
        st_t = PR['N'][ci] / tl_ + 6 * np.abs(PR['M'][ci]) / tl_ ** 2
        sigt_rl2.append(profile_rl2(st_p, st_t, vv))
        sigt_peak.append(100 * abs(st_p[vv].max() - st_t[vv].max()) / abs(st_t[vv].max()))
        crack_pp.append(100 * abs((st_p[vv] > FTK).mean() - (st_t[vv] > FTK).mean()))
        if k == 0:
            # verify analytic profile-level strain route == point-level route
            Na, Ma = analytic_strain_NM({f: P[f][k] for f in P}, tl_)
            route_check = profile_rl2(Ma, Mp, vv)
    print('[%s] analytic-vs-point M route rel-L2 on case te[0]: %.3f%%'
          % (tag, route_check), flush=True)
    out = dict(ess=stats(r_ess), N_direct=stats(rl_N_dir), M_strain=stats(rl_M),
               sigt=stats(sigt_rl2), sigt_peak=stats(sigt_peak),
               crack_pp=stats(crack_pp))
    print('[%s] test  ess %.2f | N-dir %.2f | M-strain %.2f | sigt %.2f | '
          'peak %.2f | crack %.1f pp (medians)'
          % (tag, out['ess']['median'], out['N_direct']['median'],
             out['M_strain']['median'], out['sigt']['median'],
             out['sigt_peak']['median'], out['crack_pp']['median']), flush=True)
    return out


# ---------------- genie zero-shot evaluation ----------------
def analytic_strain_NM(Pg, tl_):
    """Strain-route N/M from profile coefficients: everything is linear in
    (n-1/2), so sigma_ss = c0(s) + c1(s)*(n-1/2); N = t*c0, M = t^2*c1/12."""
    eth0 = A_L * (1 + NU_L) * (Pg['T_c0'] - T_REF)
    eth1 = A_L * (1 + NU_L) * Pg['T_slope']
    c0 = lam * (Pg['eps_m'] + Pg['enn_c0'] - 2 * eth0) + 2 * mu * (Pg['eps_m'] - eth0)
    c1 = lam * (Pg['slope_ss'] + Pg['enn_slope'] - 2 * eth1) + 2 * mu * (Pg['slope_ss'] - eth1)
    return tl_ * c0, tl_ ** 2 * c1 / 12


def eval_genie(Pg, tag):
    v_t = t3['v_t']
    N_rl2 = profile_rl2(Pg['N'], t3['N_truth'], v_t)
    _, Mg = analytic_strain_NM(Pg, GENIE_PAR[3])
    vv = v_t & pred_valid
    M_rl2 = profile_rl2(Mg, t3['M_truth'], vv)
    print('[%s] genie N-direct %.2f | M-strain %.2f' % (tag, N_rl2, M_rl2), flush=True)
    return dict(N_direct=float(N_rl2), M_strain=float(M_rl2))


# sanity: recompute the neural reference genie numbers from _t3_plotdata.npz
print('neural genie recomputed from _t3_plotdata: N %.2f (ref 1.78) | M %.2f (ref 5.80)'
      % (profile_rl2(t3['N_pred'], t3['N_truth'], t3['v_t']),
         profile_rl2(t3['M_pred_strain'], t3['M_truth'], t3['v_pT'])), flush=True)

results = {}
for tag, fitter in (('gp', fit_gp), ('pce', fit_pce)):
    print('=== fitting %s ===' % tag, flush=True)
    preds, settings = fitter()
    Pte = {f: preds[f][:20] for f in preds}
    Pg = {f: preds[f][20] for f in preds}
    results[tag] = dict(test=eval_test(Pte, tag), genie=eval_genie(Pg, tag),
                        settings=settings)

results['reference'] = dict(
    test=dict(ess_median=3.07, N_direct_median=3.19, M_strain_median=9.45,
              sigt_median=3.40),
    genie=dict(N_direct=1.78, M_strain=5.80),
    note='neural main configuration (shell+T heads, nobc/nopca, 3-seed '
         'ensemble), split RandomState(0)')
results['protocol'] = dict(
    split='RandomState(0); perm(100); te=perm[:20]; tr=perm[20:]',
    inputs='ct_params[:,:4] standardized with tr mean/std',
    outputs='8 profiles standardized by a[tr][valid[tr]].mean()/std()',
    gp='per-station GPR, C()*Matern(nu=2.5, ls=[1,1,1,1])+WhiteKernel(1e-6), '
       'normalize_y=False, n_restarts_optimizer=2',
    pce='per-station PolynomialFeatures(degree=3)+RidgeCV(logspace(-6,2,20))',
    eval='full chain identical to _e4_split_eval.py; genie strain route '
         'analytic at profile level (linear in n; N=t*c0, M=t^2*c1/12), '
         'mask v_t & pred_valid')

json.dump(results, open('/home/jiang/_e3_baselines.json', 'w'), indent=1)
print('E3_BASELINES_DONE', flush=True)
