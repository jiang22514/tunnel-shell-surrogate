#!/usr/bin/env python3
"""Task 2.3: shell-operator evaluation chain.
 (a) reconstruct eps_ss(s,n) at eval-grid lining points from PREDICTED profiles
     vs truth field  [baseline: B+FF tensor rotated to eps_ss = like-for-like]
 (b) reconstruct sigma_ss (predicted profiles + TRUTH T profiles) -> N/M vs truth
     [baseline: task 1.4 B+FF integrated N/M]
Gate: recon eps_ss <= 10% AND median N/M <= 10% -> ready for Task 3 (GenieShan)."""
import sys, json
import numpy as np, torch
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/DIMON')
sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel')
from opnn import opnn
from unroll_ct import rotate_strain_to_frame
from shell_profiles import bin_profiles, integrate_NMQ, profile_rl2, NBINS

D = '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel/data_E_CT_L_T_v2_L8'
dev = torch.device('cuda')
T_REF = 5.0; E_L, NU_L, A_L = 32.5e9, 0.2, 1.0e-5
lam = E_L * NU_L / ((1 + NU_L) * (1 - 2 * NU_L)); mu = E_L / (2 * (1 + NU_L))

d1 = np.load(f'{D}/u1_dataset.npz'); d2 = np.load(f'{D}/u2_dataset.npz')
UN = np.load(f'{D}/unroll_L8.npz'); PR = np.load(f'{D}/profiles_NM.npz')
TT = np.load(f'{D}/tprofiles_T.npz')
# A2 (2026-07-13): PREDICTED temperature profiles for the test cases — ensemble of
# the T-head training runs (each saved its own test-set predictions). Primary
# metric now uses predicted T (same footing as the GenieShan zero-shot); truth-T
# kept as the oracle-T secondary report.
import glob as _g
_tf = sorted(_g.glob('/home/jiang/_stage3_T_pred_nc*_nobc_nopca.npz'))
TPRED = None
if _tf:
    _tp = [np.load(f) for f in _tf]
    TPRED = {k: np.mean([t[k] for t in _tp], 0) for k in ('T_c0', 'T_slope')}
    print('predicted-T ensemble: %d seed(s)' % len(_tp))
par = np.load(f'{D}/ct_params.npy')
import glob
_files = sorted(glob.glob('/home/jiang/_stage3_shell_pred_ncs*_nobc_nopca.npz')) or \
    ['/home/jiang/_stage3_shell_pred.npz']
_sps = [np.load(f) for f in _files]
print('ensembling %d prediction set(s): %s' % (len(_sps), [f.split('/')[-1] for f in _files]))
for _s in _sps[1:]:
    assert np.array_equal(_s['te'], _sps[0]['te'])
_SPd = {k: (np.mean([_s[k] for _s in _sps], 0) if k not in ('te', 'sc') else _sps[0][k])
        for k in _sps[0].files}


class _W:
    files = list(_sps[0].files)
    def __getitem__(self, k):
        return _SPd[k]


SP = _W()
te = SP['te'].astype(int); sc = SP['sc']
e11, e22, e12 = d1['e11_data'], d1['e22_data'], d1['e12_data']
lin = d1['lining_masks']
edges = np.linspace(0, 1, NBINS + 1)

# ---- fair 2D baseline (ff2 ensemble, nobc inputs) ----
Ncase = e11.shape[0]
rng = np.random.RandomState(0); perm = rng.permutation(Ncase)
assert np.array_equal(perm[:20], te)
tr = perm[20:]
import glob as _g2
_pf = sorted(_g2.glob('/home/jiang/_stage2B_ff2_pred_s*.npz'))
_ps = [np.load(p) for p in _pf]
bff = {nm: np.mean([p[nm] for p in _ps], 0) for nm in ('e11', 'e22', 'g12')}

r_shell_ess, r_bff_ess, rl_N, rl_M, rl_Nd, rl_Md = [], [], [], [], [], []
NpS_all, MpS_all, vv_all = [], [], []
figN, axN = plt.subplots(4, 5, figsize=(20, 12), sharex=True)
figM, axM = plt.subplots(4, 5, figsize=(20, 12), sharex=True)
bffNM = json.load(open('/home/jiang/_t1_bff_NM_result.json'))
bff_by_case = {c['case']: c for c in bffNM['cases']}
res = []
for k, ci in enumerate(te):
    tl_ = par[ci][3]
    mrow = UN['case'] == ci
    pt = UN['pt'][mrow]; sh = UN['shat'][mrow]; nn = UN['n'][mrow]
    tx, ty = UN['tx'][mrow], UN['ty'][mrow]
    wb = np.clip(np.digitize(sh, edges) - 1, 0, NBINS - 1)
    v = PR['valid'][ci]
    ok = v[wb]
    truth_ess = rotate_strain_to_frame(tx, ty, e11[ci][pt], e22[ci][pt], e12[ci][pt])
    # (a) field reconstruction
    rec_ess = SP['eps_m'][k][wb] + (nn - 0.5) * SP['slope_ss'][k][wb]
    bff_ess = rotate_strain_to_frame(tx, ty, bff['e11'][k][pt], bff['e22'][k][pt], bff['g12'][k][pt])
    r_shell_ess.append(100 * np.sqrt(np.sum((rec_ess[ok] - truth_ess[ok]) ** 2) / np.sum(truth_ess[ok] ** 2)))
    r_bff_ess.append(100 * np.sqrt(np.sum((bff_ess[ok] - truth_ess[ok]) ** 2) / np.sum(truth_ess[ok] ** 2)))
    # (b) N/M from predicted profiles + truth T profiles
    rec_enn = SP['enn_c0'][k][wb] + (nn - 0.5) * SP['enn_slope'][k][wb]
    if TPRED is not None:
        rec_T = TPRED['T_c0'][k][wb] + (nn - 0.5) * TPRED['T_slope'][k][wb]
    else:
        rec_T = TT['T_c0'][ci][wb] + (nn - 0.5) * TT['T_slope'][ci][wb]
    eth = A_L * (1 + NU_L) * (rec_T - T_REF)
    rec_sss = lam * (rec_ess + rec_enn - 2 * eth) + 2 * mu * (rec_ess - eth)
    prof = bin_profiles(sh[ok], nn[ok], dict(sig_ss=rec_sss[ok]))
    Np, Mp, _ = integrate_NMQ(prof, tl_)
    vv = prof['valid'] & v
    rl_N.append(profile_rl2(Np, PR['N'][ci], vv)); rl_M.append(profile_rl2(Mp, PR['M'][ci], vv))
    NpS_all.append(Np.copy()); MpS_all.append(Mp.copy()); vv_all.append(vv.copy())
    # direct internal-force heads (if trained): second route
    if 'N' in SP.files:
        rl_Nd.append(profile_rl2(SP['N'][k], PR['N'][ci], v))
        rl_Md.append(profile_rl2(SP['M'][k], PR['M'][ci], v))
        Np, Mp, vv = SP['N'][k], SP['M'][k], v  # plot the direct route
    res.append(dict(case=int(ci), shell_ess=r_shell_ess[-1], bff_ess=r_bff_ess[-1],
                    rlN=rl_N[-1], rlM=rl_M[-1],
                    bff_rlN=bff_by_case[int(ci)]['rlN'], bff_rlM=bff_by_case[int(ci)]['rlM']))
    aN, aM = axN.flat[k], axM.flat[k]
    aN.plot(sc[vv], PR['N'][ci][vv] / 1e3, 'k-', lw=1.3)
    aN.plot(sc[vv], Np[vv] / 1e3, 'r--', lw=1.1)
    aM.plot(sc[vv], PR['M'][ci][vv] / 1e3, 'k-', lw=1.3)
    aM.plot(sc[vv], Mp[vv] / 1e3, 'r--', lw=1.1)
    aN.set_title('case %d relL2 %.1f%% (B+FF %.0f%%)' % (ci, rl_N[-1], bff_by_case[int(ci)]['rlN']), fontsize=9)
    aM.set_title('case %d relL2 %.1f%% (B+FF %.0f%%)' % (ci, rl_M[-1], bff_by_case[int(ci)]['rlM']), fontsize=9)
for f_, lab in ((figN, 'N(s) kN/m: truth black / SHELL red'), (figM, 'M(s) kNm/m: truth black / SHELL red')):
    f_.suptitle(lab); f_.tight_layout()
figN.savefig('/home/jiang/experiment_reports/figs/t2_shell_N.png', dpi=100)
figM.savefig('/home/jiang/experiment_reports/figs/t2_shell_M.png', dpi=100)

# plot-data export for the paper figures (gen_figs_v2.py)
np.savez('/home/jiang/_t2_plotdata.npz', te=te, sc=sc,
         valid=np.array([PR['valid'][ci] for ci in te]),
         N_truth=np.array([PR['N'][ci] for ci in te]),
         M_truth=np.array([PR['M'][ci] for ci in te]),
         N_pred=SP['N'], M_pred_direct=SP['M'],
         N_strain=np.array(NpS_all), M_strain=np.array(MpS_all), vv=np.array(vv_all),
         eps_m_pred=SP['eps_m'], slope_pred=SP['slope_ss'],
         enn0_pred=SP['enn_c0'], enn1_pred=SP['enn_slope'],
         rl_N=np.array(rl_N), rl_M=np.array(rl_M),
         r_shell_ess=np.array(r_shell_ess), r_bff_ess=np.array(r_bff_ess))
arr = lambda x: np.array(x)
r_shell_ess, r_bff_ess, rl_N, rl_M = map(arr, (r_shell_ess, r_bff_ess, rl_N, rl_M))
print('\n=== SHELL OPERATOR vs baselines (20 test cases) ===')
print('eps_ss field recon: SHELL med %.2f%% mean %.2f%% max %.2f%%'
      % (np.median(r_shell_ess), r_shell_ess.mean(), r_shell_ess.max()))
print('eps_ss field:        B+FF med %.2f%% mean %.2f%% (like-for-like baseline)'
      % (np.median(r_bff_ess), r_bff_ess.mean()))
print('N (strain route): SHELL med %.2f%% mean %.2f%% max %.2f%%   (B+FF med %.2f%%)'
      % (np.median(rl_N), rl_N.mean(), rl_N.max(), bffNM['med_rlN']))
print('M (strain route): SHELL med %.2f%% mean %.2f%% max %.2f%%   (B+FF med %.2f%%)'
      % (np.median(rl_M), rl_M.mean(), rl_M.max(), bffNM['med_rlM']))
if rl_Nd:
    rl_Nd = np.array(rl_Nd); rl_Md = np.array(rl_Md)
    print('N (direct head):  SHELL med %.2f%% mean %.2f%% max %.2f%%'
          % (np.median(rl_Nd), rl_Nd.mean(), rl_Nd.max()))
    print('M (direct head):  SHELL med %.2f%% mean %.2f%% max %.2f%%'
          % (np.median(rl_Md), rl_Md.mean(), rl_Md.max()))
# route policy (consistent across v1/v2/v3/ensemble, not per-test-draw picking):
# N via direct head (integral quantity, learns easily); M via strain route
# (physics chain distributes error; direct-M head consistently worse).
best_N = np.median(rl_Nd) if len(rl_Nd) else np.median(rl_N)
best_M = np.median(rl_M)
gate = bool(np.median(r_shell_ess) <= 10 and best_N <= 10 and best_M <= 10)
print('GATE (eps_ss<=10%% and N,M med<=10%%):', 'PASS -> Task 3 ready' if gate else 'NOT MET')
json.dump(dict(cases=res, shell_ess_med=float(np.median(r_shell_ess)),
               bff_ess_med=float(np.median(r_bff_ess)),
               med_rlN=float(np.median(rl_N)), med_rlM=float(np.median(rl_M)),
               med_rlN_direct=float(np.median(rl_Nd)) if len(rl_Nd) else None,
               med_rlM_direct=float(np.median(rl_Md)) if len(rl_Md) else None,
               gate_pass=gate), open('/home/jiang/_t2_shell_eval_result.json', 'w'), indent=1)
print('T2_SHELL_EVAL_DONE', flush=True)
