#!/usr/bin/env python3
"""Task 3: GenieShan ZERO-SHOT internal forces — the paper's final gate.
Truth side: _genie_L8full_{nodes,elems}.csv (U/NT nodes; E/S centroids),
authoritative params from genieshan_baseline_params.txt.
Prediction side: 3-seed shell-operator ensemble (v3 config) + 3-seed T-profile
heads; branch features from the x_uni reshoot mapping (_genie_xuni_mapping.mat).
Reports: fully-predicted N/M (predicted T) AND oracle-T variant (decomposition),
plus zero-cost validations (constitutive round-trip on genie, mask sanity)."""
import sys, json
import numpy as np, torch, scipy.io
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/DIMON')
sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel')
from opnn import opnn
from unroll_ct import (outline_frame, unroll_points, rotate_strain_to_frame,
                       rotate_stress_to_frame, shear_stress_in_frame)
from ct_outline import classify_region_filleted
from shell_profiles import bin_profiles, integrate_NMQ, strain_labels, profile_rl2, NBINS

D = '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel/data_E_CT_L_T_v2_L8'
W = '/mnt/d/tunnel_abaqus_workspace'
dev = torch.device('cuda')
T_REF = 5.0; E_L, NU_L, A_L = 32.5e9, 0.2, 1.0e-5
lam = E_L * NU_L / ((1 + NU_L) * (1 - 2 * NU_L)); mu = E_L / (2 * (1 + NU_L))
edges = np.linspace(0, 1, NBINS + 1)

# ---- authoritative genie params ----
gp = {}
for line in open(f'{W}/genieshan_run/genieshan_baseline_params.txt'):
    k, v = line.strip().split('=')
    gp[k] = float(v)
R_g, yk_g, ra_g, tl_g = gp['R_top'], gp['y_knee'], gp['R_inv_ratio'], gp['t_lining']
print('genie params: R_top %.3f y_knee %.3f ratio %.3f t %.3f' % (R_g, yk_g, ra_g, tl_g), flush=True)

# ---- truth side ----
UN = np.load(f'{D}/unroll_L8.npz'); lm_tgt = UN['lm_tgt']
E = np.genfromtxt(f'{W}/_genie_L8full_elems.csv', delimiter=',', names=True)
Nn = np.genfromtxt(f'{W}/_genie_L8full_nodes.csv', delimiter=',', names=True)
cxy = np.column_stack([E['Xc'], E['Yc']])
is_lin = classify_region_filleted(cxy, R_g, yk_g, ra_g, tl_g) == 1
print('genie lining elems: %d / %d' % (is_lin.sum(), len(cxy)), flush=True)
assert 3000 < is_lin.sum() < 15000, 'lining count implausible - params/classification issue'
cxy = cxy[is_lin]
fr = outline_frame(R_g, yk_g, ra_g, tl_g)
u = unroll_points(R_g, yk_g, ra_g, tl_g, cxy, lm_tgt=lm_tgt, frame=fr)
k = u['keep']; print('keep %.4f n=[%.3f,%.3f] perim %.2f' %
                     (k.mean(), u['n'][k].min(), u['n'][k].max(), u['perim']), flush=True)
tx, ty = u['That'][k].T; nx, ny = u['Nhat'][k].T
sh, nn = u['shat'][k], u['n'][k]
e11, e22, g12, s11, s22, s12 = (E[c][is_lin][k] for c in ('E11', 'E22', 'E12', 'S11', 'S22', 'S12'))
nxy = np.column_stack([Nn['X'], Nn['Y']])
Tc = griddata(nxy, Nn['NT'], cxy[k], 'linear')
b = ~np.isfinite(Tc)
if b.any():
    Tc[b] = griddata(nxy, Nn['NT'], cxy[k][b], 'nearest')
# zero-cost validation: constitutive round-trip on genie lining
eth = A_L * (1 + NU_L) * (Tc - T_REF)
hx, hy = e11 - eth, e22 - eth
rt = [100 * np.sqrt(np.sum((p - t) ** 2) / np.sum(t ** 2)) for p, t in
      ((lam * (hx + hy) + 2 * mu * hx, s11), (lam * (hx + hy) + 2 * mu * hy, s22), (mu * g12, s12))]
print('genie constitutive round-trip: s11 %.2f%% s22 %.2f%% s12 %.2f%%' % tuple(rt), flush=True)
assert max(rt) < 3.0, 'genie constitutive round-trip failed - params/geometry suspect'

sig_ss = rotate_stress_to_frame(tx, ty, s11, s22, s12)
sig_sn = shear_stress_in_frame(tx, ty, nx, ny, s11, s22, s12)
eps_ss_t = rotate_strain_to_frame(tx, ty, e11, e22, g12)
prof_t = bin_profiles(sh, nn, dict(sig_ss=sig_ss, sig_sn=sig_sn, eps_ss=eps_ss_t, T=Tc))
v_t = prof_t['valid']
N_t, M_t, Q_t = integrate_NMQ(prof_t, tl_g)
print('genie truth: N=[%.0f,%.0f]kN M=[%.2f,%.2f]kNm valid %d/%d' %
      (N_t[v_t].min() / 1e3, N_t[v_t].max() / 1e3, M_t[v_t].min() / 1e3, M_t[v_t].max() / 1e3,
       v_t.sum(), NBINS), flush=True)

# ---- prediction side: branch features ----
d1 = np.load(f'{D}/u1_dataset.npz'); d2 = np.load(f'{D}/u2_dataset.npz')
PR = np.load(f'{D}/profiles_NM.npz'); PAR4 = np.load(f'{D}/ct_params.npy')[:, :4]
x_uni = d1['x_uni']; x_mesh = d1['all_tunnel_def_grids']
bc1 = d1['bc_u1_top']; bc2 = d2['bc_u2_top']
Ncase = x_mesh.shape[0]
rng = np.random.RandomState(0); perm = rng.permutation(Ncase); tr = perm[20:]
gm = scipy.io.loadmat(f'{W}/_genie_xuni_mapping.mat')
assert np.allclose(gm['eval_points'], x_uni, atol=1e-6), 'genie mapping not on x_uni'
dx_g = (gm['mapped_pts'] - x_uni)[None]
pm = 10
dx = x_mesh - x_uni
mx = dx[tr][..., 0].mean(0); my = dx[tr][..., 1].mean(0)
px = PCA(n_components=pm).fit(dx[tr][..., 0] - mx); py = PCA(n_components=pm).fit(dx[tr][..., 1] - my)
p4m, p4s = PAR4[tr].mean(0), PAR4[tr].std(0)
par_g = (np.array([[R_g, yk_g, ra_g, tl_g]]) - p4m) / p4s
f_g = np.ones((1, pm * 2))  # nopca primary: params-only encoding
print('genie PCA branch coords (first 5): %s ; params norm: %s' %
      (np.round(f_g[0, :5], 2), np.round(par_g[0], 2)), flush=True)
# nobc: branch2 = normalized section params (no solution-derived inputs)
bc_g = par_g
fg_t = torch.tensor(f_g, dtype=torch.float32, device=dev)
bcg_t = torch.tensor(bc_g, dtype=torch.float32, device=dev)

# ---- load ensembles & predict profiles ----
PRf = {f: PR[f] for f in ('eps_m', 'slope_ss', 'enn_c0', 'enn_slope', 'N', 'M')}
TT = np.load(f'{D}/tprofiles_T.npz')
TTf = {'T_c0': TT['T_c0'], 'T_slope': TT['T_slope']}
sc = PR['sc']; valid = PR['valid']
KH = 32
ks = np.arange(1, KH + 1)
ang = 2 * np.pi * sc[:, None] * ks[None, :]
x_st = np.concatenate([np.cos(ang), np.sin(ang)], 1)
x_t = torch.tensor(x_st, dtype=torch.float32, device=dev)
db1 = [pm * 2, 100, 100, 100]
db2 = [bc_g.shape[1], 150, 150, 150, 100]
dtk = [x_st.shape[1], 100, 100, 100, 100, 100]


def predict_ensemble(model_paths, fields_src):
    preds = {f: [] for f in fields_src}
    for mp in model_paths:
        ckm = torch.load(mp, map_location=dev, weights_only=False)
        for f, a in fields_src.items():
            net = opnn(db1, db2, dtk).to(dev).float()
            net.load_state_dict(ckm[f]); net.eval()
            m = float(a[tr][valid[tr]].mean()); s = float(a[tr][valid[tr]].std())
            with torch.no_grad():
                preds[f].append(net(fg_t, bcg_t, x_t).cpu().numpy()[0] * s + m)
    return {f: np.mean(p, 0) for f, p in preds.items()}


shell = predict_ensemble([f'/home/jiang/_stage3_shell_model_ncs{i}_nobc_nopca.pt' for i in range(3)], PRf)
Theads = predict_ensemble([f'/home/jiang/_stage3_T_model_nc{i}_nobc_nopca.pt' for i in range(3)], TTf)
print('profiles predicted (zero-shot)', flush=True)

# ---- assemble N/M ----
wb = np.clip(np.digitize(sh, edges) - 1, 0, NBINS - 1)
ok = v_t[wb]
rec_ess = shell['eps_m'][wb] + (nn - 0.5) * shell['slope_ss'][wb]
rec_enn = shell['enn_c0'][wb] + (nn - 0.5) * shell['enn_slope'][wb]
r_ess_field = 100 * np.sqrt(np.sum((rec_ess[ok] - eps_ss_t[ok]) ** 2) / np.sum(eps_ss_t[ok] ** 2))


def strain_route_NM(Tprof_c0, Tprof_sl):
    rec_T = Tprof_c0[wb] + (nn - 0.5) * Tprof_sl[wb]
    ethr = A_L * (1 + NU_L) * (rec_T - T_REF)
    sss = lam * (rec_ess + rec_enn - 2 * ethr) + 2 * mu * (rec_ess - ethr)
    p = bin_profiles(sh[ok], nn[ok], dict(sig_ss=sss[ok]))
    Np, Mp, _ = integrate_NMQ(p, tl_g)
    return Np, Mp, p['valid'] & v_t


# truth-T profiles for the oracle-T variant
tT_c0, tT_sl = strain_labels(prof_t, 'T')
N_pT, M_pT, v_pT = strain_route_NM(Theads['T_c0'], Theads['T_slope'])      # fully predicted
N_oT, M_oT, v_oT = strain_route_NM(tT_c0, tT_sl)                            # oracle-T
res = dict(
    eps_ss_field=r_ess_field,
    N_direct=profile_rl2(shell['N'], N_t, v_t),
    M_direct=profile_rl2(shell['M'], M_t, v_t),
    N_strain_predT=profile_rl2(N_pT, N_t, v_pT), M_strain_predT=profile_rl2(M_pT, M_t, v_pT),
    N_strain_oracleT=profile_rl2(N_oT, N_t, v_oT), M_strain_oracleT=profile_rl2(M_oT, M_t, v_oT),
    T_rmse_c0=float(np.sqrt(np.mean((Theads['T_c0'][v_t] - tT_c0[v_t]) ** 2))),
)
print('\n=== GENIESHAN ZERO-SHOT (single real case) ===')
print('eps_ss field recon:      %.2f%%' % res['eps_ss_field'])
print('N direct head:           %.2f%%   (route policy: N <- direct)' % res['N_direct'])
print('M strain route (pred-T): %.2f%%   (route policy: M <- strain route)' % res['M_strain_predT'])
print('M strain route (oracle-T): %.2f%% | N strain (pred-T): %.2f%% | M direct: %.2f%%'
      % (res['M_strain_oracleT'], res['N_strain_predT'], res['M_direct']))
print('T profile RMSE: %.3f C' % res['T_rmse_c0'])

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
ax[0].plot(sc[v_t], N_t[v_t] / 1e3, 'k-', lw=1.6, label='FEM truth')
ax[0].plot(sc[v_t], shell['N'][v_t] / 1e3, 'r--', lw=1.3, label='shell op (direct, %.1f%%)' % res['N_direct'])
ax[0].set_title('GenieShan zero-shot N(s) [kN/m]'); ax[0].legend(); ax[0].set_xlabel('shat')
ax[1].plot(sc[v_t], M_t[v_t] / 1e3, 'k-', lw=1.6, label='FEM truth')
ax[1].plot(sc[v_pT], M_pT[v_pT] / 1e3, 'r--', lw=1.3, label='shell op (pred-T, %.1f%%)' % res['M_strain_predT'])
ax[1].set_title('GenieShan zero-shot M(s) [kNm/m]'); ax[1].legend(); ax[1].set_xlabel('shat')
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/home/jiang/experiment_reports/figs/t3_genie_NM.png', dpi=110)
np.savez('/home/jiang/_t3_plotdata.npz', sc=sc, v_t=v_t, v_pT=v_pT,
         N_truth=N_t, M_truth=M_t, Q_truth=Q_t,
         N_pred=shell['N'], M_pred_strain=M_pT, N_pred_strain=N_pT)
json.dump({k: float(np.round(vv, 3)) if np.isscalar(vv) else vv for k, vv in res.items()},
          open('/home/jiang/_t3_genie_result.json', 'w'), indent=1)
print('T3_GENIE_DONE', flush=True)
