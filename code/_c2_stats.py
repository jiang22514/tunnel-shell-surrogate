#!/usr/bin/env python3
"""C2 statistics package (reviewer R2-4/5/12/13): aggregate everything the
overnight queue produced into one JSON for numbers.tex + tables.
 - main config: per-case arrays -> mean/median/std/max/IQR + bootstrap CI of the
   median + paired Wilcoxon (shell strain-route M vs ff2-integrated M; and
   eps_ss shell vs ff2)
 - 5 alternative splits: median M/N spread
 - ablation rows (k16 / noparams / nopca), 3 seeds each: per-seed eval needed ->
   evaluate each pred npz against profiles (profile-level, cheap)
 - learning curve: 3 seeds -> mean +/- std per N
"""
import glob, json
import numpy as np
from scipy import stats

H = '/home/jiang'
D = '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel/data_E_CT_L_T_v2_L8'
PR = np.load(f'{D}/profiles_NM.npz')
T2 = np.load(f'{H}/_t2_plotdata.npz')
te = T2['te'].astype(int)
out = {}

def summ(a):
    a = np.asarray(a, float)
    q1, q3 = np.percentile(a, [25, 75])
    return dict(med=float(np.median(a)), mean=float(a.mean()), std=float(a.std()),
                max=float(a.max()), iqr=[float(q1), float(q3)])

def boot_med(a, n=20000, seed=0):
    r = np.random.RandomState(seed)
    meds = np.median(r.choice(a, (n, len(a))), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]

# ---- main config per-case stats ----
out['shell_M'] = summ(T2['rl_M']); out['shell_M']['boot95'] = boot_med(T2['rl_M'])
out['shell_N_strain'] = summ(T2['rl_N'])
out['shell_ess'] = summ(T2['r_shell_ess']); out['ff2_ess'] = summ(T2['r_bff_ess'])
bff = json.load(open(f'{H}/_t1_bff_NM_result.json'))
bffM = np.array([c['rlM'] for c in bff['cases']]); bffN = np.array([c['rlN'] for c in bff['cases']])
out['ff2_M'] = summ(bffM); out['ff2_N'] = summ(bffN)
w_M = stats.wilcoxon(T2['rl_M'], bffM)
w_e = stats.wilcoxon(T2['r_shell_ess'], T2['r_bff_ess'])
out['wilcoxon'] = dict(M_p=float(w_M.pvalue), ess_p=float(w_e.pvalue))

# N direct per-case from pred npz ensemble vs truth
files = sorted(glob.glob(f'{H}/_stage3_shell_pred_s?_nobc.npz'))
sp = [np.load(f) for f in files]
Nd = np.mean([p['N'] for p in sp], 0)
rlNd = [100 * np.sqrt(np.sum((Nd[k][PR['valid'][c]] - PR['N'][c][PR['valid'][c]]) ** 2)
        / np.sum(PR['N'][c][PR['valid'][c]] ** 2)) for k, c in enumerate(te)]
out['shell_N_direct'] = summ(rlNd); out['shell_N_direct']['boot95'] = boot_med(np.array(rlNd))

# ---- per-seed (no ensemble) spread of main config ----
def prof_med(pred, field):
    return float(np.median([100 * np.sqrt(np.sum((pred[field][k][PR['valid'][c]] - PR[field][c][PR['valid'][c]]) ** 2)
                / np.sum(PR[field][c][PR['valid'][c]] ** 2)) for k, c in enumerate(te)]))
out['seed_spread'] = {f: [prof_med(p, f) for p in sp] for f in ('eps_m', 'slope_ss', 'N', 'M')}

# ---- splits ----
sp_meds = {}
for spl in range(1, 6):
    f = glob.glob(f'{H}/_stage3_shell_pred_s0_nobc_sp{spl}.npz')
    if not f:
        continue
    p = np.load(f[0])
    tesp = p['te'].astype(int)
    for fld in ('slope_ss', 'N', 'M'):
        errs = [100 * np.sqrt(np.sum((p[fld][k][PR['valid'][c]] - PR[fld][c][PR['valid'][c]]) ** 2)
                / np.sum(PR[fld][c][PR['valid'][c]] ** 2)) for k, c in enumerate(tesp)]
        sp_meds.setdefault(fld, []).append(float(np.median(errs)))
out['splits'] = sp_meds

# ---- ablations (3 seeds each), profile-level medians ----
def abl(tagglob):
    files = sorted(glob.glob(tagglob))
    if not files:
        return None
    r = {}
    for fld in ('eps_m', 'slope_ss', 'N', 'M'):
        vals = []
        for fp in files:
            p = np.load(fp)
            vals.append(float(np.median([100 * np.sqrt(np.sum((p[fld][k][PR['valid'][c]] - PR[fld][c][PR['valid'][c]]) ** 2)
                        / np.sum(PR[fld][c][PR['valid'][c]] ** 2)) for k, c in enumerate(p['te'].astype(int))])))
        r[fld] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), seeds=vals)
    return r
out['abl_k16'] = abl(f'{H}/_stage3_shell_pred_k16s?_nobc.npz')
out['abl_noparams'] = abl(f'{H}/_stage3_shell_pred_nps?_nobc_noparams.npz')
out['abl_nopca'] = abl(f'{H}/_stage3_shell_pred_ncs?_nobc_nopca.npz')
out['abl_main'] = abl(f'{H}/_stage3_shell_pred_s?_nobc.npz')

# ---- learning curve 3 seeds ----
lcs = [json.load(open(f)) for f in sorted(glob.glob(f'{H}/_t4_lc_shell_s?.json'))]
lc = {}
for n in ('20', '40', '60', '80'):
    lc[n] = {f: dict(mean=float(np.mean([l[n][f] for l in lcs])),
                     std=float(np.std([l[n][f] for l in lcs]))) for f in ('eps_m', 'slope_ss', 'N', 'M')}
out['lc'] = lc

json.dump(out, open(f'{H}/_c2_stats.json', 'w'), indent=1)
print('main M: med %.2f boot95 %s | ff2 M med %.2f | wilcoxon M p=%.1e' %
      (out['shell_M']['med'], out['shell_M']['boot95'], out['ff2_M']['med'], out['wilcoxon']['M_p']))
print('N direct: med %.2f boot95 %s' % (out['shell_N_direct']['med'], out['shell_N_direct']['boot95']))
print('splits M medians:', out['splits'].get('M'))
print('seed spread M:', out['seed_spread']['M'])
print('abl M: main %s | k16 %s | noparams %s | nopca %s' % tuple(
    (out[k]['M']['mean'] if out[k] else None) for k in ('abl_main', 'abl_k16', 'abl_noparams', 'abl_nopca')))
print('C2_STATS_DONE')
