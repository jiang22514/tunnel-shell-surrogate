#!/usr/bin/env python3
"""C2: engineering verdict on the new-alpha M errors — compute what design
actually checks: per-station safety factor K (TB 10003-style plain-concrete
eccentric compression, C30), peak signed N/M errors, absolute M errors.
Convention note: dataset N is tension-positive (sigma integral); compression
stations have N<0 -> pass -N to the capacity check."""
import sys, json
import numpy as np
sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel')
from nm_envelope import safety_factors

H = '/home/jiang'
D = '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel/data_E_CT_L_T_v2_L8'
PR = np.load(f'{D}/profiles_NM.npz')
T2 = np.load(f'{H}/_t2_plotdata.npz')
T3 = np.load(f'{H}/_t3_plotdata.npz')
par = np.load(f'{D}/ct_params.npy')

print('=== truth magnitude context (new alpha) ===')
te = T2['te'].astype(int)
Nrms = [np.sqrt(np.mean(PR['N'][c][PR['valid'][c]] ** 2)) for c in te]
Mrms = [np.sqrt(np.mean(PR['M'][c][PR['valid'][c]] ** 2)) for c in te]
print('test fleet: N rms med %.0f kN/m ; M rms med %.2f kNm/m ; |M|max med %.2f kNm/m'
      % (np.median(Nrms) / 1e3, np.median(Mrms) / 1e3,
         np.median([np.abs(PR['M'][c][PR['valid'][c]]).max() for c in te]) / 1e3))
print('genie: N in [%.0f, %.0f] kN/m ; M in [%.2f, %.2f] kNm/m'
      % (T3['N_truth'][T3['v_t']].min() / 1e3, T3['N_truth'][T3['v_t']].max() / 1e3,
         T3['M_truth'][T3['v_t']].min() / 1e3, T3['M_truth'][T3['v_t']].max() / 1e3))

FT = 2.01e6   # C30 characteristic tensile strength f_tk [Pa]; design value 1.39 MPa


def sigt(N, M, h):
    """extreme-fiber tensile stress for tension-positive N [Pa]."""
    return N / h + 6.0 * np.abs(M) / h ** 2


def tension_check(Nt, Mt, Np, Mp, v, h):
    st_t = sigt(Nt[v], Mt[v], h); st_p = sigt(Np[v], Mp[v], h)
    rl = 100 * np.sqrt(np.sum((st_p - st_t) ** 2) / np.sum(st_t ** 2))
    pk_err = 100 * abs(st_p.max() - st_t.max()) / st_t.max()
    crack_t = float(np.mean(st_t > FT)); crack_p = float(np.mean(st_p > FT))
    return dict(sigt_rl2=float(rl), peak_err=float(pk_err),
                sigt_max_t=float(st_t.max() / 1e6), sigt_max_p=float(st_p.max() / 1e6),
                crack_frac_t=crack_t, crack_frac_p=crack_p)


print('\n=== tension-side design check: extreme-fiber tensile stress sigma_t ===')
res = []
for i2, c in enumerate(te):
    h = par[c][3]
    res.append(tension_check(PR['N'][c], PR['M'][c], T2['N_pred'][i2], T2['M_strain'][i2], T2['vv'][i2], h))
med = lambda k: float(np.median([r[k] for r in res]))
mx = lambda k: float(np.max([r[k] for r in res]))
print('test sigma_t profile rel-L2: med %.2f%% max %.2f%%' % (med('sigt_rl2'), mx('sigt_rl2')))
print('test PEAK sigma_t error:     med %.2f%% max %.2f%%' % (med('peak_err'), mx('peak_err')))
print('test cracking fraction (sigma_t > f_tk): truth med %.0f%% / pred med %.0f%%'
      % (100 * med('crack_frac_t'), 100 * med('crack_frac_p')))
crack_frac_err = [100 * abs(r['crack_frac_p'] - r['crack_frac_t']) for r in res]
print('test cracking-zone extent error: med %.1f pp max %.1f pp'
      % (np.median(crack_frac_err), np.max(crack_frac_err)))

vb = T3['v_t'] & T3['v_pT']
rg = tension_check(T3['N_truth'], T3['M_truth'], T3['N_pred'], T3['M_pred_strain'], vb, 0.4)
print('genie sigma_t rel-L2 %.2f%% ; peak err %.2f%% ; sigma_t max %.2f MPa (pred %.2f)'
      % (rg['sigt_rl2'], rg['peak_err'], rg['sigt_max_t'], rg['sigt_max_p']))
print('genie cracking fraction: truth %.0f%% pred %.0f%%' % (100 * rg['crack_frac_t'], 100 * rg['crack_frac_p']))
absM = [np.sqrt(np.mean((T2['M_strain'][i2][T2['vv'][i2]] - PR['M'][c][T2['vv'][i2]]) ** 2)) / 1e3
        for i2, c in enumerate(te)]
print('abs M rmse: test med %.2f kNm/m ; genie %.2f kNm/m'
      % (np.median(absM), np.sqrt(np.mean((T3['M_pred_strain'][vb] - T3['M_truth'][vb]) ** 2)) / 1e3))
json.dump(dict(test_sigt_med=med('sigt_rl2'), test_sigt_max=mx('sigt_rl2'),
               test_peak_med=med('peak_err'), test_peak_max=mx('peak_err'),
               test_crackfrac_err_med=float(np.median(crack_frac_err)),
               test_absM_med=float(np.median(absM)), genie=rg),
          open('/home/jiang/_c2_nm_check.json', 'w'), indent=1)
print('NM_CHECK_DONE')
