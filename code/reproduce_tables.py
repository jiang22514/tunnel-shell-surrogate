"""Recompute the headline errors of the three section families from the released weights and data.

    python code/reproduce_tables.py --device cpu
    python code/reproduce_tables.py --family closed_ring --json results.json

The operators are evaluated on the CPU from the files in `models/` and `data/`. The baselines (per-station linear
regression and Gaussian process on principal components) are read from `predictions/`; they can be refitted with
`code/baselines.py`. Every recomputed number is compared with `data/expected_numbers.json`, which holds the values
printed in the manuscript. A number passes if it agrees with the printed value to the printed digits. The script
exits with status 1 if any number fails or is missing.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

from common import FTK_C40, fibre, weighted_rel_l2

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT/'predictions'
NOTES = []


def note(text):
    NOTES.append(text)


def closed_ring_numbers():
    import closed_ring as C
    te = C.load('test')
    v = te['valid'].astype(bool)
    n = len(te['case'])
    op = C.Operator()
    pr = op.predict(te['params'])
    eN = C.rel_l2(pr['N'], te['N'], v)
    eM = C.rel_l2(pr['M_recovered'], te['M'], v)
    eMd = C.rel_l2(pr['M'], te['M'], v)
    first = te['case'] < 1000                       # sections of the first sample; the added ones are numbered from 1000
    seedM = [float(np.median(C.rel_l2(op.predict(te['params'], seeds=[s])['M_recovered'], te['M'], v))) for s in op.cfg['seeds']]
    out = dict(GenN=np.median(eN), GenNmax=eN.max(), GenM=np.median(eM), GenMpn=np.quantile(eM, .9), GenMmax=eM.max(),
               GenMret=np.median(eM[first]), GenMretMax=eM[first].max(), GenMadd=np.median(eM[~first]), GenMaddMax=eM[~first].max(),
               GenNret=np.median(eN[first]), GenNadd=np.median(eN[~first]), GenMdir=np.median(eMd),
               GenSeedMlo=min(seedM), GenSeedMhi=max(seedM), GenNtest=n)
    # recovery applied to the reference profiles
    _, Mfl = C.recover(te, te['params'][:, 3], op.constants, op.cfg['T_ref_C'])
    eFl = C.rel_l2(Mfl, te['M'], v)
    out.update(GenFloorM=np.median(eFl), GenFloorMmax=eFl.max())
    # extreme-fibre tension from the resultants
    t = te['params'][:, 3][:, None]

    def sig(N_, M_):
        return N_/t + 6*np.abs(M_)/t**2
    sF, sP = sig(te['N'], te['M']), sig(pr['N'], pr['M_recovered'])
    eS = C.rel_l2(sP, sF, v)
    pk = np.array([100*abs(sP[i][v[i]].max()/sF[i][v[i]].max() - 1) for i in range(n)])
    ck = np.array([100*abs((sP[i][v[i]] > FTK_C40).mean() - (sF[i][v[i]] > FTK_C40).mean()) for i in range(n)])
    out.update(GenSig=np.median(eS), GenSigMax=eS.max(), GenSigPeak=np.median(pk), GenSigPeakMax=pk.max(), GenCrackPP=ck.max())
    # baselines
    bl = np.load(PRED/'closed_ring'/'baselines_test.npz', allow_pickle=False)
    assert (bl['case'] == te['case']).all()
    for key, name in (('lin', 'Lin'), ('GP', 'Gp')):
        bN = C.rel_l2(bl['N_' + key], te['N'], v)
        bM = C.rel_l2(bl['M_' + key], te['M'], v)
        out.update({'Gen%sN' % name: np.median(bN), 'Gen%sNmax' % name: bN.max(), 'Gen%sM' % name: np.median(bM), 'Gen%sMmax' % name: bM.max()})
        if key == 'GP':
            out['GenBetter'] = int((eM < bM).sum())
    stored = np.load(PRED/'closed_ring'/'test_ensemble.npz', allow_pickle=False)
    diff = max(float(np.abs(pr[f] - stored[f]).max()/float(op.prep[f + '_scale'])) for f in ('N', 'M', 'slope_ss', 'T_slope'))
    note('closed ring: recomputed and stored ensemble predictions differ by at most %.1e of the field scale' % diff)
    return out


def filled_numbers():
    import filled as F
    op = F.Operator()
    out = {}

    def peak_fibre(N, M, t):
        return np.maximum(*fibre(N, M, t)).max(axis=-1)/1e6
    for name, prefix in (('test4', 'FlTest'), ('test3', 'FlTestthree'), ('test2', 'FlTesttwo'), ('test1', 'FlTestone')):
        d, st = F.load(name)
        w = F.station_weights(d['lm'])
        p = op.predict(d['params'], st)
        eN = weighted_rel_l2(p['N'], d['N'], w)
        eM = weighted_rel_l2(p['M'], d['M'], w)
        t = d['params'][:, 3]
        out.update({prefix + 'N': np.median(eN), prefix + 'M': np.median(eM), prefix + 'Mpn': np.quantile(eM, .9), prefix + 'Mmax': eM.max()})
        if name != 'test4':
            out[prefix + 'ThinM'] = np.median(eM[t < .45])
            continue
        pk = 100*abs(abs(p['M']).max(1)/abs(d['M']).max(1) - 1)
        fb = 100*abs(peak_fibre(p['N'], p['M'], d['t_st'])/peak_fibre(d['N'], d['M'], d['t_st']) - 1)
        out.update(FlTestNmax=eN.max(), FlTestPeak=np.median(pk), FlTestFibre=np.median(fb), FlThinM=np.median(eM[t < .45]),
                   FlMidM=np.median(eM[(t >= .45) & (t < .6)]), FlThickM=np.median(eM[t >= .6]), FlNthin=int((t < .45).sum()))
        p1 = op.predict(d['params'], st, second_stage=False)
        e1 = weighted_rel_l2(p1['M'], d['M'], w)
        e1N = weighted_rel_l2(p1['N'], d['N'], w)
        out.update(FlDN=np.median(e1N), FlDM=np.median(e1), FlDMpn=np.quantile(e1, .9), FlDMmax=e1.max(), FlDThinM=np.median(e1[t < .45]),
                   FlBetter=int((eM < e1).sum()), FlBetterThin=int(((eM < e1) & (t < .45)).sum()), FlPairDiff=np.median(e1 - eM))
        f = PRED/'filled'/'baselines_test4.npz'
        if f.exists():
            bl = np.load(f, allow_pickle=False)
            assert (bl['case'] == d['case']).all()
            for key, nm in (('GP', 'Gp'), ('lin', 'Lin')):
                bN = weighted_rel_l2(bl['N_' + key], d['N'], w)
                bM = weighted_rel_l2(bl['M_' + key], d['M'], w)
                out.update({'Fl%sN' % nm: np.median(bN), 'Fl%sM' % nm: np.median(bM), 'Fl%sMpn' % nm: np.quantile(bM, .9),
                            'Fl%sMmax' % nm: bM.max(), 'Fl%sThinM' % nm: np.median(bM[t < .45])})
        else:
            stored = json.loads((PRED/'filled'/'test_errors.json').read_text(encoding='utf-8'))['baselines']
            for key, nm in (('PCA+GP', 'Gp'), ('linear', 'Lin')):
                bM = np.array(stored[key]['test4_M_per_case'])
                out.update({'Fl%sN' % nm: stored[key]['test4_N']['median'], 'Fl%sM' % nm: np.median(bM), 'Fl%sMpn' % nm: np.quantile(bM, .9),
                            'Fl%sMmax' % nm: bM.max(), 'Fl%sThinM' % nm: np.median(bM[t < .45])})
            note('filled family: baseline numbers are taken from the stored per-section errors (predictions/filled/test_errors.json)')
    return out


def insulated_numbers():
    import insulated as I
    ops = {v: I.Operator(v) for v in 'AB'}
    out = {}
    worst = 0.
    for name, s in (('test_nom', 'Nom'), ('test_dev', 'Dev'), ('test_dev3', 'Three'), ('monitored_design', 'Mon')):
        d = I.load(name)
        w = I.station_weights(d)
        stored = np.load(PRED/'insulated'/('pred_%s.npz' % name), allow_pickle=False)
        assert (stored['tag'] == d['tag']).all()
        P = {}
        for v in 'AB':
            p = ops[v].predict(d)
            P[v] = (p['N'], p['M'])
            worst = max(worst, float(np.abs(p['M'] - stored['M_' + v]).max()/np.abs(d['M']).max()), float(np.abs(p['N'] - stored['N_' + v]).max()/np.abs(d['N']).max()))
        P['GP'] = (stored['N_GP'], stored['M_GP'])
        P['lin'] = (stored['N_lin'], stored['M_lin'])
        so, si = fibre(d['N'], d['M'], d['t_st'])
        smax = np.maximum(so, si).max(1)
        eMs, peaks = {}, {}
        for m, mm in (('A', 'A'), ('B', 'B'), ('GP', 'Gp'), ('lin', 'Lin')):
            N_, M_ = P[m]
            eN = weighted_rel_l2(N_, d['N'], w)
            eM = weighted_rel_l2(M_, d['M'], w)
            t = d['t_nom'] if m == 'A' else d['t_st']
            po, pi = fibre(N_, M_, t)
            pmax = np.maximum(po, pi).max(1)
            key = 'Ins%s%s' % (s, mm)
            out.update({key + 'Nmed': np.median(eN), key + 'Nmean': eN.mean(), key + 'Mmed': np.median(eM), key + 'Mmean': eM.mean(),
                        key + 'Mpn': np.percentile(eM, 90), key + 'Mmax': eM.max(),
                        key + 'Peak': np.mean(np.abs(pmax - smax)/np.abs(smax))*100, key + 'GtTen': int(np.sum(eM > 10))})
            eMs[m], peaks[m] = eM, pmax
        out['Ins%sPeakFE' % s] = smax.mean()/1e6
        if len(d['tag']) > 5:
            out['Ins%sBetter' % s] = int(np.sum(eMs['B'] < eMs['A']))
            out['Ins%sN' % s] = int(len(d['tag']))
        else:
            out.update(InsMonPeakMPaFE=smax[0]/1e6, InsMonPeakMPaA=peaks['A'][0]/1e6, InsMonPeakMPaB=peaks['B'][0]/1e6)
    note('insulated family: recomputed and stored operator predictions differ by at most %.1e of the largest reference value' % worst)
    return out


FAMILIES = dict(closed_ring=closed_ring_numbers, filled=filled_numbers, insulated=insulated_numbers)


def compare(expected, computed):
    """Rows (name, printed, recomputed, status). A number passes if it rounds to the printed value."""
    rows, failed = [], 0
    for name, printed in expected.items():
        if name not in computed:
            rows.append((name, printed, 'not recomputed', 'FAIL'))
            failed += 1
            continue
        value = computed[name]
        digits = len(printed.split('.')[1]) if '.' in printed else 0
        shown = '%.*f' % (digits, value)
        if isinstance(value, (int, np.integer)):
            ok = int(value) == int(printed)
        else:
            ok = abs(float(value) - float(printed)) <= 0.5*10.0**(-digits)*(1 + 1e-3) + 1e-12
        status = 'ok' if shown == printed else ('ok (rounding)' if ok else 'FAIL')
        failed += not ok
        rows.append((name, printed, shown, status))
    return rows, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', choices=['cpu'], default='cpu', help='the evaluation runs on the CPU')
    ap.add_argument('--family', choices=list(FAMILIES) + ['all'], default='all')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--json', type=Path, help='write the recomputed numbers to this file')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    expected = json.loads((ROOT/'data'/'expected_numbers.json').read_text(encoding='utf-8'))
    families = list(FAMILIES) if a.family == 'all' else [a.family]
    total_failed, results = 0, {}
    for fam in families:
        computed = FAMILIES[fam]()
        rows, failed = compare(expected[fam], computed)
        total_failed += failed
        results[fam] = {r[0]: dict(printed=r[1], recomputed=r[2], status=r[3]) for r in rows}
        print('\n%s: %d numbers, %d failed' % (fam, len(rows), failed))
        print('  %-22s %10s %12s  %s' % ('name', 'printed', 'recomputed', 'status'))
        for r in rows:
            print('  %-22s %10s %12s  %s' % r)
    print()
    for text in NOTES:
        print('note: ' + text)
    if a.json:
        a.json.write_text(json.dumps(results, indent=1), encoding='utf-8')
    if total_failed:
        print('FAILED: %d numbers do not agree with the printed values' % total_failed)
        sys.exit(1)
    print('All recomputed numbers agree with the printed values.')


if __name__ == '__main__':
    main()
