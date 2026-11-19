"""Training of the insulated-family operators (variant A: design parameters; variant B: shape input).

    python code/train_insulated.py first B --out runs/insulated_B_first
    python code/train_insulated.py second B --first runs/insulated_B_first --out runs/insulated_B_second
    python code/train_insulated.py first B --out runs/smoke --seeds 20261022 --max-updates 500

First stage: eight branch-trunk networks per seed, Adam with batches of 64 sections, 24 000 updates, learning
rate 1e-3 reduced tenfold after 6 000 and 10 000 updates; loss = sum of the mean-squared errors of the eight
standardised profiles plus the mean-squared error of the recovered moment over the standard deviation of M.
Second stage: pointwise network for the moment correction dM = M - M_rec of the frozen first stage, same
schedule, station-weighted mean-squared error. Weights are selected on the 150 validation sections only. The
released weights were trained on CPU threads; a retraining reproduces the reported errors statistically.
"""
from pathlib import Path
import argparse
import copy
import json
import random
import time

import numpy as np
import torch

from opnn import opnn
from common import FIELDS, KH, Res, fourier, weighted_rel_l2
import insulated as I

SEEDS = [20261022, 20261023, 20261024, 20261025, 20261026]
DB1 = [20, 100, 100, 100]
DTK = [2*KH, 150, 150, 150, 150, 150, 150, 100]
MAX, CHECK, BATCH, DROPS = 24000, 250, 64, (6000, 10000)
DEV = torch.device('cpu')


def cv(q):
    return torch.as_tensor(np.asarray(q), device=DEV, dtype=torch.float32)


def first(a):
    v = a.variant
    tr, va = I.load('train'), I.load('validation')
    X = I.inputs(tr, v)
    pm, ps = X.mean(0), X.std(0) + 1e-12
    DB2 = [X.shape[1], 150, 150, 150, 150, 150, 100]
    off = {f: float(tr[f].mean()) for f in FIELDS}
    sca = {f: float(tr[f].std()) for f in FIELDS}
    sc = tr['sc']
    np.savez_compressed(a.out/'preprocessing.npz', parameter_mean=pm, parameter_std=ps, sc=sc,
                        **{f + '_offset': off[f] for f in FIELDS}, **{f + '_scale': sca[f] for f in FIELDS})
    cfg = dict(family='insulated', stage='first', variant=v, n_inputs=int(X.shape[1]), branch1=DB1, branch2=DB2, trunk=DTK, KH=KH,
               seeds=a.seeds, fields=FIELDS, max_updates=a.max_updates, check_every=CHECK, batch=BATCH, lr_drops=list(DROPS),
               E_Pa=I.E_, nu=I.NU, alpha_per_K=I.AL, moment_loss_weight=1.0, selection='lowest validation recovered-M MSE',
               n_train=int(len(tr['tag'])), n_validation=int(len(va['tag'])), device='cpu', threads=torch.get_num_threads(), torch=torch.__version__)
    (a.out/'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    x = cv(fourier(sc))
    nt, nv = len(tr['tag']), len(va['tag'])
    fi, fv = cv(np.ones((nt, DB1[0]))), cv(np.ones((nv, DB1[0])))
    bi, bv = cv((X - pm)/ps), cv((I.inputs(va, v) - pm)/ps)
    tgt = {f: cv((tr[f] - off[f])/sca[f]) for f in FIELDS}
    tt, Mt = cv(I.local_thickness(tr, v)), cv(tr['M'])
    tv, Mv = cv(I.local_thickness(va, v)), cv(va['M'])
    ms = sca['M']
    for seed in a.seeds:
        d = a.out/str(seed)
        d.mkdir()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        nets = {f: opnn(DB1, DB2, DTK).to(DEV).float() for f in FIELDS}
        opt = torch.optim.Adam([p for n in nets.values() for p in n.parameters()], lr=.001)
        best, bs, t0 = float('inf'), 0, time.time()
        for step in range(1, a.max_updates + 1):
            if step - 1 in DROPS:
                for g in opt.param_groups:
                    g['lr'] *= .1
            ii = torch.randperm(nt)[:BATCH]
            opt.zero_grad(set_to_none=True)
            pr = {f: nets[f](fi[ii], bi[ii], x) for f in FIELDS}
            loss = sum(((pr[f] - tgt[f][ii])**2).mean() for f in FIELDS)
            m = tt[ii]**2/12*(I.CS*(pr['slope_ss']*sca['slope_ss'] + off['slope_ss']) + I.CN*(pr['enn_slope']*sca['enn_slope'] + off['enn_slope'])
                              + I.CT*(pr['T_slope']*sca['T_slope'] + off['T_slope']))
            loss = loss + (((m - Mt[ii])/ms)**2).mean()
            loss.backward()
            opt.step()
            if step % CHECK == 0:
                with torch.no_grad():
                    p = {f: nets[f](fv, bv, x)*sca[f] + off[f] for f in ('slope_ss', 'enn_slope', 'T_slope')}
                    m = tv**2/12*(I.CS*p['slope_ss'] + I.CN*p['enn_slope'] + I.CT*p['T_slope'])
                    met = float((((m - Mv)/ms)**2).mean())
                if met < best:
                    best, bs = met, step
                    torch.save({f: {k: w.detach().clone() for k, w in n.state_dict().items()} for f, n in nets.items()}, d/'best_models.pt')
        run = dict(seed=seed, selected_step=bs, validation_M=best, seconds=time.time() - t0)
        (d/'run.json').write_text(json.dumps(run), encoding='utf-8')
        print('first stage %s seed %d: selected step %d, validation M %.5g, %.0f s' % (v, seed, bs, best, run['seconds']), flush=True)


def second(a, width=256, depth=4):
    v = a.variant
    tr, va = I.load('train'), I.load('validation')
    D = I.FirstStage(v, a.first)
    pt, pv = D.predict(tr), D.predict(va)
    dMt, dMv = tr['M'] - pt['M_rec'], va['M'] - pv['M_rec']
    X = I.inputs(tr, v)
    prep = dict(pm=X.mean(0), ps=X.std(0) + 1e-12, dmu=I.station_inputs(tr, v).reshape(-1, 6).mean(0),
                dsd=I.station_inputs(tr, v).reshape(-1, 6).std(0) + 1e-12, mscale=float(D.prep['M_scale']))
    np.savez_compressed(a.out/'preprocessing.npz', **prep)
    ms = prep['mscale']
    Zt, Zv = cv(I.features(tr, v, prep)), cv(I.features(va, v, prep))
    Yt, Yv = cv(dMt/ms), cv(dMv/ms)
    wt, wv = I.station_weights(tr), I.station_weights(va)
    Wt, Wv = cv(wt/wt.mean(1, keepdims=True)), cv(wv/wv.mean(1, keepdims=True))
    cfg = dict(family='insulated', stage='second', variant=v, n_inputs=int(Zt.shape[-1]), width=width, depth=depth, seeds=a.seeds,
               max_updates=a.max_updates, lr_drops=list(DROPS), batch=BATCH, loss='weighted MSE of dM/M_scale', device='cpu')
    (a.out/'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    nt = len(tr['tag'])
    for seed in a.seeds:
        d = a.out/str(seed)
        d.mkdir()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = Res(Zt.shape[-1], width, depth)
        opt = torch.optim.Adam(net.parameters(), lr=.001)
        best, bs, bst, t0 = float('inf'), 0, None, time.time()
        for step in range(1, a.max_updates + 1):
            if step - 1 in DROPS:
                for g in opt.param_groups:
                    g['lr'] *= .1
            ii = torch.randperm(nt)[:BATCH]
            opt.zero_grad(set_to_none=True)
            L = (Wt[ii]*(net(Zt[ii]) - Yt[ii])**2).mean()
            L.backward()
            opt.step()
            if step % CHECK == 0:
                with torch.no_grad():
                    lv = float((Wv*(net(Zv) - Yv)**2).mean())
                if lv < best:
                    best, bs, bst = lv, step, copy.deepcopy(net.state_dict())
        torch.save(bst, d/'best.pt')
        net.load_state_dict(bst)
        with torch.no_grad():
            dm = net(Zv).numpy().astype(float)*ms
        e = weighted_rel_l2(pv['M_rec'] + dm, va['M'], wv)
        run = dict(seed=seed, selected_step=bs, validation_M_median=float(np.median(e)), seconds=time.time() - t0)
        (d/'run.json').write_text(json.dumps(run), encoding='utf-8')
        print('second stage %s seed %d: selected step %d, validation M median %.3f %%, %.0f s' % (v, seed, bs, np.median(e), run['seconds']), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['first', 'second'])
    ap.add_argument('variant', choices=['A', 'B'])
    ap.add_argument('--out', type=Path, required=True, help='new directory for the trained models')
    ap.add_argument('--first', type=Path, default=None, help='first stage used by the second stage (default: the released one)')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--max-updates', type=int, default=MAX)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('The output directory exists; choose a new one.')
    a.out.mkdir(parents=True)
    torch.set_num_threads(a.threads)
    (first if a.stage == 'first' else second)(a)


if __name__ == '__main__':
    main()
