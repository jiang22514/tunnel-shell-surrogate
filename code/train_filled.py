"""Training of the filled-family operator.

    python code/train_filled.py first --out runs/filled_first
    python code/train_filled.py second --first runs/filled_first --out runs/filled_second
    python code/train_filled.py first --out runs/smoke --seeds 20261022 --max-updates 500

First stage: eight branch-trunk networks per seed, Adam with batches of 64 sections, 24 000 updates, learning
rate 1e-3 reduced tenfold after 6 000 and 10 000 updates; loss = sum of the mean-squared errors of the eight
standardised profiles plus the mean-squared error of the recovered moment over the standard deviation of M.
Second stage: pointwise network for the moment correction dM = M - M_rec of the frozen first stage, same schedule,
station-weighted mean-squared error. Weights are selected on the validation set only. The released weights were
trained on a GPU; a retraining reproduces the reported errors statistically, not bit for bit.
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
from common import FIELDS, KH, Res, fourier, plane_strain_constants, weighted_rel_l2
import filled as F

SEEDS = [20261022, 20261023, 20261024, 20261025, 20261026]
DB1 = [20, 100, 100, 100]
DB2 = [8, 150, 150, 150, 150, 150, 100]
DTK = [2*KH, 150, 150, 150, 150, 150, 150, 100]
MAX, CHECK, BATCH, DROPS = 24000, 250, 64, (6000, 10000)
E, NU, ALPHA = 32.5e9, .2, 1e-5
WIDTH, DEPTH = 256, 4


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def first(a, device):
    cs, cn, ct = plane_strain_constants(E, NU, ALPHA)
    tr, _ = F.load('train')
    va, _ = F.load('validation')
    off = {f: float(tr[f].mean()) for f in FIELDS}
    sca = {f: float(tr[f].std()) for f in FIELDS}
    X = F.encode(tr['params'])
    pm, ps = X.mean(0), X.std(0)
    sc = tr['sc']
    np.savez_compressed(a.out/'preprocessing.npz', parameter_mean=pm, parameter_std=ps, sc=sc, lm_tgt=tr['lm_tgt'],
                        **{f + '_offset': off[f] for f in FIELDS}, **{f + '_scale': sca[f] for f in FIELDS})
    cfg = dict(family='filled', stage='first', moment_loss_weight=1.0, batch=BATCH, lr_drops=list(DROPS),
               inputs=['R_top', 'y_knee', 'R_inv_ratio', 't', 'ln r_k', 'rho', 'phi', 's'], seeds=a.seeds, fields=FIELDS,
               branch1=DB1, branch2=DB2, trunk=DTK, KH=KH, max_updates=a.max_updates, check_every=CHECK,
               E_Pa=E, nu=NU, alpha_per_K=ALPHA, T_ref_C=5., n_train=int(len(tr['case'])), n_validation=int(len(va['case'])),
               selection='lowest validation error of the recovered moment', device=str(device), torch_version=torch.__version__)
    (a.out/'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')

    def cv(q):
        return torch.as_tensor(q, device=device, dtype=torch.float32)
    x = cv(fourier(sc))
    nt, nv = len(tr['case']), len(va['case'])
    fi, fv = cv(np.ones((nt, DB1[0]))), cv(np.ones((nv, DB1[0])))
    bi, bv = cv((X - pm)/ps), cv((F.encode(va['params']) - pm)/ps)
    tv, tru = cv(va['t_st']), cv(va['M'])
    tgt = {f: cv((tr[f] - off[f])/sca[f]) for f in FIELDS}
    mscale = sca['M']
    tt, Mt = cv(tr['t_st']), cv(tr['M'])
    for seed in a.seeds:
        d = a.out/str(seed)
        d.mkdir()
        seed_all(seed, device)
        nets = {f: opnn(DB1, DB2, DTK).to(device).float() for f in FIELDS}
        opt = torch.optim.Adam([p for n in nets.values() for p in n.parameters()], lr=.001)
        best, bs, hist, t0 = float('inf'), 0, [], time.time()
        for step in range(1, a.max_updates + 1):
            if step - 1 in DROPS:
                for g in opt.param_groups:
                    g['lr'] *= .1
            ii = torch.randperm(nt, device=device)[:BATCH]
            opt.zero_grad(set_to_none=True)
            pr = {f: nets[f](fi[ii], bi[ii], x) for f in FIELDS}
            loss = sum(((pr[f] - tgt[f][ii])**2).mean() for f in FIELDS)
            m = tt[ii]**2/12*(cs*(pr['slope_ss']*sca['slope_ss'] + off['slope_ss']) + cn*(pr['enn_slope']*sca['enn_slope'] + off['enn_slope'])
                              + ct*(pr['T_slope']*sca['T_slope'] + off['T_slope']))
            loss = loss + (((m - Mt[ii])/mscale)**2).mean()
            loss.backward()
            opt.step()
            if step % CHECK == 0:
                with torch.no_grad():
                    p = {f: nets[f](fv, bv, x)*sca[f] + off[f] for f in ('slope_ss', 'enn_slope', 'T_slope')}
                    m = tv**2/12*(cs*p['slope_ss'] + cn*p['enn_slope'] + ct*p['T_slope'])
                    met = float((((m - tru)/mscale)**2).mean())
                if met < best:
                    best, bs = met, step
                    torch.save({f: {k: v.detach().cpu().clone() for k, v in n.state_dict().items()} for f, n in nets.items()}, d/'best_models.pt')
                hist.append(dict(step=step, train=float(loss.detach()), val_M=met))
        (d/'history.json').write_text(json.dumps(hist), encoding='utf-8')
        run = dict(seed=seed, updates=step, selected_step=bs, validation_M=best, seconds=time.time() - t0)
        (d/'run.json').write_text(json.dumps(run), encoding='utf-8')
        print('first stage seed %d: selected step %d, validation M %.5g, %.0f s' % (seed, bs, best, run['seconds']), flush=True)


def second(a, device):
    tr, str_ = F.load('train')
    va, sva = F.load('validation')
    D = F.FirstStage(a.first)
    pD_tr = D.predict(tr['params'], str_['t_geo'])
    pD_va = D.predict(va['params'], sva['t_geo'])
    dM_tr, dM_va = tr['M'] - pD_tr['M_rec'], va['M'] - pD_va['M_rec']
    mscale = float(D.prep['M_scale'])
    wtr, wva = F.station_weights(tr['lm']), F.station_weights(va['lm'])
    prep = dict(pm=D.prep['parameter_mean'], ps=D.prep['parameter_std'], dmu=str_['desc'].reshape(-1, 5).mean(0),
                dsd=str_['desc'].reshape(-1, 5).std(0), mscale=mscale)
    np.savez_compressed(a.out/'preprocessing.npz', **prep)
    Ztr = F.features(tr['params'], str_['desc'], prep)
    Zva = F.features(va['params'], sva['desc'], prep)
    nin = Ztr.shape[-1]

    def cv(q):
        return torch.as_tensor(np.asarray(q), device=device, dtype=torch.float32)
    Zt, Zv = cv(Ztr), cv(Zva)
    Yt, Yv = cv(dM_tr/mscale), cv(dM_va/mscale)
    Wt, Wv = cv(wtr/wtr.mean(1, keepdims=True)), cv(wva/wva.mean(1, keepdims=True))
    cfg = dict(family='filled', stage='second', use_desc=True, n_inputs=int(nin), width=WIDTH, depth=DEPTH, activation='SiLU',
               output_init='zero', seeds=a.seeds, max_updates=a.max_updates, lr_drops=list(DROPS), batch=BATCH, check_every=CHECK,
               loss='mean of w_normalised (dM_pred - dM)^2 / M_scale^2', selection='lowest validation loss',
               n_train=int(len(tr['case'])), n_validation=int(len(va['case'])), device=str(device), torch_version=torch.__version__)
    (a.out/'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    nt = len(tr['case'])
    for seed in a.seeds:
        d = a.out/str(seed)
        d.mkdir()
        seed_all(seed, device)
        net = Res(nin, WIDTH, DEPTH).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=.001)
        best, bs, bst, t0 = float('inf'), 0, None, time.time()
        for step in range(1, a.max_updates + 1):
            if step - 1 in DROPS:
                for g in opt.param_groups:
                    g['lr'] *= .1
            ii = torch.randperm(nt, device=device)[:BATCH]
            opt.zero_grad(set_to_none=True)
            L = (Wt[ii]*(net(Zt[ii]) - Yt[ii])**2).mean()
            L.backward()
            opt.step()
            if step % CHECK == 0:
                with torch.no_grad():
                    lv = float((Wv*(net(Zv) - Yv)**2).mean())
                if lv < best:
                    best, bs, bst = lv, step, copy.deepcopy(net.state_dict())
        torch.save({k: v.cpu() for k, v in bst.items()}, d/'best.pt')
        net.load_state_dict(bst)
        with torch.no_grad():
            dm = net(Zv).cpu().numpy().astype(float)*mscale
        e = weighted_rel_l2(pD_va['M_rec'] + dm, va['M'], wva)
        run = dict(seed=seed, selected_step=bs, validation_loss=best, validation_M_median=float(np.median(e)), seconds=time.time() - t0)
        (d/'run.json').write_text(json.dumps(run), encoding='utf-8')
        print('second stage seed %d: selected step %d, validation M median %.3f %%, %.0f s' % (seed, bs, np.median(e), run['seconds']), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['first', 'second'])
    ap.add_argument('--out', type=Path, required=True, help='new directory for the trained models')
    ap.add_argument('--first', type=Path, default=F.MODELS/'first_stage', help='first stage used by the second stage')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--max-updates', type=int, default=MAX)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('The output directory exists; choose a new one.')
    a.out.mkdir(parents=True)
    device = torch.device(a.device)
    (first if a.stage == 'first' else second)(a, device)


if __name__ == '__main__':
    main()
