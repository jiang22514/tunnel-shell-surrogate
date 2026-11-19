"""Training of the closed-ring operator: eight branch-trunk networks per seed, full-batch Adam.

    python code/train_closed_ring.py --out runs/closed_ring            # the released setting (three seeds)
    python code/train_closed_ring.py --out runs/smoke --seeds 20261022 --max-updates 500

The loss is the sum of the masked mean-squared errors of the eight standardised profiles. Every 250 updates the
recovered moment is evaluated on the 28 validation sections and the weights with the lowest validation error are
kept. Training stops after 20 000 updates or when the validation error has not improved for 6 000 updates (not
before 4 000 updates). The test sections are not read. The released weights were trained on a GPU; a retraining
reproduces the reported errors statistically, not bit for bit.
"""
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch

from opnn import opnn
from common import FIELDS, KH, fourier, plane_strain_constants
import closed_ring as C

SEEDS = [20261022, 20261023, 20261024]
DB1 = [20, 100, 100, 100]
DB2 = [4, 150, 150, 150, 100]
DTK = [2*KH, 100, 100, 100, 100, 100]
MAX_UPDATES, CHECK, MIN_UPDATES, PATIENCE = 20000, 250, 4000, 6000
E, NU, ALPHA, TREF = 32.5e9, .2, 1e-5, 5.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True, help='new directory for the trained models')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--max-updates', type=int, default=MAX_UPDATES)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('The output directory exists; choose a new one.')
    a.out.mkdir(parents=True)
    device = torch.device(a.device)
    cs, cn, ct = plane_strain_constants(E, NU, ALPHA)
    tr, va = C.load('train'), C.load('validation')
    mask, vmask = tr['valid'], va['valid']
    offset = {f: float(tr[f][mask].mean()) for f in FIELDS}
    scale = {f: float(tr[f][mask].std()) for f in FIELDS}
    pm, ps = tr['params'].mean(0), tr['params'].std(0)
    sc = tr['sc'].copy()
    mscale = scale['M']
    np.savez_compressed(a.out/'preprocessing.npz', parameter_mean=pm, parameter_std=ps, sc=sc, lm_tgt=tr['lm_tgt'],
                        **{f + '_offset': offset[f] for f in FIELDS}, **{f + '_scale': scale[f] for f in FIELDS})
    config = dict(family='closed_ring', seeds=a.seeds, fields=FIELDS, branch1=DB1, branch2=DB2, trunk=DTK, KH=KH,
                  max_updates=a.max_updates, check_every=CHECK, minimum_updates=MIN_UPDATES, patience_updates=PATIENCE,
                  n_train=int(len(tr['case'])), n_validation=int(len(va['case'])), E_Pa=E, nu=NU, alpha_per_K=ALPHA, T_ref_C=TREF,
                  loss='sum of the masked mean-squared errors of the eight standardised profiles',
                  optimizer=dict(name='Adam', lr=.001, decay_after_updates=[12000, 16000], decay_factor=.1),
                  selection='lowest validation error of the recovered moment', device=str(device), torch_version=torch.__version__)
    (a.out/'config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')

    def cv(x):
        return torch.as_tensor(x, device=device, dtype=torch.float32)
    x = cv(fourier(sc))
    nt, nv = len(tr['case']), len(va['case'])
    fi, fv = cv(np.ones((nt, DB1[0]))), cv(np.ones((nv, DB1[0])))
    bi, bv = cv((tr['params'] - pm)/ps), cv((va['params'] - pm)/ps)
    wi, wv = cv(mask), cv(vmask)
    tv, vtruth = cv(va['params'][:, 3]), cv(va['M'])
    target = {f: cv(np.where(mask, (tr[f] - offset[f])/scale[f], 0)) for f in FIELDS}
    for seed in a.seeds:
        dest = a.out/str(seed)
        dest.mkdir()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        nets = {f: opnn(DB1, DB2, DTK).to(device).float() for f in FIELDS}
        opt = torch.optim.Adam([p for n in nets.values() for p in n.parameters()], lr=.001)
        best, best_step, history, start = float('inf'), 0, [], time.time()
        for step in range(1, a.max_updates + 1):
            if step - 1 in (12000, 16000):
                for group in opt.param_groups:
                    group['lr'] *= .1
            opt.zero_grad(set_to_none=True)
            z = {f: nets[f](fi, bi, x) for f in FIELDS}
            loss = sum((wi*(z[f] - target[f])**2).sum()/wi.sum() for f in FIELDS)
            loss.backward()
            opt.step()
            if step % CHECK == 0:
                with torch.no_grad():
                    p = {f: nets[f](fv, bv, x)*scale[f] + offset[f] for f in ('slope_ss', 'enn_slope', 'T_slope')}
                    m = tv[:, None]**2/12*(cs*p['slope_ss'] + cn*p['enn_slope'] + ct*p['T_slope'])
                    metric = float((wv*((m - vtruth)/mscale)**2).sum()/wv.sum())
                if metric < best:
                    best, best_step = metric, step
                    torch.save({f: {k: v.detach().cpu().clone() for k, v in n.state_dict().items()} for f, n in nets.items()}, dest/'best_models.pt')
                history.append(dict(step=step, train_loss=float(loss.detach()), validation_M=metric, best_step=best_step))
                if step >= MIN_UPDATES and step - best_step >= PATIENCE:
                    break
        run = dict(seed=seed, updates=step, selected_step=best_step, validation_M=best, seconds=time.time() - start)
        (dest/'history.json').write_text(json.dumps(history), encoding='utf-8')
        (dest/'run.json').write_text(json.dumps(run), encoding='utf-8')
        print('seed %d: %d updates, selected step %d, validation M %.5g, %.0f s' % (seed, step, best_step, best, run['seconds']), flush=True)


if __name__ == '__main__':
    main()
