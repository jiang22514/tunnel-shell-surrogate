"""Closed-ring family: 164 closed linings in rock under a cold-region thermal load, four geometric inputs.

Inputs of a section: crown radius R_top [m], knee elevation y_knee [m], invert-to-crown radius ratio R_inv/R_top
and lining thickness t [m]. The operator returns, at 200 landmark-aligned stations, the eight profiles of
`common.FIELDS`; the axial force N is taken from its own network and the bending moment M is recovered from the
strain and temperature profiles (`M_recovered`).

    python code/closed_ring.py --params 6.2 -3.1 1.4 0.4 --out prediction.csv
"""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
import torch

from opnn import opnn
from common import FIELDS, fourier, plane_strain_constants

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'/'closed_ring'
MODELS = ROOT/'models'/'closed_ring'
LOWER = np.array([4.5, -4.0, 1.0, 0.3])
UPPER = np.array([8.0, -2.0, 3.0, 0.5])


def load(split):
    """Reference profiles of 'train', 'validation' or 'test'."""
    return dict(np.load(DATA/('profiles_%s.npz' % split), allow_pickle=False))


def rel_l2(pred, truth, valid):
    """Relative L2 error [%] of each profile over its valid stations."""
    valid = np.asarray(valid, bool)
    return np.array([100*np.linalg.norm((p - t)[v])/np.linalg.norm(t[v]) for p, t, v in zip(pred, truth, valid)])


def recover(profiles, thickness, constants, t_ref):
    """N and M from the profile coefficients (Eqs. of the shell reduction); thickness has one value per section."""
    cs, cn, ct = constants
    t = np.asarray(thickness, float)[:, None]
    n = t*(cs*profiles['eps_m'] + cn*profiles['enn_c0'] + ct*(profiles['T_c0'] - t_ref))
    m = t**2/12*(cs*profiles['slope_ss'] + cn*profiles['enn_slope'] + ct*profiles['T_slope'])
    return n, m


class Operator:
    """The released operator: three seeds of eight branch-trunk networks."""

    def __init__(self, directory=MODELS):
        self.dir = Path(directory)
        self.cfg = json.loads((self.dir/'config.json').read_text(encoding='utf-8'))
        self.prep = dict(np.load(self.dir/'preprocessing.npz', allow_pickle=False))
        self.constants = plane_strain_constants(self.cfg['E_Pa'], self.cfg['nu'], self.cfg['alpha_per_K'])
        self.x = torch.as_tensor(fourier(self.prep['sc'], self.cfg['KH']), dtype=torch.float32)
        self.nets = {}
        for seed in self.cfg['seeds']:
            weights = torch.load(self.dir/str(seed)/'best_models.pt', map_location='cpu', weights_only=True)
            self.nets[seed] = {}
            for field in FIELDS:
                net = opnn(self.cfg['branch1'], self.cfg['branch2'], self.cfg['trunk'])
                net.load_state_dict(weights[field], strict=True)
                net.eval()
                self.nets[seed][field] = net

    def predict(self, params, seeds=None, check_range=True):
        """Profiles of the sections `params` (n, 4): seed mean of the eight fields plus N_recovered and M_recovered."""
        params = np.asarray(params, dtype=float)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        if params.ndim != 2 or params.shape[1] != 4 or not np.isfinite(params).all():
            raise ValueError('Provide finite rows [R_top, y_knee, R_inv_ratio, t].')
        if check_range and (np.any(params < LOWER) or np.any(params > UPPER) or np.any(params[:, 2] <= 1)):
            raise ValueError('Range of the family: R_top 4.5 to 8 m, y_knee -4 to -2 m, 1 < ratio <= 3, t 0.3 to 0.5 m.')
        b = torch.as_tensor((params - self.prep['parameter_mean'])/self.prep['parameter_std'], dtype=torch.float32)
        f = torch.ones(len(params), self.cfg['branch1'][0])
        predictions = []
        for seed in (seeds or self.cfg['seeds']):
            p = {}
            for field in FIELDS:
                with torch.no_grad():
                    z = self.nets[seed][field](f, b, self.x).numpy().astype(np.float64)
                p[field] = z*float(self.prep[field + '_scale']) + float(self.prep[field + '_offset'])
            predictions.append(p)
        out = {field: np.mean([p[field] for p in predictions], axis=0) for field in FIELDS}
        out['N_recovered'], out['M_recovered'] = recover(out, params[:, 3], self.constants, self.cfg['T_ref_C'])
        return out

    @property
    def stations(self):
        return self.prep['sc'].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--params', nargs=4, type=float, required=True, metavar=('R_TOP', 'Y_KNEE', 'RATIO', 'THICKNESS'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Choose a new output file.')
    torch.set_num_threads(1)
    op = Operator()
    p = op.predict(args.params)
    with args.out.open('x', encoding='utf-8', newline='') as stream:
        names = ['s_hat', 'N_N_per_m', 'M_Nm_per_m', 'eps_m', 'slope_ss', 'enn_c0', 'enn_slope', 'T_c0_C', 'T_slope_K']
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for j, s in enumerate(op.stations):
            writer.writerow({'s_hat': s, 'N_N_per_m': p['N'][0, j], 'M_Nm_per_m': p['M_recovered'][0, j],
                             'eps_m': p['eps_m'][0, j], 'slope_ss': p['slope_ss'][0, j], 'enn_c0': p['enn_c0'][0, j],
                             'enn_slope': p['enn_slope'][0, j], 'T_c0_C': p['T_c0'][0, j], 'T_slope_K': p['T_slope'][0, j]})
    print('saved %s (%d stations, %d seeds)' % (args.out, len(op.stations), len(op.cfg['seeds'])))


if __name__ == '__main__':
    main()
