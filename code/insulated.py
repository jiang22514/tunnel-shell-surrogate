"""Insulated family: single-track sections with a polyurethane insulation layer between the primary support and
the secondary lining, generated on the monitored design section of a high-geothermal railway tunnel.

Thirteen inputs (columns of `params`): crown radius r1 [m], crown half-angle a1 [deg], wall-to-crown radius ratio
k2, wall arc angle a2 [deg], invert radius r3 [m], lining thickness t [m], extrados departure angle b [deg],
insulation thickness t_pu [m], fill height hf [m], rock temperature T_far [deg C], air temperature T_air [deg C],
heat transfer coefficient h [W/(m2 K)] and casting temperature T_cast [deg C]. Half of the training sections
carry local thickness deviations.

Two operators are released. Variant A sees the design parameters only: station inputs and thickness come from the
design outline. Variant B is the shape-input operator: station inputs and thickness come from the as-built
outline, and the branch also receives eight Fourier coefficients of the thickness deviation. Each operator has a
first stage (five seeds of eight branch-trunk networks; N direct, M recovered from the strain and temperature
profiles with the local thickness) and a second stage (pointwise correction of the recovered moment).
"""
from pathlib import Path
import json

import numpy as np
import torch

from opnn import opnn
from common import FIELDS, KH, NB, Res, fourier, plane_strain_constants, weighted_rel_l2  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'/'insulated'
MODELS = ROOT/'models'/'insulated'
SETS = ['train', 'validation', 'test_nom', 'test_dev', 'test_dev3', 'monitored_design']
E_, NU, AL = 31.5e9, .2, 1e-5
CS, CN, CT = plane_strain_constants(E_, NU, AL)
_EDGES = np.linspace(0, 1, NB + 1)


def load(name):
    """Design inputs, station inputs and reference profiles of one set (see DATA_DICTIONARY.md)."""
    return dict(np.load(DATA/('profiles_%s.npz' % name), allow_pickle=False))


def encode(P):
    """Network inputs from the thirteen parameters (the heat transfer coefficient enters as its logarithm)."""
    P = np.asarray(P, float)
    return np.c_[P[:, :11], np.log(P[:, 11]), P[:, 12:]]


def inputs(d, v):
    X = encode(d['params'])
    return np.c_[X, d['dev_fc']] if v == 'B' else X


def local_thickness(d, v):
    return d['t_st'] if v == 'B' else d['t_nom']


def station_inputs(d, v):
    return d['desc_act'] if v == 'B' else d['desc_nom']


def unwarp(sh, lm, tgt):
    return np.interp(sh, np.r_[0, tgt, 1], np.r_[0, lm, 1])


def station_weights(d):
    """Arc-length share of each station, one row per section."""
    return np.array([np.diff(unwarp(_EDGES, l, d['lm_tgt'])) for l in d['lm']])


class FirstStage:
    def __init__(self, v, directory=None):
        self.v = v
        self.dir = Path(directory) if directory else MODELS/v/'first_stage'
        self.cfg = json.loads((self.dir/'config.json').read_text(encoding='utf-8'))
        self.prep = dict(np.load(self.dir/'preprocessing.npz', allow_pickle=False))
        self.X = torch.as_tensor(fourier(self.prep['sc'], self.cfg['KH']), dtype=torch.float32)
        self.nets = {}
        for seed in self.cfg['seeds']:
            weights = torch.load(self.dir/str(seed)/'best_models.pt', map_location='cpu', weights_only=True)
            self.nets[seed] = {}
            for k in FIELDS:
                net = opnn(self.cfg['branch1'], self.cfg['branch2'], self.cfg['trunk'])
                net.load_state_dict(weights[k])
                net.eval()
                self.nets[seed][k] = net

    def predict(self, d):
        """Seed mean of the eight fields and the recovered moment M_rec."""
        b = torch.as_tensor((inputs(d, self.v) - self.prep['parameter_mean'])/self.prep['parameter_std'], dtype=torch.float32)
        f = torch.ones(len(b), self.cfg['branch1'][0])
        o = {k: 0. for k in FIELDS}
        for seed in self.cfg['seeds']:
            with torch.no_grad():
                for k in FIELDS:
                    o[k] = o[k] + self.nets[seed][k](f, b, self.X).numpy().astype(float)*float(self.prep[k + '_scale']) + float(self.prep[k + '_offset'])
        o = {k: o[k]/len(self.cfg['seeds']) for k in FIELDS}
        t = local_thickness(d, self.v)
        o['M_rec'] = t**2/12*(CS*o['slope_ss'] + CN*o['enn_slope'] + CT*o['T_slope'])
        return o


def features(d, v, prep):
    """(n, 200, nin) inputs of the second stage: standardised branch inputs, Fourier features of the station
    coordinate and the six standardised station inputs."""
    b = (inputs(d, v) - prep['pm'])/prep['ps']
    F = fourier(d['sc'])
    n = len(b)
    D = (station_inputs(d, v) - prep['dmu'])/prep['dsd']
    return np.concatenate([np.repeat(b[:, None, :], len(d['sc']), 1), np.repeat(F[None], n, 0), D], -1).astype(np.float32)


class SecondStage:
    def __init__(self, v, directory=None):
        self.v = v
        self.dir = Path(directory) if directory else MODELS/v/'second_stage'
        self.cfg = json.loads((self.dir/'config.json').read_text(encoding='utf-8'))
        self.prep = dict(np.load(self.dir/'preprocessing.npz', allow_pickle=False))
        self.nets = []
        for seed in self.cfg['seeds']:
            net = Res(self.cfg['n_inputs'], self.cfg['width'], self.cfg['depth'])
            net.load_state_dict(torch.load(self.dir/str(seed)/'best.pt', map_location='cpu', weights_only=True))
            net.eval()
            self.nets.append(net)

    def dM(self, d):
        """Moment correction at every station (seed mean)."""
        Z = torch.as_tensor(features(d, self.v, self.prep), dtype=torch.float32)
        with torch.no_grad():
            return float(self.prep['mscale'])*np.mean([net(Z).numpy().astype(float) for net in self.nets], 0)


class Operator:
    """Both stages of variant 'A' (design parameters) or 'B' (shape input)."""

    def __init__(self, v):
        self.v = v
        self.first = FirstStage(v)
        self.second = SecondStage(v)

    def predict(self, d):
        """N (direct), M (recovered and corrected), M_rec (first stage) and the temperature profile."""
        p = self.first.predict(d)
        return dict(N=p['N'], M=p['M_rec'] + self.second.dM(d), M_rec=p['M_rec'], T_c0=p['T_c0'], T_slope=p['T_slope'])
