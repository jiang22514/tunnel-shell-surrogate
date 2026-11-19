"""Filled family: double-track sections with an invert fill and floor voids, eight inputs, cold-region thermal load.

Inputs of a section (columns of `params`): R_top [m], y_knee [m], R_inv/R_top, arch thickness t [m], knee radius
r_k [m], invert-to-arch thickness ratio rho, relative junction height phi, floor scale s. The operator has two
stages. The first stage is the profile operator (five seeds of eight branch-trunk networks; N from its own
network, M recovered from the strain and temperature profiles with the local thickness). The second stage is a
pointwise network that corrects the recovered moment from the parameters, the station coordinate and five
geometric station inputs (file `stations_<set>.npz`, array `desc`).
"""
from pathlib import Path
import json

import numpy as np
import torch

from opnn import opnn
from common import FIELDS, KH, NB, Res, fourier, plane_strain_constants, weighted_rel_l2  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'/'filled'
MODELS = ROOT/'models'/'filled'
SETS = ['train', 'validation', 'test1', 'test2', 'test3', 'test4']
LM_TGT = np.array([.12, .24, .32, .62, .92])   # aligned positions of the five interior landmarks
SC = (np.arange(NB) + .5)/NB                   # aligned station coordinate
_EDGES = np.linspace(0, 1, NB + 1)


def load(name):
    """Reference profiles and geometric station inputs of one set."""
    profiles = dict(np.load(DATA/('profiles_%s.npz' % name), allow_pickle=False))
    stations = dict(np.load(DATA/('stations_%s.npz' % name), allow_pickle=False))
    assert (profiles['case'] == stations['case']).all()
    return profiles, stations


def unwarp(sh, lm):
    """Aligned coordinate to the arc-length fraction of a section with landmarks lm."""
    return np.interp(sh, np.r_[0, LM_TGT, 1], np.r_[0, lm, 1])


def station_weights(lm):
    """Arc-length share of each station, one row per section."""
    return np.array([np.diff(unwarp(_EDGES, l)) for l in lm])


def encode(P):
    """Network inputs from the eight parameters (the knee radius enters as its logarithm)."""
    P = np.atleast_2d(np.asarray(P, float))
    return np.c_[P[:, :4], np.log(P[:, 4]), P[:, 5:8]]


class FirstStage:
    def __init__(self, directory=MODELS/'first_stage'):
        self.dir = Path(directory)
        self.cfg = json.loads((self.dir/'config.json').read_text(encoding='utf-8'))
        self.prep = dict(np.load(self.dir/'preprocessing.npz', allow_pickle=False))
        self.C = plane_strain_constants(self.cfg['E_Pa'], self.cfg['nu'], self.cfg['alpha_per_K'])
        self.X = torch.as_tensor(fourier(SC, self.cfg['KH']), dtype=torch.float32)
        self.nets = {}
        for seed in self.cfg['seeds']:
            weights = torch.load(self.dir/str(seed)/'best_models.pt', map_location='cpu', weights_only=True)
            self.nets[seed] = {}
            for k in FIELDS:
                net = opnn(self.cfg['branch1'], self.cfg['branch2'], self.cfg['trunk'])
                net.load_state_dict(weights[k])
                net.eval()
                self.nets[seed][k] = net

    def raw(self, P, seed):
        P = np.atleast_2d(np.asarray(P, float))
        b = torch.as_tensor((encode(P) - self.prep['parameter_mean'])/self.prep['parameter_std'], dtype=torch.float32)
        f = torch.ones(len(P), self.cfg['branch1'][0])
        out = {}
        for k in FIELDS:
            with torch.no_grad():
                out[k] = self.nets[seed][k](f, b, self.X).numpy().astype(float)*float(self.prep[k + '_scale']) + float(self.prep[k + '_offset'])
        return out

    def predict(self, P, t, seeds=None):
        """Seed mean of the eight fields and the recovered moment M_rec; t is the (n, 200) local thickness."""
        R = [self.raw(P, seed) for seed in (seeds or self.cfg['seeds'])]
        out = {k: np.mean([r[k] for r in R], 0) for k in FIELDS}
        cs, cn, ct = self.C
        out['M_rec'] = t**2/12*(cs*out['slope_ss'] + cn*out['enn_slope'] + ct*out['T_slope'])
        return out


def features(P, desc, prep, use_desc=True):
    """(n, 200, nin) inputs of the second stage: standardised parameters, Fourier features of the station
    coordinate and the standardised geometric station inputs."""
    n = len(P)
    b = (encode(P) - prep['pm'])/prep['ps']
    Z = [np.repeat(b[:, None, :], NB, 1), np.repeat(fourier(SC)[None], n, 0)]
    if use_desc:
        Z.append((desc - prep['dmu'])/prep['dsd'])
    return np.concatenate(Z, -1).astype(np.float32)


class SecondStage:
    def __init__(self, directory=MODELS/'second_stage'):
        self.dir = Path(directory)
        self.cfg = json.loads((self.dir/'config.json').read_text(encoding='utf-8'))
        self.prep = dict(np.load(self.dir/'preprocessing.npz', allow_pickle=False))
        self.nets = {}
        for seed in self.cfg['seeds']:
            net = Res(self.cfg['n_inputs'], self.cfg['width'], self.cfg['depth'])
            net.load_state_dict(torch.load(self.dir/str(seed)/'best.pt', map_location='cpu', weights_only=True))
            net.eval()
            self.nets[seed] = net

    def dM(self, P, desc, seeds=None):
        """Moment correction at every station (seed mean)."""
        Z = torch.as_tensor(features(P, desc, self.prep, self.cfg['use_desc']))
        with torch.no_grad():
            return float(self.prep['mscale'])*np.mean([self.nets[seed](Z).numpy().astype(float) for seed in (seeds or self.cfg['seeds'])], 0)


class Operator:
    """Both stages. `predict` returns the fields of the first stage, `M_rec` (first stage) and `M` (corrected);
    the correction is carried by the strain slope, so the returned profiles stay consistent with `M`."""

    def __init__(self, first=None, second=None):
        self.first = first or FirstStage()
        self.second = second or SecondStage()

    def predict(self, P, stations, second_stage=True, seeds=None):
        t = stations['t_geo']
        p = self.first.predict(P, t, seeds)
        p['M'] = p['M_rec']
        if second_stage:
            dm = self.second.dM(P, stations['desc'], seeds)
            p['M'] = p['M_rec'] + dm
            p['slope_ss'] = p['slope_ss'] + 12*dm/(self.first.C[0]*t**2)
        p['t_st'] = t
        return p
