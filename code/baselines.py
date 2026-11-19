"""Baselines fitted to the training sections of each family (needs scikit-learn).

    python code/baselines.py closed_ring --out runs/baselines_closed_ring.npz
    python code/baselines.py filled --out runs/baselines_filled_test4.npz
    python code/baselines.py insulated --out-dir runs/baselines_insulated

Two models regress the profiles N and M on the standardised inputs: a per-station linear regression with
intercept, and a Gaussian-process regression (Matern 5/2 kernel with one length scale per input and a white-noise
term) of the first 12 principal components. The insulated family uses the inputs of the shape-input operator
(thirteen parameters and eight deviation coefficients). The predictions stored in `predictions/` were produced
with these functions; the Gaussian-process fit of the filled and insulated families takes minutes to hours.
"""
from pathlib import Path
import argparse

import numpy as np


def linear(Xt, Yt, Xe):
    """Per-station least-squares regression with intercept."""
    def H(Z):
        return np.c_[np.ones(len(Z)), Z]
    return H(Xe) @ np.linalg.lstsq(H(Xt), Yt, rcond=None)[0]


def pca_gp(Xt, Yt, Xe_list, k=12):
    """Gaussian processes on the first k principal components of the centred training profiles.
    Xe_list is a list of evaluation inputs; a list of predicted profiles is returned."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    mean = Yt.mean(0)
    U, S, Vt = np.linalg.svd(Yt - mean, full_matrices=False)
    Z = U[:, :k]*S[:k]
    out = [[] for _ in Xe_list]
    for j in range(k):
        s = Z[:, j].std()
        kernel = ConstantKernel(1., (1e-2, 1e2))*Matern(length_scale=np.full(Xt.shape[1], 3.), length_scale_bounds=(.3, 100), nu=2.5) \
            + WhiteKernel(1e-3, (1e-7, 1e-1))
        gp = GaussianProcessRegressor(kernel, random_state=0)
        gp.fit(Xt, Z[:, j]/s)
        for o, Xe in zip(out, Xe_list):
            o.append(gp.predict(Xe)*s)
    return [mean + np.array(o).T @ Vt[:k] for o in out]


def fill_invalid(A, valid):
    """Fill the invalid stations of each training profile by periodic interpolation along the ring."""
    A = A.copy()
    i = np.arange(A.shape[1])
    n = A.shape[1]
    for k in range(len(A)):
        m = valid[k].astype(bool)
        if (~m).any():
            A[k, ~m] = np.interp(i[~m], np.r_[i[m] - n, i[m], i[m] + n], np.tile(A[k, m], 3))
    return A


def closed_ring():
    """Baseline predictions on the 40 test sections."""
    import closed_ring as C
    tr, te = C.load('train'), C.load('test')
    X = tr['params']
    mu, sd = X.mean(0), X.std(0)
    Xt, Xe = (X - mu)/sd, (te['params'] - mu)/sd
    out = dict(case=te['case'])
    for k in ('N', 'M'):
        Y = fill_invalid(tr[k], tr['valid'])
        out[k + '_lin'] = linear(Xt, Y, Xe)
        out[k + '_GP'] = pca_gp(Xt, Y, [Xe])[0]
    return out


def filled(name='test4'):
    """Baseline predictions on one test set of the filled family."""
    import filled as F
    tr, _ = F.load('train')
    te, _ = F.load(name)
    X = F.encode(tr['params'])
    mu, sd = X.mean(0), X.std(0)
    Xt, Xe = (X - mu)/sd, (F.encode(te['params']) - mu)/sd
    out = dict(case=te['case'])
    for k in ('N', 'M'):
        out[k + '_lin'] = linear(Xt, tr[k], Xe)
        out[k + '_GP'] = pca_gp(Xt, tr[k], [Xe])[0]
    return out


def insulated(names=('test_nom', 'test_dev', 'test_dev3', 'monitored_design')):
    """Baseline predictions on the test sets of the insulated family, one dictionary per set."""
    import insulated as I
    tr = I.load('train')
    X = I.inputs(tr, 'B')
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xt = (X - mu)/sd
    sets = [I.load(n) for n in names]
    Xe = [(I.inputs(d, 'B') - mu)/sd for d in sets]
    out = [dict(tag=d['tag']) for d in sets]
    for k in ('N', 'M'):
        for o, x in zip(out, Xe):
            o[k + '_lin'] = linear(Xt, tr[k], x)
        for o, p in zip(out, pca_gp(Xt, tr[k], Xe)):
            o[k + '_GP'] = p
    return dict(zip(names, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('family', choices=['closed_ring', 'filled', 'insulated'])
    ap.add_argument('--out', type=Path, help='output file (closed_ring, filled)')
    ap.add_argument('--out-dir', type=Path, help='output directory (insulated)')
    a = ap.parse_args()
    if a.family == 'insulated':
        if a.out_dir is None or a.out_dir.exists():
            raise SystemExit('Give a new --out-dir.')
        a.out_dir.mkdir(parents=True)
        for name, o in insulated().items():
            np.savez_compressed(a.out_dir/('baselines_%s.npz' % name), **o)
    else:
        if a.out is None or a.out.exists():
            raise SystemExit('Give a new --out file.')
        np.savez_compressed(a.out, **(closed_ring() if a.family == 'closed_ring' else filled()))
    print('done')


if __name__ == '__main__':
    main()
