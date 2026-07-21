#!/usr/bin/env python3
"""FAIR 2D baseline (reviewer R2-5): direct strain-field regression with Fourier
features, given IDENTICAL inputs to the shell operator (branch1 = shape PCA20,
branch2 = raw section params; no solution-derived inputs) and the same 3-seed
ensembling. argv: [EPOCHS] [TAG]"""
import sys, json, time
import numpy as np, torch
from sklearn.decomposition import PCA
sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/DIMON')
from opnn import opnn

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
TAG = sys.argv[2] if len(sys.argv) > 2 else ''
if TAG:
    torch.manual_seed(abs(hash(TAG)) % 100000 + 7)
dev = torch.device('cuda')
D = '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel/data_E_CT_L_T_v2_L8'
d1 = np.load(f'{D}/u1_dataset.npz'); d2 = np.load(f'{D}/u2_dataset.npz')
PAR4 = np.load(f'{D}/ct_params.npy')[:, :4]
x_uni = d1['x_uni']; x_mesh = d1['all_tunnel_def_grids']
lin = d1['lining_masks']
e11 = d1['e11_data']; e22 = d1['e22_data']; e12 = d1['e12_data']
Ncase, P = e11.shape
rng = np.random.RandomState(0); perm = rng.permutation(Ncase)
te = perm[:20]; tr = perm[20:]
ffrng = np.random.RandomState(7)
Bmat = np.vstack([ffrng.randn(32, 2) * s for s in (0.3, 1.0, 3.0)])
ang = 2 * np.pi * (x_uni @ Bmat.T)
x_ff = np.concatenate([x_uni, np.cos(ang), np.sin(ang)], 1)
x_t = torch.tensor(x_ff, dtype=torch.float32, device=dev)
pm = 10
dx = x_mesh - x_uni
mx = dx[tr][..., 0].mean(0); my = dx[tr][..., 1].mean(0)
px = PCA(n_components=pm).fit(dx[tr][..., 0] - mx); py = PCA(n_components=pm).fit(dx[tr][..., 1] - my)
p4m, p4s = PAR4[tr].mean(0), PAR4[tr].std(0)
def feat(idx): return np.concatenate([px.transform(dx[idx][..., 0] - mx), py.transform(dx[idx][..., 1] - my)], 1)
pr = (PAR4 - p4m) / p4s
ftr = torch.tensor(feat(tr), dtype=torch.float32, device=dev)
fte = torch.tensor(feat(te), dtype=torch.float32, device=dev)
prtr = torch.tensor(pr[tr], dtype=torch.float32, device=dev)
prte = torch.tensor(pr[te], dtype=torch.float32, device=dev)
comps = {}
for nm, e in [('e11', e11), ('e22', e22), ('g12', e12)]:
    m = float(e[tr][lin[tr]].mean()); s = float(e[tr][lin[tr]].std())
    comps[nm] = {'truth': e, 'm': m, 's': s, 'tgt': torch.tensor((e[tr] - m) / s, dtype=torch.float32, device=dev)}
wm = torch.tensor(lin[tr].astype(np.float32), device=dev); wsum = wm.sum()
db1 = [pm * 2, 100, 100, 100]; db2 = [4, 150, 150, 150, 100]
dtk = [x_ff.shape[1], 100, 100, 100, 100, 100]
nets = {nm: opnn(db1, db2, dtk).to(dev).float() for nm in comps}
params = []
for n in nets.values():
    params += list(n.parameters())
opt = torch.optim.Adam(params, lr=1e-3)
decay = [int(0.6 * EPOCHS), int(0.8 * EPOCHS)]
t0 = time.time()
for ep in range(EPOCHS):
    if ep in decay:
        for g in opt.param_groups:
            g['lr'] *= 0.1
    opt.zero_grad()
    loss = 0
    for nm in comps:
        pred = nets[nm](ftr, prtr, x_t)
        loss = loss + ((pred - comps[nm]['tgt']) ** 2 * wm).sum() / wsum
    loss.backward(); opt.step()
    if ep % 20000 == 0:
        print('ep %d loss %.3e (%.0fs)' % (ep, float(loss.detach()), time.time() - t0), flush=True)
print('train done %.1f min' % ((time.time() - t0) / 60), flush=True)
torch.save({'Bmat': Bmat, **{nm: nets[nm].state_dict() for nm in nets}},
           '/home/jiang/_stage2B_ff2_model%s.pt' % TAG)
out = {'te': te}
res = {}
for nm in comps:
    nets[nm].eval()
    with torch.no_grad():
        p = nets[nm](fte, prte, x_t).cpu().numpy() * comps[nm]['s'] + comps[nm]['m']
    out[nm] = p
    truth = comps[nm]['truth']
    r = [100 * np.sqrt(np.sum((p[k][lin[ci]] - truth[ci][lin[ci]]) ** 2)
                       / (np.sum(truth[ci][lin[ci]] ** 2) + 1e-30)) for k, ci in enumerate(te)]
    res[nm] = dict(med=float(np.median(r)), mean=float(np.mean(r)), max=float(np.max(r)))
    print('%s: med %.2f mean %.2f max %.2f' % (nm, res[nm]['med'], res[nm]['mean'], res[nm]['max']), flush=True)
np.savez('/home/jiang/_stage2B_ff2_pred%s.npz' % TAG, **out)
json.dump(res, open('/home/jiang/_stage2B_ff2_result%s.json' % TAG, 'w'))
print('BFF2_DONE', flush=True)
