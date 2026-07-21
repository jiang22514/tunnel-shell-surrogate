#!/usr/bin/env python3
"""Task 3 prerequisite: predicted-temperature profile heads (T_c0, T_slope per
shat station) so the GenieShan N/M is fully operator-predicted (no truth T).
Same architecture/branch/trunk as _stage3_shell_train v3. argv: [EPOCHS] [KH] [TAG]"""
import sys, time
import numpy as np, torch
from sklearn.decomposition import PCA
sys.path.insert(0, '/home/jiang/fshapesTK/tunnel_model/DIMON')
from opnn import opnn

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
KH = int(sys.argv[2]) if len(sys.argv) > 2 else 32
TAG = sys.argv[3] if len(sys.argv) > 3 else ''
NOBC = '--nobc' in sys.argv
NOPCA = '--nopca' in sys.argv
if TAG:
    torch.manual_seed(abs(hash(TAG)) % 100000 + 11)
dev = torch.device('cuda')
D = '/home/jiang/fshapesTK/tunnel_model/simplified_tunnel/data_E_CT_L_T_v2_L8'
d1 = np.load(f'{D}/u1_dataset.npz'); d2 = np.load(f'{D}/u2_dataset.npz')
PR = np.load(f'{D}/profiles_NM.npz'); TT = np.load(f'{D}/tprofiles_T.npz')
PAR4 = np.load(f'{D}/ct_params.npy')[:, :4]
x_uni = d1['x_uni']; x_mesh = d1['all_tunnel_def_grids']
bc1 = d1['bc_u1_top']; bc2 = d2['bc_u2_top']
Ncase = x_mesh.shape[0]
FIELDS = {'T_c0': TT['T_c0'], 'T_slope': TT['T_slope']}
sc = PR['sc']; valid = PR['valid']
rng = np.random.RandomState(0); perm = rng.permutation(Ncase)
te = perm[:20]; tr = perm[20:]
ks = np.arange(1, KH + 1)
ang = 2 * np.pi * sc[:, None] * ks[None, :]
x_st = np.concatenate([np.cos(ang), np.sin(ang)], 1)
x_t = torch.tensor(x_st, dtype=torch.float32, device=dev)
pm = 10
dx = x_mesh - x_uni
mx = dx[tr][..., 0].mean(0); my = dx[tr][..., 1].mean(0)
px = PCA(n_components=pm).fit(dx[tr][..., 0] - mx); py = PCA(n_components=pm).fit(dx[tr][..., 1] - my)
p4m, p4s = PAR4[tr].mean(0), PAR4[tr].std(0)
if NOBC:
    def feat(idx):
        f_ = np.concatenate([px.transform(dx[idx][..., 0] - mx),
                             py.transform(dx[idx][..., 1] - my)], 1)
        return np.ones_like(f_) if NOPCA else f_
    bc = (PAR4 - p4m) / p4s
else:
    def feat(idx): return np.concatenate([px.transform(dx[idx][..., 0] - mx),
                                          py.transform(dx[idx][..., 1] - my),
                                          (PAR4[idx] - p4m) / p4s], 1)
    b1m, b1s = bc1[tr].mean(), bc1[tr].std(); b2m, b2s = bc2[tr].mean(), bc2[tr].std()
    bc = np.concatenate([(bc1 - b1m) / b1s, (bc2 - b2m) / b2s], 1)
ftr = torch.tensor(feat(tr), dtype=torch.float32, device=dev)
fte = torch.tensor(feat(te), dtype=torch.float32, device=dev)
bctr = torch.tensor(bc[tr], dtype=torch.float32, device=dev)
bcte = torch.tensor(bc[te], dtype=torch.float32, device=dev)
wm = torch.tensor(valid[tr].astype(np.float32), device=dev); wsum = wm.sum()
norm, tgt = {}, {}
for f, a in FIELDS.items():
    m = float(a[tr][valid[tr]].mean()); s = float(a[tr][valid[tr]].std())
    norm[f] = (m, s)
    tgt[f] = torch.tensor((a[tr] - m) / s, dtype=torch.float32, device=dev)
db1 = [pm * 2 + (0 if NOBC else 4), 100, 100, 100]; db2 = [bc.shape[1], 150, 150, 150, 100]
dtk = [x_st.shape[1], 100, 100, 100, 100, 100]
nets = {f: opnn(db1, db2, dtk).to(dev).float() for f in FIELDS}
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
    for f in FIELDS:
        loss = loss + ((nets[f](ftr, bctr, x_t) - tgt[f]) ** 2 * wm).sum() / wsum
    loss.backward(); opt.step()
    if ep % 20000 == 0:
        print('ep %d loss %.3e (%.0fs)' % (ep, float(loss.detach()), time.time() - t0), flush=True)
print('train done %.1f min' % ((time.time() - t0) / 60), flush=True)
out = {'te': te, 'sc': sc}
for f, a in FIELDS.items():
    nets[f].eval()
    with torch.no_grad():
        out[f] = nets[f](fte, bcte, x_t).cpu().numpy() * norm[f][1] + norm[f][0]
    r = [100 * np.sqrt(np.sum((out[f][k][valid[c]] - a[c][valid[c]]) ** 2)
                       / (np.sum((a[c][valid[c]] - a[c][valid[c]].mean() * 0) ** 2) + 1e-30))
         for k, c in enumerate(te)]
    # rel-L2 on (T - T_REF) scale would divide by ~24C; report absolute RMSE in C instead
    rmse = [float(np.sqrt(np.mean((out[f][k][valid[c]] - a[c][valid[c]]) ** 2))) for k, c in enumerate(te)]
    print('%-8s RMSE: med %.4f mean %.4f max %.4f (C or C/n-unit)' %
          (f, np.median(rmse), np.mean(rmse), np.max(rmse)), flush=True)
FULLTAG = TAG + ('_nobc' if NOBC else '') + ('_nopca' if NOPCA else '')
torch.save({f: nets[f].state_dict() for f in FIELDS}, '/home/jiang/_stage3_T_model%s.pt' % FULLTAG)
np.savez('/home/jiang/_stage3_T_pred%s.npz' % FULLTAG, **out)
print('STAGE3_T_DONE', flush=True)
