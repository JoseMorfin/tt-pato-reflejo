"""Techo de referencia en punto flotante: MLP pequeno con H cuadros de historia (cada 2 pasos = 40 ms).
Dice cuanto se puede anticipar con estos 6 rasgos de IMU y cuanta historia necesita el chip.
Guarda puntaje por paso (max votos de caida - no caida) en mlp_H<h>.npy para evaluar_reflejo.py.
Uso: mlp_ref.py <dir_npz> <H> <epocas>     (cowork 21-nov-2026)"""
import sys, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dirn, H, EPOCAS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0); torch.set_num_threads(4)
X, Y, D, EP = [], [], [], []
for i, f in enumerate(sorted(glob.glob(dirn + '/caida_s*.npz'))):
    z = np.load(f); X.append(z['X']); Y.append(z['Y']); D.append(z['D']); EP.append(z['EP'] + 100000 * i)
X, Y, D, EP = map(np.concatenate, (X, Y, D, EP))
import os
if os.environ.get('VENT_ETQ'):                         # reetiquetar: 'cae' = los ultimos W pasos de un episodio con caida
    W = int(os.environ['VENT_ETQ']); i0 = np.r_[0, np.nonzero(EP[1:] != EP[:-1])[0] + 1]; f0 = np.r_[i0[1:], len(EP)]
    largo = np.diff(np.r_[i0, len(EP)]); cae_ep = np.add.reduceat(Y, i0) > 0
    faltan = np.repeat(f0, largo) - np.arange(len(EP)); dfin = np.repeat(D[f0 - 1], largo)
    Y = np.repeat(cae_ep, largo) & (faltan <= W); D = np.where(Y, dfin, D)
ini = np.r_[0, np.nonzero(EP[1:] != EP[:-1])[0] + 1]
pos = np.arange(len(EP)) - np.repeat(ini, np.diff(np.r_[ini, len(EP)]))      # paso dentro del episodio
cols = []
for h in range(H):
    r = 2 * h; p = np.roll(X, r, axis=0); m = pos < r; p[m] = X[m]; cols.append(p)
Fx = np.concatenate(cols, 1); mu, sd = Fx.mean(0), Fx.std(0) + 1e-6; Fx = ((Fx - mu) / sd).astype(np.float32)
lab = np.where(Y, D + 1, 0)
eps = np.unique(EP); rng = np.random.default_rng(0); prueba = set(rng.choice(eps, len(eps) // 5, replace=False))
mp = np.isin(EP, list(prueba))
net = nn.Sequential(nn.Linear(Fx.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))
w = torch.tensor(1.0 / np.maximum(np.bincount(lab[~mp], minlength=5), 1), dtype=torch.float32); w = w / w.sum() * 5
opt = torch.optim.Adam(net.parameters(), 3e-3)
Xt, yt = torch.tensor(Fx[~mp]), torch.tensor(lab[~mp])
for e in range(EPOCAS):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 2048):
        j = perm[i:i + 2048]; loss = F.cross_entropy(net(Xt[j]), yt[j], weight=w); opt.zero_grad(); loss.backward(); opt.step()
    print('H', H, 'epoca', e, 'perdida %.3f' % loss.item(), flush=True)
with torch.no_grad():
    lo = torch.cat([net(torch.tensor(Fx[i:i + 200000])) for i in range(0, len(Fx), 200000)]).numpy()
nom = 'mlp_H%d%s.npy' % (H, ('_W' + os.environ['VENT_ETQ']) if os.environ.get('VENT_ETQ') else '')
np.save(nom, lo[:, 1:].max(1) - lo[:, 0]); print('listo', nom)
