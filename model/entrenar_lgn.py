"""Red de compuertas logicas diferenciable (estilo Petersen et al. NeurIPS 2022) para el reflejo anticaidas del pato.
Implementacion propia en PyTorch (CPU/MPS, sin CUDA). Cada neurona = 2 entradas fijas al azar + mezcla softmax de las 16
compuertas de 2 entradas; al final se elige la compuerta mas probable y la red ES un circuito.
Entrada: 6 rasgos de IMU x 2 cuadros (actual y hace 2 pasos) codificados en termometro de 8 niveles = 96 bits.
Salida: 5 clases (sin caida, cae al frente, atras, izq, der) por conteo de votos (GroupSum).
Uso: entrenar_lgn.py <dir_npz> <capas> <ancho> <epocas> <salida.pt>   (cowork 21-nov-2026)"""
import sys, glob, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
dirn, CAPAS, ANCHO, EPOCAS, SAL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.manual_seed(0); np.random.seed(0)
import os
NIV, RETRASO = 8, 2
CLASES = int(os.environ.get('CLASES', 5))            # 2 = solo alarma (la direccion la da el signo de la gravedad en el chip)
RESID = os.environ.get('RESID', '0') == '1'          # inicializacion residual (Petersen et al. NeurIPS 2024): compuertas = paso 'a'
TAU = float(os.environ.get('TAU', 8.0))

# ---------- datos ----------
X, Y, D, EP = [], [], [], []
for i, f in enumerate(sorted(glob.glob(dirn + '/caida_s*.npz'))):
    z = np.load(f); X.append(z['X']); Y.append(z['Y']); D.append(z['D']); EP.append(z['EP'] + 100000 * i)
X, Y, D, EP = map(np.concatenate, (X, Y, D, EP))
import os
if os.environ.get('VENT_ETQ'):                         # reetiquetar: 'cae' = los ultimos W pasos de un episodio con caida
    W = int(os.environ['VENT_ETQ']); ini = np.r_[0, np.nonzero(EP[1:] != EP[:-1])[0] + 1]; fin = np.r_[ini[1:], len(EP)]
    largo = np.diff(np.r_[ini, len(EP)]); cae_ep = np.add.reduceat(Y, ini) > 0
    faltan = np.repeat(fin, largo) - np.arange(len(EP)); dfin = np.repeat(D[fin - 1], largo)
    Y = np.repeat(cae_ep, largo) & (faltan <= W); D = np.where(Y, dfin, D)
prev = np.roll(X, RETRASO, axis=0); prim = np.r_[True, EP[1:] != EP[:-1]]
for k in range(1, RETRASO): prim |= np.roll(prim, k)
prev[prim] = X[prim]
F12 = np.c_[X, prev]
etiqueta = np.where(Y, D + 1, 0) if CLASES == 5 else Y.astype(int)
umbr = np.percentile(F12, np.linspace(0, 100, NIV + 2)[1:-1], axis=0).T          # (12, 8) umbrales por percentiles
import os
ENTERO = os.environ.get('ENTERO', '0') == '1'        # termometro en dominio int8 (lo que hace el chip): bit-exacto por construccion
esc = tq = None
DERIV = os.environ.get('DERIV', '0') == '1'          # agrega inclinacion aprox. max(|gx|,|gy|) de t y t-2 (el chip la calcula)
if ENTERO:
    # v2 (21-nov 06:20): los umbrales por percentil de TODO el conjunto quedaban en +-3 grados y +-0.3 rad/s (caminata
    # tranquila) y la red no veia la zona donde ocurren las caidas. Ahora: escala que cubre las caidas y umbrales por
    # percentiles BALANCEADOS (mitad pasos pre-caida, mitad normales).
    NIV = int(os.environ.get('NIV', NIV))
    bal = np.r_[np.nonzero(Y)[0], np.random.default_rng(3).choice(np.nonzero(~Y)[0], Y.sum(), replace=False)]
    esc = np.percentile(np.abs(X[bal]), 99.5, axis=0) / 127.0
    esc[3:6] = 1.0 / 127.0                              # gravedad: +-1 g
    e12 = np.r_[esc, esc]; Q = np.clip(np.round(F12 / e12), -127, 127)
    if DERIV:
        Q = np.c_[Q, np.maximum(np.abs(Q[:, 3]), np.abs(Q[:, 4])), np.maximum(np.abs(Q[:, 9]), np.abs(Q[:, 10]))]
    tq = np.floor(np.percentile(Q[bal], np.linspace(0, 100, NIV + 2)[1:-1], axis=0).T).astype(int)
    for k in range(1, NIV): tq[:, k] = np.maximum(tq[:, k], tq[:, k - 1] + 1)   # estrictamente crecientes
    tq = np.clip(tq, -127, 126); F12 = Q; umbr_eff = tq
else:
    umbr_eff = umbr
def termo(a): return (a[:, :, None] > umbr_eff[None]).reshape(len(a), -1).astype(np.float32)
B = termo(F12)
eps = np.unique(EP); rng = np.random.default_rng(0); prueba = set(rng.choice(eps, len(eps) // 5, replace=False))
mp = np.isin(EP, list(prueba))
Xtr, ytr, Xte, yte = B[~mp], etiqueta[~mp], B[mp], etiqueta[mp]
print('muestras', len(B), 'bits', B.shape[1], 'clases', np.bincount(etiqueta, minlength=CLASES))

# ---------- las 16 compuertas de 2 entradas (relajacion real) ----------
def compuertas(a, b):
    ab = a * b
    return torch.stack([torch.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab,
                        1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a, 1 - a + ab, 1 - ab, torch.ones_like(a)], -1)

class Capa(nn.Module):
    def __init__(s, nin, nout):
        super().__init__()
        s.register_buffer('ia', torch.randint(0, nin, (nout,))); s.register_buffer('ib', torch.randint(0, nin, (nout,)))
        w0 = torch.randn(nout, 16) * 0.1
        if RESID: w0[:, 3] += 5.0                     # ~90 % de probabilidad inicial en la compuerta 'a' (pasa el bit)
        s.w = nn.Parameter(w0)
    def forward(s, x, duro=False):
        g = compuertas(x[:, s.ia], x[:, s.ib])
        p = F.one_hot(s.w.argmax(-1), 16).float() if duro else F.softmax(s.w, -1)
        return (g * p).sum(-1)

ANCHOS = [int(a) for a in os.environ['ANCHOS'].split(',')] if os.environ.get('ANCHOS') else [ANCHO] * CAPAS
CAPAS, ANCHO = len(ANCHOS), ANCHOS[-1]               # la ultima capa define los votos (area del popcount)
capas = nn.ModuleList([Capa(Xtr.shape[1] if i == 0 else ANCHOS[i - 1], ANCHOS[i]) for i in range(CAPAS)])
def red(x, duro=False):
    for c in capas: x = c(x, duro)
    return x.view(len(x), CLASES, -1).sum(-1) / TAU            # GroupSum: votos por clase
assert ANCHO % CLASES == 0
peso = torch.tensor(1.0 / np.maximum(np.bincount(ytr, minlength=CLASES), 1), dtype=torch.float32); peso = peso / peso.sum() * CLASES
opt = torch.optim.Adam(capas.parameters(), lr=float(os.environ.get('LR', 0.05)))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS, eta_min=float(os.environ.get('LR', 0.05)) * 0.05) \
    if os.environ.get('COSENO') == '1' else None
Xt, yt = torch.tensor(Xtr), torch.tensor(ytr); Xv, yv = torch.tensor(Xte), torch.tensor(yte)
for ep in range(EPOCAS):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 1024):
        j = perm[i:i + 1024]; loss = F.cross_entropy(red(Xt[j]), yt[j], weight=peso)
        opt.zero_grad(); loss.backward(); opt.step()
    if sched: sched.step()
    with torch.no_grad():
        pr = red(Xv, duro=True).argmax(-1)
        cae_real, cae_pred = yv > 0, pr > 0
        rec = (cae_pred & cae_real).sum().item() / max(cae_real.sum().item(), 1)
        fa = (cae_pred & ~cae_real).sum().item() / max((~cae_real).sum().item(), 1)
        dir_ok = (pr[cae_real & cae_pred] == yv[cae_real & cae_pred]).float().mean().item() if (cae_real & cae_pred).any() else 0
    print('epoca %d perdida %.3f | detecta %.1f %% de los pasos pre-caida, falsa alarma %.2f %%, direccion %.1f %%' % (ep, loss.item(), 100 * rec, 100 * fa, 100 * dir_ok), flush=True)
    torch.save({'capas': capas.state_dict(), 'umbr': umbr, 'CAPAS': CAPAS, 'ANCHO': ANCHO, 'NIV': NIV, 'nin': Xtr.shape[1],
                'esc': esc, 'tq': tq, 'epoca': ep, 'DERIV': DERIV, 'CLASES': CLASES}, SAL)          # punto de control cada epoca
json.dump({'compuertas_logicas': CAPAS * ANCHO, 'detecta_pct': 100 * rec, 'falsa_alarma_pct': 100 * fa, 'direccion_pct': 100 * dir_ok}, open(SAL + '.json', 'w'), indent=1)
