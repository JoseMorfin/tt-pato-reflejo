"""Evaluacion por EVENTO del reflejo anticaidas (lo que importa en el robot, no la exactitud por paso).
Para cada episodio de prueba (mismo 20 % que entrenar_lgn.py):
  - episodio con caida: detectado si la alarma (con antirrebote de K pasos) dispara en los ultimos VENT pasos antes de caer;
    se mide la anticipacion (ms). Alarma antes de esa ventana = 'prematura'.
  - episodio sin caida: cualquier alarma = falsa alarma.
Compara: (a) umbral de giroscopio, (b) umbral de inclinacion, (c) giro O inclinacion, (d) red de compuertas (.pt) con
umbral sobre el margen de votos. Barrido de umbrales -> curva; se reporta el mejor punto con falsas <= 5 % y <= 2 %.
Uso: evaluar_reflejo.py <dir_npz> [modelo.pt ...]     (cowork 21-nov-2026)"""
import sys, glob, json, numpy as np
dirn, modelos = sys.argv[1], sys.argv[2:]
HZ, VENT, RETRASO = 50, 50, 2          # ventana de deteccion valida: 1.0 s antes de la caida
X, Y, D, EP = [], [], [], []
for i, f in enumerate(sorted(glob.glob(dirn + '/caida_s*.npz'))):
    z = np.load(f); X.append(z['X']); Y.append(z['Y']); D.append(z['D']); EP.append(z['EP'] + 100000 * i)
X, Y, D, EP = map(np.concatenate, (X, Y, D, EP))
eps = np.unique(EP); rng = np.random.default_rng(0); prueba = set(rng.choice(eps, len(eps) // 5, replace=False))
mp = np.isin(EP, list(prueba))
ini = np.r_[0, np.nonzero(EP[1:] != EP[:-1])[0] + 1]; fin = np.r_[ini[1:], len(EP)]
segs = [(a, b) for a, b in zip(ini, fin) if mp[a]]
cae = np.array([Y[a:b].any() for a, b in segs])
print('episodios de prueba', len(segs), 'con caida', cae.sum(), 'sin caida', (~cae).sum())

def evento(score, thr, K):
    """score: arreglo por paso (mayor = mas riesgo). Devuelve detecta, falsas, prematuras, anticipacion mediana ms."""
    al = score > thr
    det, fa, pre, lead = 0, 0, 0, []
    for (a, b), c in zip(segs, cae):
        s = al[a:b].astype(int)
        if K > 1: s = np.convolve(s, np.ones(K, int), 'full')[:len(s)] >= K
        idx = np.nonzero(s)[0]
        if c:
            n = b - a
            if len(idx) and idx[0] < n - VENT: pre += 1
            elif len(idx): det += 1; lead.append((n - idx[0]) * 1000 / HZ)
        elif len(idx): fa += 1
    nc, ns = cae.sum(), (~cae).sum(); lead = np.array(lead)
    return det / nc, fa / max(ns, 1), pre / nc, float(np.median(lead)) if len(lead) else 0.0, \
        {L: float((lead >= L).sum() / nc) for L in ANTIC}

ANTIC = (100, 200, 300, 400)   # ms de anticipacion exigidos: el reflejo necesita tiempo para actuar (servos ~100 ms)

def barrido(nombre, score):
    qs = np.unique(np.quantile(score[mp], np.linspace(0.80, 0.9999, 60)))
    filas = []
    for K in (1, 2, 3):
        for t in qs:
            d, f, p, l, dl = evento(score, t, K); filas.append((d, f, p, l, float(t), K, dl))
    res = {'metodo': nombre}
    ok = [r for r in filas if r[1] <= 0.02]          # falsas alarmas <= 2 % de episodios sin caida
    for L in ANTIC:
        if ok:
            d, f, p, l, t, K, dl = max(ok, key=lambda r: (r[6][L], -r[1]))
            res['det@%dms' % L] = dict(detecta=round(100 * dl[L], 1), falsas=round(100 * f, 1), prematuras=round(100 * p, 1), mediana_ms=l, umbral=round(t, 3), K=K)
    print(json.dumps(res, ensure_ascii=False)); return res

gnorm = np.linalg.norm(X[:, :3], axis=1)
g = X[:, 3:6]; ref = np.median(g[~Y], axis=0); ref /= np.linalg.norm(ref)
incl = np.degrees(np.arccos(np.clip((g @ ref) / np.maximum(np.linalg.norm(g, axis=1), 1e-9), -1, 1)))
dincl = np.r_[0, np.diff(incl)] * HZ; dincl[np.r_[True, EP[1:] != EP[:-1]]] = 0
out = [barrido('extrapola_inclinacion_200ms', incl + 0.2 * dincl), barrido('umbral_giro', gnorm), barrido('umbral_inclinacion', incl),
       barrido('giro_o_inclinacion', np.maximum(gnorm / np.quantile(gnorm[mp], 0.99), incl / np.quantile(incl[mp], 0.99)))]

GATES = [lambda a, b: np.zeros_like(a), lambda a, b: a & b, lambda a, b: a & ~b, lambda a, b: a, lambda a, b: ~a & b,
         lambda a, b: b, lambda a, b: a ^ b, lambda a, b: a | b, lambda a, b: ~(a | b), lambda a, b: ~(a ^ b),
         lambda a, b: ~b, lambda a, b: a | ~b, lambda a, b: ~a, lambda a, b: ~a | b, lambda a, b: ~(a & b), lambda a, b: np.ones_like(a)]
prev = np.roll(X, RETRASO, axis=0); prim = np.r_[True, EP[1:] != EP[:-1]]
for k in range(1, RETRASO): prim |= np.roll(prim, k)
prev[prim] = X[prim]; F12 = np.c_[X, prev]
for m in modelos:
    if m.endswith('.npy'):                       # puntaje ya calculado (p. ej. MLP de referencia en flotante)
        out.append(barrido(m.split('/')[-1], np.load(m))); continue
    import torch
    ck = torch.load(m, weights_only=False); sd = ck['capas']
    if ck.get('tq') is not None:                 # dominio entero, identico al chip
        e12 = np.r_[ck['esc'], ck['esc']]; Q = np.clip(np.round(F12 / e12), -127, 127)
        if ck.get('DERIV'):
            Q = np.c_[Q, np.maximum(np.abs(Q[:, 3]), np.abs(Q[:, 4])), np.maximum(np.abs(Q[:, 9]), np.abs(Q[:, 10]))]
        x = (Q[:, :, None] > ck['tq'][None]).reshape(len(Q), -1)
    else:
        x = (F12[:, :, None] > ck['umbr'][None]).reshape(len(F12), -1)
    for i in range(ck['CAPAS']):
        ia, ib, gt = sd['%d.ia' % i].numpy(), sd['%d.ib' % i].numpy(), sd['%d.w' % i].argmax(-1).numpy()
        a, b = x[:, ia], x[:, ib]; y = np.zeros_like(a)
        for k in np.unique(gt):
            c = gt == k; y[:, c] = GATES[k](a[:, c], b[:, c])
        x = y
    v = x.reshape(len(x), ck.get('CLASES', 5), -1).sum(-1).astype(float)
    out.append(barrido(m.split('/')[-1], v[:, 1:].max(1) - v[:, 0]))
json.dump(out, open('eval_reflejo.json', 'w'), indent=1, ensure_ascii=False)
