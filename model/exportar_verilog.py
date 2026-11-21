"""Red de compuertas entrenada (.pt de entrenar_lgn.py) -> Verilog combinacional + vectores dorados para cocotb.
El chip recibe los rasgos de IMU ya cuantizados a int8 (q = round(x / escala), recortado a [-127, 127]) y hace el
termometro adentro comparando contra constantes; por eso el modelo de referencia trabaja en el dominio entero y la
prueba es bit-exacta. Salidas: votos por clase (popcount de cada grupo).
Uso: exportar_verilog.py <modelo.pt> <dir_npz> <dir_salida>     (cowork 21-nov-2026)"""
import sys, glob, json, numpy as np, torch
m, dirn, out = sys.argv[1], sys.argv[2], sys.argv[3]
ck = torch.load(m, weights_only=False); sd = ck['capas']; CAPAS, ANCHO, NIV = ck['CAPAS'], ck['ANCHO'], ck['NIV']
umbr = ck['umbr']                                   # (12, NIV) en unidades fisicas
X = np.concatenate([np.load(f)['X'] for f in sorted(glob.glob(dirn + '/caida_s*.npz'))])
# escala por rasgo: el umbral mas extremo cae en +-121; todo lo que lo rebase satura sin cambiar ningun bit del termometro
esc = np.maximum(np.abs(np.r_[umbr[:6], umbr[6:]].reshape(2, 6, -1)).max(axis=(0, 2)), 1e-6) * 1.05 / 127.0
esc[3:6] = np.maximum(esc[3:6], 1.0 / 127.0)       # gravedad: cubrir +-1 g para que la linea base de inclinacion quepa
if ck.get('esc') is not None: esc = ck['esc']                 # modelo entrenado ya en dominio entero (ENTERO=1)
esc12 = np.r_[esc, esc]
tq = ck['tq'] if ck.get('tq') is not None else np.clip(np.round(umbr / esc12[:, None]), -127, 126).astype(int)
def quant(x): return np.clip(np.round(x / esc), -127, 127).astype(int)
DERIV = bool(ck.get('DERIV')); NF = tq.shape[0]; NIV = tq.shape[1]
def feats(q12):
    q12 = np.asarray(q12)
    if DERIV:
        q12 = np.c_[q12, np.maximum(np.abs(q12[:, 3]), np.abs(q12[:, 4])), np.maximum(np.abs(q12[:, 9]), np.abs(q12[:, 10]))]
    return q12
def bits_de(q12): f = feats(q12); return (f[:, :, None] > tq[None]).reshape(len(f), -1)       # (n, NF*NIV)
TIPOS = [sd['%d.w' % i].argmax(-1).numpy() for i in range(CAPAS)]
IA = [sd['%d.ia' % i].numpy() for i in range(CAPAS)]; IB = [sd['%d.ib' % i].numpy() for i in range(CAPAS)]
EXPR = ["1'b0", "{a}&{b}", "{a}&~{b}", "{a}", "~{a}&{b}", "{b}", "{a}^{b}", "{a}|{b}",
        "~({a}|{b})", "~({a}^{b})", "~{b}", "{a}|~{b}", "~{a}", "~{a}|{b}", "~({a}&{b})", "1'b1"]
C = int(ck.get("CLASES", 5)); G = ANCHO // C; NB = int(np.ceil(np.log2(G + 1)))
v = ['// generado por exportar_verilog.py desde %s (capas %d x %d, %d compuertas)' % (m.split('/')[-1], CAPAS, ANCHO, CAPAS * ANCHO),
     '`default_nettype none', 'module lgn_red (', '  input  wire [%d:0] q,   // 12 rasgos int8 con signo: q[8k+7:8k] = rasgo k' % (12 * 8 - 1),
     '  output wire [%d:0] votos  // %d clases x %d bits: votos[%d*c +: %d]' % (C * NB - 1, C, NB, NB, NB), ');']
nin = NF * NIV
v.append('  wire [%d:0] t;' % (nin - 1))
fuente = ['q[%d:%d]' % (8 * k + 7, 8 * k) for k in range(12)]
if DERIV:                                               # inclinacion aproximada: max(|gx|, |gy|) de t y de t-2
    for nm, (i, j) in (('d0', (3, 4)), ('d1', (9, 10))):
        v.append('  wire signed [7:0] %s_a = q[%d:%d], %s_b = q[%d:%d];' % (nm, 8 * i + 7, 8 * i, nm, 8 * j + 7, 8 * j))
        v.append('  wire [7:0] %s_aa = %s_a[7] ? -%s_a : %s_a, %s_ab = %s_b[7] ? -%s_b : %s_b;' % ((nm,) * 8))
        v.append('  wire [7:0] %s = (%s_aa > %s_ab) ? %s_aa : %s_ab;' % ((nm,) * 5))
    fuente += ['d0', 'd1']
for k in range(NF):
    for j in range(NIV):
        v.append("  assign t[%d] = $signed(%s) > $signed(8'h%02X);  // %d" % (k * NIV + j, fuente[k], tq[k, j] & 0xFF, tq[k, j]))
prev = 't'
for i in range(CAPAS):
    v.append("  wire [%d:0] c%d;" % (len(IA[i]) - 1, i))
    for n in range(len(IA[i])):
        a, b = '%s[%d]' % (prev, IA[i][n]), '%s[%d]' % (prev, IB[i][n])
        v.append('  assign c%d[%d] = %s;' % (i, n, EXPR[TIPOS[i][n]].format(a=a, b=b)))
    prev = 'c%d' % i
for c in range(C):                                      # popcount por grupo
    terms = ' + '.join("{%d'd0, %s[%d]}" % (NB - 1, prev, c * G + j) for j in range(G))
    v.append('  assign votos[%d +: %d] = %s;' % (NB * c, NB, terms))
v += ['endmodule', '']
import os; os.makedirs(out, exist_ok=True)
open(out + '/lgn_red.v', 'w').write('\n'.join(v))

# ---------- modelo de referencia entero (lo que el chip calcula) + vectores dorados ----------
GATES = [lambda a, b: np.zeros_like(a), lambda a, b: a & b, lambda a, b: a & ~b, lambda a, b: a, lambda a, b: ~a & b,
         lambda a, b: b, lambda a, b: a ^ b, lambda a, b: a | b, lambda a, b: ~(a | b), lambda a, b: ~(a ^ b),
         lambda a, b: ~b, lambda a, b: a | ~b, lambda a, b: ~a, lambda a, b: ~a | b, lambda a, b: ~(a & b), lambda a, b: np.ones_like(a)]
def votos_de(q12):
    x = bits_de(q12)
    for i in range(CAPAS):
        a, b = x[:, IA[i]], x[:, IB[i]]; y = np.zeros_like(a)
        for k in np.unique(TIPOS[i]):
            c = TIPOS[i] == k; y[:, c] = GATES[k](a[:, c], b[:, c])
        x = y
    return x.reshape(len(x), C, -1).sum(-1)
Z = [np.load(f) for f in sorted(glob.glob(dirn + '/caida_s*.npz'))]
Y = np.concatenate([z['Y'] for z in Z]); EP = np.concatenate([z['EP'] + 100000 * i for i, z in enumerate(Z)])
prev = np.roll(X, 2, axis=0); prim = np.r_[True, EP[1:] != EP[:-1]]; prim |= np.roll(prim, 1); prev[prim] = X[prim]
q12 = np.c_[quant(X), quant(prev)]
rng = np.random.default_rng(1)
sel = np.r_[rng.choice(np.nonzero(Y)[0], 300, replace=False), rng.choice(np.nonzero(~Y)[0], 300, replace=False)]
vv = votos_de(q12[sel])
with open(out + '/vectores.txt', 'w') as f:
    for qq, vo in zip(q12[sel], vv):
        f.write(' '.join('%d' % a for a in qq) + ' | ' + ' '.join('%d' % a for a in vo) + '\n')
# acuerdo entre el modelo flotante (termometro con umbrales reales) y el entero en todo el conjunto (muestra de 200k)
s = rng.choice(len(X), 200000, replace=False)
F12 = np.c_[X, prev][s]
xf = bits_de(q12[s]) if ck.get('tq') is not None else (F12[:, :, None] > umbr[None]).reshape(len(s), -1)
def votos_bits(x):
    for i in range(CAPAS):
        a, b = x[:, IA[i]], x[:, IB[i]]; y = np.zeros_like(a)
        for k in np.unique(TIPOS[i]):
            c = TIPOS[i] == k; y[:, c] = GATES[k](a[:, c], b[:, c])
        x = y
    return x.reshape(len(x), C, -1).sum(-1)
af = votos_bits(xf).argmax(1); ai = votos_de(q12[s]).argmax(1)
info = dict(modelo=m.split('/')[-1], capas=CAPAS, ancho=ANCHO, compuertas=CAPAS * ANCHO, bits_voto=NB,
            escala_int8=esc.tolist(), umbrales_int=tq.tolist(), acuerdo_flotante_vs_entero_pct=round(100 * float((af == ai).mean()), 2))
json.dump(info, open(out + '/lgn_red.json', 'w'), indent=1)
print(json.dumps({k: info[k] for k in ('compuertas', 'bits_voto', 'acuerdo_flotante_vs_entero_pct')}))

# ---------- episodios completos para la prueba del chip entero (historia, antirrebote, linea base) ----------
UMB = int(sys.argv[4]) if len(sys.argv) > 4 else 0; KD = int(sys.argv[5]) if len(sys.argv) > 5 else 3
SESGO_Y = int(np.round(np.median(quant(X)[:, 4][~Y])))   # postura de pie del pato (gy en reposo)
BASE = int(round(np.sin(np.radians(26)) / np.mean(esc[3:5])))
BASE = min(BASE, 255)
def chip(qf, umb=UMB, K=KD, base=BASE):
    """qf: (n, 6) cuadros int8 de un episodio -> lista de (alarma, dir, latido, base, margen) tras cada cuadro."""
    prevq = np.array([qf[i - 2] if i >= 2 else np.zeros_like(qf[i]) for i in range(len(qf))])   # chip v3: historia en 0
    vo = votos_de(np.c_[qf, prevq]); res = []; cuenta = 0; lat = 0
    if C == 2:                                       # v2: alarma de 2 clases + direccion por signo de la gravedad
        for i in range(len(qf)):
            v = vo[i]; marg = int(v[1]) - int(v[0]); cruda = marg > umb
            al = int(cruda and cuenta + 1 >= max(K, 1)); cuenta = min(cuenta + 1, 7) if cruda else 0; lat ^= 1
            dx, dy = int(qf[i, 3]), int(qf[i, 4]) - SESGO_Y
            dd = (0 if dx > 0 else 1) if abs(dx) >= abs(dy) else (2 if dy > 0 else 3)
            b = int(max(abs(int(qf[i, 3])), abs(int(qf[i, 4]))) > base)
            res.append((al, dd, lat, b, 0 if marg < 0 else min(marg, 7)))
        return res
    for i in range(len(qf)):
        v = vo[i]; f = v[1:]; dmax = int(np.argmax(f))                  # empate -> el de menor indice
        # el Verilog compara por pares (1 vs 2, 3 vs 4, luego ganador): reproducimos ese desempate exacto
        m01, d01 = (f[1], 1) if f[1] > f[0] else (f[0], 0)
        m23, d23 = (f[3], 3) if f[3] > f[2] else (f[2], 2)
        vmax, dd = (m23, d23) if m23 > m01 else (m01, d01)
        marg = int(vmax) - int(v[0]); cruda = marg > umb
        al = int(cruda and cuenta + 1 >= max(K, 1)); cuenta = min(cuenta + 1, 7) if cruda else 0; lat ^= 1
        b = int(max(abs(int(qf[i, 3])), abs(int(qf[i, 4]))) > base)
        res.append((al, dd, lat, b, 0 if marg < 0 else min(marg, 7)))
    return res
ini = np.r_[0, np.nonzero(EP[1:] != EP[:-1])[0] + 1]; fin = np.r_[ini[1:], len(EP)]
qa = quant(X); elegidos = rng.choice(len(ini), 12, replace=False)
with open(out + '/episodios.txt', 'w') as f:
    f.write('# umbral %d K %d base %d\n' % (UMB, KD, BASE))
    for e in elegidos:
        qf = qa[ini[e]:fin[e]]; f.write('EP %d %d\n' % (e, int(Y[ini[e]:fin[e]].any())))
        for row, r in zip(qf, chip(qf)):
            f.write(' '.join('%d' % a for a in row) + ' | ' + ' '.join('%d' % a for a in r) + '\n')
NBv = NB
cab = ('`define K_DEF 3\'d%d\n' % KD) + '`define LGN_NB %d\n`define UMBRAL_DEF %s\n`define BASE_DEF 8\'d%d\n`define SESGO_Y %s\n' % (
    NBv, ("8'sd%d" % UMB) if UMB >= 0 else ("-8'sd%d" % -UMB), BASE, ("8'sd%d" % SESGO_Y) if SESGO_Y >= 0 else ("-8'sd%d" % -SESGO_Y))
txt = open(out + '/lgn_red.v').read(); open(out + '/lgn_red.v', 'w').write(cab + txt)   # va primero en la lista de fuentes
print('episodios', len(elegidos), 'base', BASE)
