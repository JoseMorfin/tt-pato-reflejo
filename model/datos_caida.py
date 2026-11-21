"""Dataset para el chip anticaidas (cowork 21-nov-2026). Episodios en MuJoCo con la politica ONNX caminando:
velocidad aleatoria, empujon aleatorio (direccion, impulso, instante), ruido de IMU. Por cada paso de control (50 Hz) guarda
6 rasgos de IMU en el marco del torso (giro x,y,z en rad/s; gravedad proyectada x,y,z) y la etiqueta de caida futura.
Caida = altura del torso < 0.10 m o inclinacion > 60 grados. Direccion = cuadrante de la inclinacion al caer (0 frente, 1 atras, 2 izq, 3 der).
Uso (desde Open_Duck_Playground con su .venv): datos_caida.py <onnx> <n_episodios> <semilla> <salida.npz>"""
import sys, time, contextlib, numpy as np, mujoco, mujoco.viewer
time.sleep = lambda s: None   # sin pausa de tiempo real
onnx, N, semilla, sal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(semilla)
from playground.open_duck_mini_v2.mujoco_infer import MjInfer
def nuevo():
    return MjInfer("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml", "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl", onnx, False)
mj = nuevo(); m, d = mj.model, mj.data
TR = m.body('trunk_assembly').id
PASOS_CTRL = int(round(0.02 / m.opt.timestep))
X, Y, D, EP = [], [], [], []


class Alto(Exception):
    pass


def rasgos():
    R = d.xmat[TR].reshape(3, 3)
    w = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, TR, w, 1)   # marco local
    g = R.T @ np.array([0, 0, -1.0])
    return np.r_[w[:3], g]


class Registro:
    def __init__(s, t0, imp, ang, dur, tfin):
        s.n, s.t0, s.imp, s.ang, s.dur, s.tfin = 0, t0, imp, ang, dur, tfin
        s.f, s.caida_t, s.dir = [], None, -1

    def sync(s):
        s.n += 1; t = s.n * m.opt.timestep
        d.xfrc_applied[TR, :] = 0
        if s.t0 <= t < s.t0 + s.dur:
            F = s.imp / s.dur
            d.xfrc_applied[TR, 0] = F * np.cos(s.ang); d.xfrc_applied[TR, 1] = F * np.sin(s.ang)
        if s.n % PASOS_CTRL == 0:
            s.f.append(rasgos() + rng.normal(0, [0.05, 0.05, 0.05, 0.02, 0.02, 0.02]))
        z = float(d.xpos[TR, 2]); R = d.xmat[TR].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        if s.caida_t is None and (z < 0.10 or tilt > 60):
            s.caida_t = len(s.f)
            ax, ay = R[0, 2], R[1, 2]   # hacia donde apunta el eje z del torso
            s.dir = (0 if ax > 0 else 1) if abs(ax) >= abs(ay) else (2 if ay > 0 else 3)
            raise Alto()
        if t >= s.tfin:
            raise Alto()


for e in range(N):
    mj = nuevo(); m, d = mj.model, mj.data   # estado limpio de la politica y del simulador
    mj.commands[0] = float(rng.uniform(0.0, 0.15))
    imp = float(rng.choice([0.0, rng.uniform(0.3, 4.0)], p=[0.2, 0.8]))
    reg = Registro(float(rng.uniform(1.5, 4.0)), imp, float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(0.05, 0.2)), 7.0)
    with contextlib.suppress(Alto):
        mujoco.viewer.launch_passive = lambda *a, **k: contextlib.nullcontext(reg)
        mj.run()
    f = np.array(reg.f); n = len(f)
    if n == 0: continue
    paso_caida = reg.caida_t if reg.caida_t is not None else 10 ** 9
    X.append(f); EP.append(np.full(n, e))
    Y.append(np.arange(n) >= paso_caida - 25)     # caida dentro de los proximos 0.5 s
    D.append(np.full(n, reg.dir))
    if e % 50 == 0: print(e, 'caidas', sum(1 for y in Y if y.any()), flush=True)
np.savez_compressed(sal, X=np.concatenate(X), Y=np.concatenate(Y), D=np.concatenate(D), EP=np.concatenate(EP))
print('listo', sal, sum(len(x) for x in X), 'muestras')
