"""Curva de robustez a empujones de una politica ONNX (cowork 21-nov-2026).
Para cada impulso fijo (N*s) corre episodios caminando (velocidad 0-0.15 m/s al azar), empuja el torso en direccion
al azar durante 0.1 s en t = 2-3 s y mide si cae (altura < 0.10 m o inclinacion > 60 grados) antes de t = 6 s.
Referencia: el entrenamiento empuja con cambios de velocidad de 0.1-1.0 m/s; con ~1.3 kg eso es ~0.13-1.3 N*s.
Uso (desde Open_Duck_Playground con su .venv, PYTHONPATH=.): curva_empujes.py <onnx> <impulso> <n> <semilla> <salida.json>"""
import sys, time, json, contextlib, numpy as np, mujoco, mujoco.viewer
time.sleep = lambda s: None
onnx, IMP, N, semilla, sal = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rng = np.random.default_rng(semilla)
from playground.open_duck_mini_v2.mujoco_infer import MjInfer


class Alto(Exception):
    pass


class Reg:
    def __init__(s, m, d, t0, ang):
        s.m, s.d, s.n, s.t0, s.ang, s.cae = m, d, 0, t0, ang, False
        s.TR = m.body('trunk_assembly').id

    def sync(s):
        m, d = s.m, s.d
        s.n += 1; t = s.n * m.opt.timestep
        d.xfrc_applied[s.TR, :] = 0
        if s.t0 <= t < s.t0 + 0.1:
            F = IMP / 0.1; d.xfrc_applied[s.TR, 0] = F * np.cos(s.ang); d.xfrc_applied[s.TR, 1] = F * np.sin(s.ang)
        R = d.xmat[s.TR].reshape(3, 3)
        if d.xpos[s.TR, 2] < 0.10 or np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))) > 60:
            s.cae = True; raise Alto()
        if t >= 6.0: raise Alto()


caidas = 0; masa = None
for e in range(N):
    mj = MjInfer("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
                 "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl", onnx, False)
    if masa is None: masa = float(mj.model.body_subtreemass[1])
    mj.commands[0] = float(rng.uniform(0.0, 0.15))
    reg = Reg(mj.model, mj.data, float(rng.uniform(2.0, 3.0)), float(rng.uniform(-np.pi, np.pi)))
    with contextlib.suppress(Alto):
        mujoco.viewer.launch_passive = lambda *a, **k: contextlib.nullcontext(reg)
        mj.run()
    caidas += reg.cae
json.dump({'onnx': onnx.split('/')[-1], 'impulso_Ns': IMP, 'masa_kg': masa, 'dv_mps': IMP / masa if masa else None,
           'episodios': N, 'caidas': caidas, 'tasa': caidas / N}, open(sal, 'w'))
print(IMP, caidas, '/', N)
