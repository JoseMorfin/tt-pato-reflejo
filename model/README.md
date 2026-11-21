# Model: from MuJoCo to logic gates

1. `datos_caida.py` - runs the Open Duck Mini v2 walking policy (ONNX) in MuJoCo with random speeds and pushes and logs
   6 IMU features at 50 Hz plus a "falls within the next W steps" label. 8 x 1000 episodes, ~2.05 M samples.
2. `entrenar_lgn.py` - differentiable logic gate network (Petersen et al., NeurIPS 2022/2024) in plain PyTorch.
   Final run: `ENTERO=1 DERIV=1 CLASES=2 NIV=16 VENT_ETQ=60 LR=0.01 COSENO=1 ANCHOS=200,200`, 25 epochs, CPU.
   `ENTERO=1` trains on the int8 thermometer the chip computes, so silicon and model agree bit for bit.
3. `evaluar_reflejo.py` - per-event evaluation (warning time vs false alarms) against classic rules and float MLPs.
4. `mlp_ref.py` - float MLP upper bound with H frames of history.
5. `exportar_verilog.py` - .pt -> `src/lgn_red.v` + golden episodes for the cocotb test (`src/episodios.txt`).
6. `curva_empujes.py` - push-robustness curve of the walking policy (fall rate vs push size).

Key lessons: the thermometer thresholds must cover the region where falls happen (balanced percentiles), a binary
alarm trains far better than 5 classes, and depth helps (4 x 300 reaches 74 % at 200 ms) but costs area.
