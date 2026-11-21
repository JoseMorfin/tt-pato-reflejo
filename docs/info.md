## How it works

A **fall reflex for a small biped robot** (Open Duck Mini v2), implemented as a trained
**differentiable logic gate network** (Petersen et al., NeurIPS 2022/2024) turned directly into silicon:
no multipliers, no weights in memory. Every neuron is one 2-input logic gate chosen during training.

The host (RP2350 or Raspberry Pi) sends one IMU frame every 20 ms: 6 signed bytes
(trunk angular velocity x, y, z and projected gravity x, y, z, int8). The chip:

1. keeps a short history (frames t, t-1, t-2),
2. thermometer-encodes the features against constants fixed at training time (plus an on-chip tilt proxy max(|gx|,|gy|)),
3. evaluates the gate network in a single clock cycle and counts votes per class (popcount),
4. debounces the alarm over K frames and reports a confidence margin,
5. also runs a **classic baseline** (tilt threshold) on the same die, so both can be compared on the real robot.

The network was trained on 8,000 MuJoCo episodes (random walking speeds and pushes of 0.3-4 N*s) with the
robot's own servo and IMU models. Evaluation is per event: did the alarm fire early enough before the fall,
and how many non-falling episodes triggered it (false alarms capped at 2 %).

Training, export to Verilog and the bit-exact golden model live in the companion repository (Python + PyTorch).

## Pinout

| pin | dir | function |
|---|---|---|
| ui[7:0] | in | data byte |
| uio[0] WR | in | rising edge writes one byte; the 6th byte of a frame triggers an evaluation |
| uio[1] CFG | in | 1 = bytes go to config registers in order: threshold, K, baseline threshold, debug select |
| uio[2] SYNC | in | 1 = reset byte pointers (re-align frames) |
| uo[0] ALARM | out | fall predicted (gate network, debounced) |
| uo[2:1] DIR | out | 0 front, 1 back, 2 left, 3 right |
| uo[3] FRAME | out | toggles after each evaluated frame |
| uo[4] ALARM_BASE | out | classic tilt baseline |
| uo[7:5] MARGIN | out | votes(fall) - votes(no fall), saturated 0-7 |
| uio[7:4] | out | debug: vote count of the selected class |

## How to test

Hold SYNC high for a few cycles, then stream frames: for each of the 6 bytes put the value on ui and pulse WR
(at least 2 clocks low, 2 clocks high). Read uo after the 6th byte. The cocotb test replays full simulated
episodes and checks every output bit against the Python reference.

## External hardware

None required. On the robot: an IMU (BNO055) read by the host MCU, which streams frames to the chip and uses
ALARM / DIR to trigger a protective posture on the 14 Feetech STS3215 servos.
