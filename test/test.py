# Pruebas cocotb del reflejo anticaidas: episodios completos de simulacion MuJoCo, comparados bit a bit contra el
# modelo de referencia en Python (exportar_verilog.py). cowork 21-nov-2026
import os, cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

AQUI = os.path.dirname(os.path.abspath(__file__))
def episodios():
    eps, cur = [], None
    for l in open(os.path.join(AQUI, '..', 'src', 'episodios.txt')):
        if l.startswith('#'): continue
        if l.startswith('EP'): cur = []; eps.append(cur); continue
        a, b = l.split('|'); cur.append(([int(v) for v in a.split()], [int(v) for v in b.split()]))
    return eps

async def reiniciar(dut):
    dut.rst_n.value = 0; dut.ui_in.value = 0; dut.uio_in.value = 0
    await ClockCycles(dut.clk, 5); dut.rst_n.value = 1; await ClockCycles(dut.clk, 2)

async def byte(dut, v, cfg=0):
    dut.ui_in.value = v & 0xFF; dut.uio_in.value = cfg << 1; await ClockCycles(dut.clk, 2)
    dut.uio_in.value = (cfg << 1) | 1; await ClockCycles(dut.clk, 3)
    dut.uio_in.value = cfg << 1; await ClockCycles(dut.clk, 2)

@cocotb.test()
async def episodios_completos(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit='ns').start()); dut.ena.value = 1
    n = mal = alarmas = 0
    for ep in episodios():
        await reiniciar(dut)
        for cuadro, esperado in ep:
            for v in cuadro: await byte(dut, v)
            await ClockCycles(dut.clk, 4)
            o = int(dut.uo_out.value)
            got = [o & 1, (o >> 1) & 3, (o >> 3) & 1, (o >> 4) & 1, (o >> 5) & 7]
            if got != esperado:
                mal += 1
                if mal < 10: dut._log.error('cuadro %d: chip %s esperado %s' % (n, got, esperado))
            n += 1; alarmas += got[0]
    dut._log.info('cuadros %d, diferencias %d, alarmas %d' % (n, mal, alarmas))
    assert mal == 0

@cocotb.test()
async def configuracion_y_sync(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit='ns').start()); dut.ena.value = 1
    await reiniciar(dut)
    dut.uio_in.value = 4; await ClockCycles(dut.clk, 2); dut.uio_in.value = 0     # SYNC
    for v in (127, 1, 0, 4): await byte(dut, v, cfg=1)                             # umbral 127 -> nunca alarma
    for _ in range(5):
        for v in (100, -100, 50, 90, 90, 0): await byte(dut, v)
        await ClockCycles(dut.clk, 4)
        o = int(dut.uo_out.value)
        assert o & 1 == 0, 'con umbral 127 no debe sonar'
        assert (o >> 4) & 1 == 1, 'linea base: |g| horizontal 90 > 0 debe sonar'
    assert int(dut.uio_oe.value) == 0xF0
