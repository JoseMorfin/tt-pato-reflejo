/*
 * Reflejo anticaidas para robot bipedo: red de compuertas logicas entrenada (DLGN) en silicio.
 * El anfitrion (RP2350 / Pi) manda cada 20 ms un cuadro de 6 bytes int8 de la IMU:
 *   [0..2] velocidad angular del tronco (x, y, z)   [3..5] gravedad proyectada en el tronco (x, y, z)
 * El chip guarda la historia (t, t-1, t-2), hace el termometro contra constantes y evalua la red en un ciclo.
 * Pines:
 *   ui_in[7:0]   byte de datos
 *   uio_in[0]    WR   flanco de subida = escribir byte (al 6.o byte del cuadro se evalua)
 *   uio_in[1]    CFG  1 = los bytes van a configuracion: umbral, K, umbral_base, sel (en ese orden)
 *   uio_in[2]    SYNC 1 = reinicia los punteros de byte (resincroniza el cuadro)
 *   uo_out[0]    ALARMA       red de compuertas, con antirrebote de K cuadros
 *   uo_out[2:1]  DIRECCION    0 frente, 1 atras, 2 izquierda, 3 derecha
 *   uo_out[3]    CUADRO       cambia de valor en cada cuadro evaluado (latido)
 *   uo_out[4]    ALARMA_BASE  linea base clasica en el mismo chip: max(|gx|,|gy|) > umbral_base
 *   uo_out[7:5]  MARGEN       votos de caida - votos de no caida, saturado 0..7
 *   uio_out[7:4] votos de la clase 'sel' (depuracion)
 * cowork 21-nov-2026, José Morfín
 */
`default_nettype none
`ifndef LGN_NB
`define LGN_NB 4
`endif
`ifndef UMBRAL_DEF
`define UMBRAL_DEF 8'sd0
`endif
`ifndef SESGO_Y
`define SESGO_Y 8'sd8
`endif
`ifndef BASE_DEF
`define BASE_DEF 8'd53
`endif
module tt_um_josemorfin_reflejo (
    input  wire [7:0] ui_in, output wire [7:0] uo_out,
    input  wire [7:0] uio_in, output wire [7:0] uio_out, output wire [7:0] uio_oe,
    input  wire ena, input wire clk, input wire rst_n);
  localparam NB = `LGN_NB;
  reg [2:0] wr_s;
  always @(posedge clk) wr_s <= rst_n ? {wr_s[1:0], uio_in[0]} : 3'b000;
  wire wr = wr_s[1] & ~wr_s[2];
  wire cfg = uio_in[1], sync = uio_in[2];

  // ---------- captura de cuadros y configuracion ----------
  reg [2:0] ptr; reg [1:0] cptr; reg [47:0] f0, f1, f2; reg listo;
  reg signed [7:0] umbral; reg [2:0] K; reg [7:0] umbral_base; reg [2:0] sel;
  always @(posedge clk) begin
    listo <= 1'b0;
    if (!rst_n) begin
      ptr <= 0; cptr <= 0; f0 <= 0; f1 <= 0; f2 <= 0;
      umbral <= `UMBRAL_DEF; K <= 3'd3; umbral_base <= `BASE_DEF; sel <= 3'd0;
    end else if (sync) begin
      ptr <= 0; cptr <= 0;
    end else if (wr && cfg) begin
      case (cptr)
        2'd0: umbral <= ui_in;
        2'd1: K <= ui_in[2:0];
        2'd2: umbral_base <= ui_in;
        2'd3: sel <= ui_in[2:0];
      endcase
      cptr <= cptr + 2'd1;
    end else if (wr) begin
      // v2 (area): sin bufer; el primer byte de cada cuadro recorre la historia y los bytes van directo a f0.
      // No hay evaluacion a medio cuadro: 'listo' solo se levanta con el 6.o byte.
      // v3 (area): registro de corrimiento, sin demultiplexor por indice de byte
      if (ptr == 3'd0) begin f2 <= f1; f1 <= f0; end
      f0 <= {ui_in, f0[47:8]};                          // tras 6 bytes: byte k en f0[8k+7:8k]
      if (ptr == 3'd5) begin ptr <= 0; listo <= 1'b1; end
      else ptr <= ptr + 3'd1;
    end
  end

  // ---------- red de compuertas (combinacional, generada) ----------
  // al inicio, sin historia, el cuadro t-2 se toma igual al actual (como en el entrenamiento)
  wire [47:0] previo = f2;                         // v3: al arrancar la historia vale 0 (2 cuadros, robot de pie)
  // 2 clases: votos[0] = no cae, votos[1] = cae.  (v2, 21-nov: la direccion ya no la vota la red, ver abajo)
  wire [2*NB-1:0] votos;
  lgn_red red (.q({previo, f0}), .votos(votos));
  wire [NB-1:0] v0 = votos[0 +: NB], v1 = votos[NB +: NB];
  wire signed [NB+1:0] margen = $signed({2'b00, v1}) - $signed({2'b00, v0});
  wire alarma_cruda = margen > $signed(umbral);
  // direccion: eje dominante de la gravedad proyectada y su signo (gy corregido por el sesgo de postura del pato)
  wire signed [8:0] dx = $signed(f0[31:24]), dy = $signed(f0[39:32]) - $signed(`SESGO_Y);
  wire [8:0] adx = dx[8] ? -dx : dx, ady = dy[8] ? -dy : dy;
  wire [1:0] dir = (adx >= ady) ? (dx > 0 ? 2'd0 : 2'd1) : (dy > 0 ? 2'd2 : 2'd3);

  // ---------- linea base clasica (para comparar en la misma placa) ----------
  wire signed [7:0] gx = f0[31:24], gy = f0[39:32];
  wire [7:0] ax = gx[7] ? -gx : gx, ay = gy[7] ? -gy : gy;
  wire base_cruda = ((ax > ay) ? ax : ay) > umbral_base;

  // ---------- salidas registradas, una vez por cuadro ----------
  reg [2:0] cuenta; reg alarma, base, latido; reg [1:0] dir_r; reg [2:0] marg_r; reg evaluar;
  wire [2:0] k_ef = (K == 3'd0) ? 3'd1 : K;
  always @(posedge clk) begin
    evaluar <= listo;                                  // un ciclo despues de guardar el cuadro
    if (!rst_n) begin
      cuenta <= 0; alarma <= 0; base <= 0; latido <= 0; dir_r <= 0; marg_r <= 0; evaluar <= 0;
    end else if (evaluar) begin
      cuenta <= alarma_cruda ? ((cuenta == 3'd7) ? cuenta : cuenta + 3'd1) : 3'd0;
      alarma <= alarma_cruda && (({1'b0, cuenta} + 4'd1) >= {1'b0, k_ef});
      base <= base_cruda; latido <= ~latido; dir_r <= dir;
      marg_r <= margen[NB+1] ? 3'd0 : (margen > 7 ? 3'd7 : margen[2:0]);
    end
  end
  wire [NB-1:0] vsel = sel[0] ? v1 : v0;
  wire [3:0] vdep = sel[1] ? vsel[NB-1 -: 4] : vsel[3:0];   // sel[1] = nibble alto
  assign uo_out = {marg_r, base, latido, dir_r, alarma};
  assign uio_out = {vdep, 4'b0000};
  assign uio_oe = 8'b1111_0000;
  wire _sin_uso = &{ena, uio_in[7:3], 1'b0};
endmodule
