`timescale 1ns / 1ps
`include "layer_config.svh"

module activation #(
    parameter N_BITS     = 8,
    parameter DEPTH_BITS = 16,
    parameter LUT_ADDR_W = 13
)(
    input  logic                        clk,
    input  logic                        rst_n,

    /* Layer config (active layer) */
    input  logic [1:0]                  layer_type,
    input  logic [4:0]                  layer_idx,

    /* Input from conv3d RES output */
    input  logic                        in_valid,
    input  logic [DEPTH_BITS-1:0]       in_addr,
    input  logic signed [N_BITS-1:0]    in_data,

    /* SiLU LUT read port (sdp_ram port B in top.sv) */
    output logic                        lut_en,
    output logic [LUT_ADDR_W-1:0]       lut_addr,
    input  logic [N_BITS-1:0]           lut_rdata,

    /* Output (to downstream: max_pool or feature map) */
    output logic                        out_valid,
    output logic [DEPTH_BITS-1:0]       out_addr,
    output logic signed [N_BITS-1:0]    out_data
);

    /* ================================================================
     *  Bypass Detection
     * ================================================================ */
    wire bypass = (layer_type == CONV1_LIN);

    /* ================================================================
     *  Stage 0 Pipeline Registers
     *
     *  Register lut_addr, lut_en, and companion metadata so the
     *  silu_mem BRAM address port is driven from a flop, not directly
     *  from the combinational conv engine output.  This breaks the
     *  10.5 ns / 28-logic-level cone that runs from u_acc_mem/DOUT
     *  through conv1d's requantization (add + DSP multiply + shift +
     *  saturate) and this module's lut_addr expression into
     *  u_silu_mem/ADDRBWRADDR[*].  Self-contained to this module —
     *  no edits needed in conv1d/conv3d.
     *
     *  Address = {layer_idx, unsigned_offset}
     *  where unsigned_offset = in_data + 128 = {~in_data[7], in_data[6:0]}
     * ================================================================ */
    logic                        r0_valid;
    logic [DEPTH_BITS-1:0]       r0_addr;
    logic signed [N_BITS-1:0]    r0_data;
    logic                        r0_bypass;
    logic [LUT_ADDR_W-1:0]       r0_lut_addr;
    logic                        r0_lut_en;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r0_valid    <= 1'b0;
            r0_addr     <= '0;
            r0_data     <= '0;
            r0_bypass   <= 1'b0;
            r0_lut_addr <= '0;
            r0_lut_en   <= 1'b0;
        end else begin
            r0_valid    <= in_valid;
            r0_addr     <= in_addr;
            r0_data     <= in_data;
            r0_bypass   <= bypass;
            r0_lut_addr <= {layer_idx, ~in_data[N_BITS-1], in_data[N_BITS-2:0]};
            r0_lut_en   <= in_valid & ~bypass;
        end
    end

    assign lut_en   = r0_lut_en;
    assign lut_addr = r0_lut_addr;

    /* ================================================================
     *  Stage 1 Pipeline Registers
     *
     *  Matches the silu_mem BRAM read latency so that lut_rdata arrives
     *  at the output mux in the same cycle as the companion metadata
     *  (r_data for bypass, r_bypass/r_valid/r_addr for control).
     * ================================================================ */
    logic                        r_valid;
    logic [DEPTH_BITS-1:0]       r_addr;
    logic signed [N_BITS-1:0]    r_data;
    logic                        r_bypass;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_valid  <= 1'b0;
            r_addr   <= '0;
            r_data   <= '0;
            r_bypass <= 1'b0;
        end else begin
            r_valid  <= r0_valid;
            r_addr   <= r0_addr;
            r_data   <= r0_data;
            r_bypass <= r0_bypass;
        end
    end

    /* ================================================================
     *  Output Mux
     *
     *  Bypass: pass registered conv output unchanged.
     *  Normal: use LUT read data (reinterpret unsigned as signed).
     * ================================================================ */
    assign out_valid = r_valid;
    assign out_addr  = r_addr;
    assign out_data  = r_bypass ? r_data : $signed(lut_rdata);

endmodule
