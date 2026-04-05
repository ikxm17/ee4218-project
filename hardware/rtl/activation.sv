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
     *  Stage 0 → 1 Pipeline Registers
     * ================================================================ */
    logic                        r_valid;
    logic [DEPTH_BITS-1:0]       r_addr;
    logic signed [N_BITS-1:0]    r_data;
    logic                        r_bypass;

    /* ================================================================
     *  Stage 0: LUT Address Drive (combinational)
     *
     *  Address = {layer_idx, unsigned_offset}
     *  where unsigned_offset = in_data + 128 = {~in_data[7], in_data[6:0]}
     * ================================================================ */
    assign lut_en   = in_valid & ~bypass;
    assign lut_addr = {layer_idx, ~in_data[N_BITS-1], in_data[N_BITS-2:0]};

    /* ================================================================
     *  Stage 0 → 1 Register
     * ================================================================ */
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_valid  <= 1'b0;
            r_addr   <= '0;
            r_data   <= '0;
            r_bypass <= 1'b0;
        end else begin
            r_valid  <= in_valid;
            r_addr   <= in_addr;
            r_data   <= in_data;
            r_bypass <= bypass;
        end
    end

    /* ================================================================
     *  Stage 1: Output Mux
     *
     *  Bypass: pass registered conv output unchanged.
     *  Normal: use LUT read data (reinterpret unsigned as signed).
     * ================================================================ */
    assign out_valid = r_valid;
    assign out_addr  = r_addr;
    assign out_data  = r_bypass ? r_data : $signed(lut_rdata);

endmodule
