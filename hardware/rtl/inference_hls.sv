`timescale 1ns / 1ps
`include "layer_config.svh"

/* ============================================================================
 *  inference_hls — placeholder for the HLS-generated inference engine.
 *
 *  Mirrors the port list of inference_hdl exactly so the two are drop-in
 *  swappable inside inference_top via the USE_HLS generate selector.
 *
 *  All outputs are tied to safe defaults; the engine never asserts done.
 *  Replace the body with the Vitis HLS-generated RTL when available.
 * ============================================================================ */
module inference_hls #(
    parameter MAX_PARALLEL = C_PARALLEL,
    parameter K            = 3,
    parameter N_BITS       = 8,
    parameter ACC_BITS     = 32,
    parameter DEPTH_BITS   = 16,
    parameter KSQ          = K * K,
    parameter WT_DATA_W    = MAX_PARALLEL * N_BITS,
    parameter WT_ADDR_W    = $clog2(32768),
    parameter QP_ADDR_W    = $clog2(1024),
    parameter LUT_ADDR_W   = $clog2(ACT_LUT_DEPTH),
    parameter FMAP_DATA_W  = MAX_PARALLEL * N_BITS,
    parameter FMAP_ADDR_W  = $clog2(16384)
)(
    input  logic                             aclk,
    input  logic                             aresetn,
    input  logic                             start,
    output logic                             done,

    /* Weight ROM read port */
    output logic                             wt_mem_en_b,
    output logic [WT_ADDR_W-1:0]             wt_mem_addr_b,
    input  logic [WT_DATA_W-1:0]             wt_mem_dout_b,

    /* QP ROM read port */
    output logic                             qp_mem_en_b,
    output logic [QP_ADDR_W-1:0]             qp_mem_addr_b,
    input  logic [71:0]                      qp_mem_dout_b,

    /* SiLU LUT ROM read port */
    output logic                             silu_mem_en_b,
    output logic [LUT_ADDR_W-1:0]            silu_mem_addr_b,
    input  logic [N_BITS-1:0]                silu_mem_dout_b,

    /* Input feature-map read port */
    output logic [DEPTH_BITS-1:0]            in_buf_rd_addr,
    output logic                             in_buf_rd_en,
    input  logic [MAX_PARALLEL*N_BITS-1:0]   in_buf_rd_data,

    /* Output feature-map RMW read-back port */
    output logic                             out_buf_rd_en,
    output logic [FMAP_ADDR_W-1:0]           out_buf_rd_addr,
    input  logic [FMAP_DATA_W-1:0]           out_buf_rd_data,

    /* Output feature-map RMW write port */
    output logic                             out_buf_wr_en,
    output logic [FMAP_ADDR_W-1:0]           out_buf_wr_addr,
    output logic [FMAP_DATA_W-1:0]           out_buf_wr_data,

    /* Status */
    output logic [4:0]                       curr_layer_idx,

    /* Sub-pingpong config */
    output logic                             curr_pp_buf_sel,
    output logic [13:0]                      curr_pp_rd_offset
);

    /* Tie all outputs to safe defaults — engine is inert until the
       Vitis HLS-generated body replaces this stub. */
    assign done              = 1'b0;

    assign wt_mem_en_b       = 1'b0;
    assign wt_mem_addr_b     = '0;
    assign qp_mem_en_b       = 1'b0;
    assign qp_mem_addr_b     = '0;
    assign silu_mem_en_b     = 1'b0;
    assign silu_mem_addr_b   = '0;

    assign in_buf_rd_en      = 1'b0;
    assign in_buf_rd_addr    = '0;

    assign out_buf_rd_en     = 1'b0;
    assign out_buf_rd_addr   = '0;
    assign out_buf_wr_en     = 1'b0;
    assign out_buf_wr_addr   = '0;
    assign out_buf_wr_data   = '0;

    assign curr_layer_idx    = '0;
    assign curr_pp_buf_sel   = 1'b0;
    assign curr_pp_rd_offset = '0;

    /* Suppress unused-input warnings — these will be consumed by the
       real HLS body, but the stub doesn't read them. */
    wire _unused = &{1'b0,
        aclk, aresetn, start,
        wt_mem_dout_b, qp_mem_dout_b, silu_mem_dout_b,
        in_buf_rd_data, out_buf_rd_data,
        1'b0};

endmodule
