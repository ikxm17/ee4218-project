`timescale 1ns / 1ps
`include "layer_config.svh"

/* ============================================================================
 *  inference_hls — black-box wrapper around the Vitis-HLS-generated
 *  `tinyissimo_layer_top` module.
 *
 *  The HLS engine walks all NUM_LAYERS layers in a single ap_start
 *  invocation, reading per-layer parameters from a compile-time LAYER_CFG[]
 *  array (mirrored from layer_config.svh into hardware/hls/layer_config.h).
 *  It exposes:
 *    - ap_clk / ap_rst (active-HIGH synchronous reset)
 *    - ap_start / ap_done / ap_idle / ap_ready  block-level handshake
 *    - 5 BRAM-style memory ports
 *        fmap_a   — port A (RW) + port B (R-only)
 *        fmap_b   — port A (RW) + port B (R-only)
 *        wt_mem   — port A only (R-only)
 *        qp_mem   — port A only (R-only)
 *        silu_mem — port A only (R-only)
 *    - curr_layer_out + curr_layer_out_ap_vld scalar status output
 *
 *  This wrapper:
 *    1. Inverts aresetn -> ap_rst (HLS uses active-high reset).
 *    2. Converts the level-high `start` input into a 1-cycle ap_start pulse.
 *    3. Latches ap_done into a level-high `done` until the next start.
 *    4. Latches curr_layer_out on ap_vld pulses into curr_layer_idx[4:0].
 *    5. Translates the HLS BRAM-style memory ports onto the SystemVerilog
 *       sdp_ram interface used by inference_top:
 *         - sdp_ram has SIMPLE dual-port: port A is write-only, port B is
 *           read-only.  HLS port A is RW (read-AND-write on the same wire
 *           in different cycles, distinguished by WEN_A).  We split it:
 *             writes  (EN_A &  |WEN_A) -> sdp_ram port A
 *             reads   (EN_A & ~|WEN_A) -> sdp_ram port B
 *         - HLS also exposes a separate read-only port B per fmap; we OR
 *           it onto the same sdp_ram port B (merging the two HLS read
 *           sources).  At any time only one of {HLS port A, HLS port B}
 *           reads — they belong to the unrolled if/else branches and only
 *           one branch runs per layer — so the merge is conflict-free.
 *    6. Strips the byte-aligned shift HLS applies to BRAM addresses
 *       (Addr_A is byte-addressed; we shift right by 4 for 128-bit words
 *       and pass through directly for 8-bit silu_mem).
 *    7. Zero-extends the 72-bit qp_mem URAM dout to the 128-bit width
 *       HLS pads to (the upper 56 bits are unused by the C++ code).
 * ============================================================================ */
module inference_hls #(
    parameter FMAP_DATA_W = 128,
    parameter FMAP_ADDR_W = 14,
    parameter WT_DATA_W   = 128,
    parameter WT_ADDR_W   = 15,
    parameter QP_DATA_W   = 72,
    parameter QP_ADDR_W   = 10,
    parameter SILU_ADDR_W = 13   // $clog2(ACT_LUT_DEPTH=4352)
)(
    input  logic                   aclk,
    input  logic                   aresetn,
    input  logic                   start,
    output logic                   done,

    /* fmap_a — both physical ports of u_fmap_a */
    output logic                   fmap_a_en_a,
    output logic                   fmap_a_we_a,
    output logic [FMAP_ADDR_W-1:0] fmap_a_addr_a,
    output logic [FMAP_DATA_W-1:0] fmap_a_din_a,
    output logic                   fmap_a_en_b,
    output logic [FMAP_ADDR_W-1:0] fmap_a_addr_b,
    input  logic [FMAP_DATA_W-1:0] fmap_a_dout_b,

    /* fmap_b — both physical ports of u_fmap_b */
    output logic                   fmap_b_en_a,
    output logic                   fmap_b_we_a,
    output logic [FMAP_ADDR_W-1:0] fmap_b_addr_a,
    output logic [FMAP_DATA_W-1:0] fmap_b_din_a,
    output logic                   fmap_b_en_b,
    output logic [FMAP_ADDR_W-1:0] fmap_b_addr_b,
    input  logic [FMAP_DATA_W-1:0] fmap_b_dout_b,

    /* wt_mem read port (port B of u_wt_mem) */
    output logic                   wt_mem_en_b,
    output logic [WT_ADDR_W-1:0]   wt_mem_addr_b,
    input  logic [WT_DATA_W-1:0]   wt_mem_dout_b,

    /* qp_mem read port (72-bit URAM, zero-extended internally to 128b) */
    output logic                   qp_mem_en_b,
    output logic [QP_ADDR_W-1:0]   qp_mem_addr_b,
    input  logic [QP_DATA_W-1:0]   qp_mem_dout_b,

    /* silu_mem read port (8-bit BRAM) */
    output logic                   silu_mem_en_b,
    output logic [SILU_ADDR_W-1:0] silu_mem_addr_b,
    input  logic [7:0]             silu_mem_dout_b,

    /* Status sideband — current layer index for AXI status readback */
    output logic [4:0]             curr_layer_idx
);

    // ─────────────────────────────────────────────────────────────────
    //  Raw HLS top-level signals
    // ─────────────────────────────────────────────────────────────────
    logic        ap_rst;
    logic        ap_start, ap_done, ap_idle, ap_ready;

    // fmap_a HLS ports
    logic [31:0]  fa_addr_A_raw, fa_addr_B_raw;
    logic         fa_en_A_raw,   fa_en_B_raw;
    logic [15:0]  fa_wen_A_raw,  fa_wen_B_raw;
    logic [127:0] fa_din_A_raw,  fa_din_B_raw;
    logic [127:0] fa_dout_A_raw, fa_dout_B_raw;

    // fmap_b HLS ports
    logic [31:0]  fb_addr_A_raw, fb_addr_B_raw;
    logic         fb_en_A_raw,   fb_en_B_raw;
    logic [15:0]  fb_wen_A_raw,  fb_wen_B_raw;
    logic [127:0] fb_din_A_raw,  fb_din_B_raw;
    logic [127:0] fb_dout_A_raw, fb_dout_B_raw;

    // wt/qp/silu HLS ports (single port A, read-only — WEN/Din tied off
    // to 0 by HLS, so we drop them on the floor here)
    logic [31:0]  wt_addr_raw, qp_addr_raw, su_addr_raw;
    logic         wt_en_raw,   qp_en_raw,   su_en_raw;
    logic [127:0] wt_dout_raw, qp_dout_raw;
    logic [7:0]   su_dout_raw;

    // Scalar status output
    logic [31:0] curr_layer_raw;
    logic        curr_layer_vld;

    // ─────────────────────────────────────────────────────────────────
    //  Reset and start/done handshake
    // ─────────────────────────────────────────────────────────────────
    assign ap_rst = ~aresetn;

    // Level-high `start` -> 1-cycle ap_start pulse on rising edge
    logic start_r;
    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn) start_r <= 1'b0;
        else          start_r <= start;
    end
    assign ap_start = start & ~start_r;

    // Latch ap_done into level-high done; clear on next start
    logic done_l;
    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn)        done_l <= 1'b0;
        else if (ap_start)   done_l <= 1'b0;
        else if (ap_done)    done_l <= 1'b1;
    end
    assign done = done_l;

    // Latch curr_layer_out on each ap_vld pulse
    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn)            curr_layer_idx <= 5'd0;
        else if (ap_start)       curr_layer_idx <= 5'd0;
        else if (curr_layer_vld) curr_layer_idx <= curr_layer_raw[4:0];
    end

    // ─────────────────────────────────────────────────────────────────
    //  HLS black-box instance
    // ─────────────────────────────────────────────────────────────────
    tinyissimo_layer_top u_hls (
        .ap_clk         (aclk),
        .ap_rst         (ap_rst),
        .ap_start       (ap_start),
        .ap_done        (ap_done),
        .ap_idle        (ap_idle),
        .ap_ready       (ap_ready),

        // fmap_a port A (RW)
        .fmap_a_Addr_A  (fa_addr_A_raw),
        .fmap_a_EN_A    (fa_en_A_raw),
        .fmap_a_WEN_A   (fa_wen_A_raw),
        .fmap_a_Din_A   (fa_din_A_raw),
        .fmap_a_Dout_A  (fa_dout_A_raw),
        .fmap_a_Clk_A   (),
        .fmap_a_Rst_A   (),
        // fmap_a port B (R-only)
        .fmap_a_Addr_B  (fa_addr_B_raw),
        .fmap_a_EN_B    (fa_en_B_raw),
        .fmap_a_WEN_B   (fa_wen_B_raw),  // tied to 0 inside HLS top
        .fmap_a_Din_B   (fa_din_B_raw),  // tied to 0 inside HLS top
        .fmap_a_Dout_B  (fa_dout_B_raw),
        .fmap_a_Clk_B   (),
        .fmap_a_Rst_B   (),

        // fmap_b port A (RW)
        .fmap_b_Addr_A  (fb_addr_A_raw),
        .fmap_b_EN_A    (fb_en_A_raw),
        .fmap_b_WEN_A   (fb_wen_A_raw),
        .fmap_b_Din_A   (fb_din_A_raw),
        .fmap_b_Dout_A  (fb_dout_A_raw),
        .fmap_b_Clk_A   (),
        .fmap_b_Rst_A   (),
        // fmap_b port B (R-only)
        .fmap_b_Addr_B  (fb_addr_B_raw),
        .fmap_b_EN_B    (fb_en_B_raw),
        .fmap_b_WEN_B   (fb_wen_B_raw),  // tied to 0 inside HLS top
        .fmap_b_Din_B   (fb_din_B_raw),  // tied to 0 inside HLS top
        .fmap_b_Dout_B  (fb_dout_B_raw),
        .fmap_b_Clk_B   (),
        .fmap_b_Rst_B   (),

        // wt_mem (read-only)
        .wt_mem_Addr_A  (wt_addr_raw),
        .wt_mem_EN_A    (wt_en_raw),
        .wt_mem_WEN_A   (),
        .wt_mem_Din_A   (),
        .wt_mem_Dout_A  (wt_dout_raw),
        .wt_mem_Clk_A   (),
        .wt_mem_Rst_A   (),

        // qp_mem (read-only, 128-bit padded)
        .qp_mem_Addr_A  (qp_addr_raw),
        .qp_mem_EN_A    (qp_en_raw),
        .qp_mem_WEN_A   (),
        .qp_mem_Din_A   (),
        .qp_mem_Dout_A  (qp_dout_raw),
        .qp_mem_Clk_A   (),
        .qp_mem_Rst_A   (),

        // silu_mem (read-only)
        .silu_mem_Addr_A(su_addr_raw),
        .silu_mem_EN_A  (su_en_raw),
        .silu_mem_WEN_A (),
        .silu_mem_Din_A (),
        .silu_mem_Dout_A(su_dout_raw),
        .silu_mem_Clk_A (),
        .silu_mem_Rst_A (),

        // Status
        .curr_layer_out        (curr_layer_raw),
        .curr_layer_out_ap_vld (curr_layer_vld)
    );

    // ─────────────────────────────────────────────────────────────────
    //  Read-only ROM ports: address shift + data passthrough
    //  HLS Addr_A is byte-addressed; sdp_ram is word-addressed.
    //    wt_mem  : 128-bit (16 B/word) -> shift right by 4
    //    qp_mem  : 128-bit (16 B/word) -> shift right by 4
    //    silu_mem:   8-bit (1  B/word) -> no shift
    // ─────────────────────────────────────────────────────────────────
    assign wt_mem_addr_b   = wt_addr_raw[WT_ADDR_W+4-1:4];
    assign wt_mem_en_b     = wt_en_raw;
    assign wt_dout_raw     = wt_mem_dout_b;

    assign qp_mem_addr_b   = qp_addr_raw[QP_ADDR_W+4-1:4];
    assign qp_mem_en_b     = qp_en_raw;
    assign qp_dout_raw     = {56'd0, qp_mem_dout_b};   // 72 -> 128

    assign silu_mem_addr_b = su_addr_raw[SILU_ADDR_W-1:0];
    assign silu_mem_en_b   = su_en_raw;
    assign su_dout_raw     = silu_mem_dout_b;

    // ─────────────────────────────────────────────────────────────────
    //  fmap_a port mapping
    //
    //  HLS port A is RW (one wire group, distinguished by WEN_A).
    //  HLS port B is read-only (WEN_B/Din_B tied to 0 inside HLS top).
    //  sdp_ram port A is write-only, port B is read-only.
    //
    //  Map:
    //    HLS  port A write  -> sdp_ram port A
    //    HLS  port A read   -> sdp_ram port B  (when port B HLS is idle)
    //    HLS  port B read   -> sdp_ram port B  (when port A HLS is idle)
    //
    //  Because the two HLS ports belong to the unrolled CONV_LOOP /
    //  CONV_LOOP4 pipelines and only one branch executes per layer,
    //  port A reads and port B reads can never happen simultaneously.
    //  We OR-mux them, with port A taking priority when both are
    //  somehow asserted in the same cycle (they shouldn't be).
    // ─────────────────────────────────────────────────────────────────
    wire fa_pa_is_wr = fa_en_A_raw &  (|fa_wen_A_raw);
    wire fa_pa_is_rd = fa_en_A_raw & ~(|fa_wen_A_raw);

    assign fmap_a_en_a   = fa_pa_is_wr;
    assign fmap_a_we_a   = fa_pa_is_wr;
    assign fmap_a_addr_a = fa_addr_A_raw[FMAP_ADDR_W+4-1:4];
    assign fmap_a_din_a  = fa_din_A_raw;

    assign fmap_a_en_b   = fa_pa_is_rd | fa_en_B_raw;
    assign fmap_a_addr_b = fa_pa_is_rd ? fa_addr_A_raw[FMAP_ADDR_W+4-1:4]
                                       : fa_addr_B_raw[FMAP_ADDR_W+4-1:4];

    // sdp_ram port B output goes BACK to the HLS top.  Both HLS ports
    // see the same data — only the active port acts on it.
    assign fa_dout_A_raw = fmap_a_dout_b;
    assign fa_dout_B_raw = fmap_a_dout_b;

    // ─────────────────────────────────────────────────────────────────
    //  fmap_b port mapping (same pattern as fmap_a)
    // ─────────────────────────────────────────────────────────────────
    wire fb_pa_is_wr = fb_en_A_raw &  (|fb_wen_A_raw);
    wire fb_pa_is_rd = fb_en_A_raw & ~(|fb_wen_A_raw);

    assign fmap_b_en_a   = fb_pa_is_wr;
    assign fmap_b_we_a   = fb_pa_is_wr;
    assign fmap_b_addr_a = fb_addr_A_raw[FMAP_ADDR_W+4-1:4];
    assign fmap_b_din_a  = fb_din_A_raw;

    assign fmap_b_en_b   = fb_pa_is_rd | fb_en_B_raw;
    assign fmap_b_addr_b = fb_pa_is_rd ? fb_addr_A_raw[FMAP_ADDR_W+4-1:4]
                                       : fb_addr_B_raw[FMAP_ADDR_W+4-1:4];

    assign fb_dout_A_raw = fmap_b_dout_b;
    assign fb_dout_B_raw = fmap_b_dout_b;

endmodule
