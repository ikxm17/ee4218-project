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
 *        fmap_a   — port A (R-only) + port B (W-only)   (after ram_s2p)
 *        fmap_b   — port A (R-only) + port B (W-only)   (after ram_s2p)
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
 *           read-only.
 *         - After the `storage_type=ram_s2p` pragma on fmap_a/fmap_b in
 *           tinyissimo_layer_top.cpp, the HLS top exposes each fmap as
 *           clean SDP at the top-level interface:
 *               HLS port A is READ-ONLY  (WEN_A hardwired to 0 in HLS top)
 *               HLS port B is WRITE-ONLY (Din_B / WEN_B carry the writes)
 *           Map:
 *               HLS port A read   -> sdp_ram port B (read)
 *               HLS port B write  -> sdp_ram port A (write)
 *         - The HLS top muxes the per-layer kernel's single read port and
 *           single write port onto the appropriate top fmap based on the
 *           ping-pong parity register (cfg_pp_buf_sel_reg_*).  For any
 *           given fmap, the HLS top NEVER asserts port A (read) and port B
 *           (write) in the same cycle, because the kernel reads from one
 *           buffer and writes to the OTHER each layer.  The two SDP ports
 *           are therefore conflict-free across the entire 17-layer ping-
 *           pong walk.
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

    // Latch ap_done into level-high done; clear on next start.
    //
    // Combinationally mask `done` with `~ap_start` to break a one-cycle
    // race with inference_top.sv's PH_RUN FSM check:
    //
    //   - For test N+1, the wrapper enters PH_RUN with done_l still high
    //     from test N (the registered clear-on-ap_start hasn't fired yet
    //     at the moment the FSM evaluates phase_next combinationally).
    //   - inference_top sees inference_done == 1 in cycle 0 of PH_RUN
    //     and immediately transitions to PH_DONE, skipping the actual
    //     inference (cycle count = 0).
    //   - Masking with ~ap_start forces `done` to 0 during the cycle
    //     when ap_start fires, so the FSM sees done == 0 and stays in
    //     PH_RUN.  done_l clears at the next rising edge and stays at 0
    //     until the HLS top asserts ap_done at the end of the new run.
    //
    // The HDL engine doesn't need this mask because its `done` is a
    // single-cycle combinational pulse with `done <= 1'b0` as the default
    // assignment (inference_hdl.sv:503, 526) — no latched state.
    logic done_l;
    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn)        done_l <= 1'b0;
        else if (ap_start)   done_l <= 1'b0;
        else if (ap_done)    done_l <= 1'b1;
    end
    assign done = done_l & ~ap_start;

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

        // fmap_a port A (R-only after ram_s2p; WEN_A / Din_A unused)
        .fmap_a_Addr_A  (fa_addr_A_raw),
        .fmap_a_EN_A    (fa_en_A_raw),
        .fmap_a_WEN_A   (fa_wen_A_raw),
        .fmap_a_Din_A   (fa_din_A_raw),
        .fmap_a_Dout_A  (fa_dout_A_raw),
        .fmap_a_Clk_A   (),
        .fmap_a_Rst_A   (),
        // fmap_a port B (W-only after ram_s2p; Dout_B unused)
        .fmap_a_Addr_B  (fa_addr_B_raw),
        .fmap_a_EN_B    (fa_en_B_raw),
        .fmap_a_WEN_B   (fa_wen_B_raw),
        .fmap_a_Din_B   (fa_din_B_raw),
        .fmap_a_Dout_B  (fa_dout_B_raw),  // tied to 0 in wrapper
        .fmap_a_Clk_B   (),
        .fmap_a_Rst_B   (),

        // fmap_b port A (R-only after ram_s2p; WEN_A / Din_A unused)
        .fmap_b_Addr_A  (fb_addr_A_raw),
        .fmap_b_EN_A    (fb_en_A_raw),
        .fmap_b_WEN_A   (fb_wen_A_raw),
        .fmap_b_Din_A   (fb_din_A_raw),
        .fmap_b_Dout_A  (fb_dout_A_raw),
        .fmap_b_Clk_A   (),
        .fmap_b_Rst_A   (),
        // fmap_b port B (W-only after ram_s2p; Dout_B unused)
        .fmap_b_Addr_B  (fb_addr_B_raw),
        .fmap_b_EN_B    (fb_en_B_raw),
        .fmap_b_WEN_B   (fb_wen_B_raw),
        .fmap_b_Din_B   (fb_din_B_raw),
        .fmap_b_Dout_B  (fb_dout_B_raw),  // tied to 0 in wrapper
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
    //  fmap_a / fmap_b port mapping  (post-`storage_type=ram_s2p`)
    //
    //  After the ram_s2p pragma in tinyissimo_layer_top.cpp, the HLS
    //  top exposes each fmap as clean SDP:
    //      HLS port A is READ-ONLY   (WEN_A hardwired to 0 in HLS top,
    //                                 Din_A unused)
    //      HLS port B is WRITE-ONLY  (Din_B / WEN_B carry the writes;
    //                                 Dout_B never consumed by HLS)
    //  sdp_ram port A is write-only, port B is read-only.
    //
    //  Map:
    //      HLS port A read   -> sdp_ram port B (read)
    //      HLS port B write  -> sdp_ram port A (write)
    //
    //  Ping-pong correctness across the 17-layer walk:
    //  ------------------------------------------------------------------
    //  The HLS top muxes the per-layer kernel's single read port and
    //  single write port onto the appropriate top fmap based on the
    //  parity register `cfg_pp_buf_sel_reg_*`.  For any given fmap,
    //  port A (read) and port B (write) are NEVER asserted in the same
    //  cycle, because the kernel reads from one buffer and writes to
    //  the OTHER each layer:
    //
    //    Layer N (parity=1): reads top fmap_a, writes top fmap_b
    //      -> HLS asserts fa_en_A and fb_en_B / fb_din_B / fb_wen_B
    //      -> wrapper drives u_fmap_a.en_b (read) and
    //                        u_fmap_b.en_a + we_a + din_a (write)
    //
    //    Layer N+1 (parity=0): reads top fmap_b, writes top fmap_a
    //      -> HLS asserts fb_en_A and fa_en_B / fa_din_B / fa_wen_B
    //      -> wrapper drives u_fmap_b.en_b (read) and
    //                        u_fmap_a.en_a + we_a + din_a (write)
    //
    //  The two SDP ports of each URAM are therefore conflict-free.
    //
    //  WEN_B from HLS is 16-bit (one byte enable per byte of the 128-bit
    //  word), but the HLS C++ writes whole `ap_uint<128>` words at the
    //  OUT_COL_STORE pipeline, so all 16 bits assert together.  We OR-
    //  reduce to the 1-bit sdp_ram.we_a.  If a future cpp change emits
    //  partial-word writes, this OR-reduction would silently widen them
    //  to full-word writes — flag and revisit.
    //
    //  Layer 0 RGB unpack: handled end-to-end inside the HLS C++ (see
    //  tinyissimo_layer.cpp `packed_rgb_input` branch — it computes its
    //  own `fmap_in[read_addr]` word index and extracts the lane in
    //  software).  The wrapper does NOT apply a second `>> 2` or
    //  substitute a zero-padded word.  Layer 0 reads the camera frame
    //  from u_fmap_b via the standard read path; the AXI preload that
    //  populated u_fmap_b earlier is masked out by the preload_active
    //  mux in inference_top.sv before the HLS engine starts.
    // ─────────────────────────────────────────────────────────────────

    // fmap_a — HLS port A (read) -> SDP port B; HLS port B (write) -> SDP port A
    assign fmap_a_en_a   = fa_en_B_raw;
    assign fmap_a_we_a   = |fa_wen_B_raw;
    assign fmap_a_addr_a = fa_addr_B_raw[FMAP_ADDR_W+4-1:4];
    assign fmap_a_din_a  = fa_din_B_raw;

    assign fmap_a_en_b   = fa_en_A_raw;
    assign fmap_a_addr_b = fa_addr_A_raw[FMAP_ADDR_W+4-1:4];

    // SDP read data returns through HLS port A's Dout.  HLS port B is
    // write-only and never reads, so its Dout input is tied off.
    assign fa_dout_A_raw = fmap_a_dout_b;
    assign fa_dout_B_raw = 128'd0;

    // fmap_b — same SDP mapping as fmap_a.
    assign fmap_b_en_a   = fb_en_B_raw;
    assign fmap_b_we_a   = |fb_wen_B_raw;
    assign fmap_b_addr_a = fb_addr_B_raw[FMAP_ADDR_W+4-1:4];
    assign fmap_b_din_a  = fb_din_B_raw;

    assign fmap_b_en_b   = fb_en_A_raw;
    assign fmap_b_addr_b = fb_addr_A_raw[FMAP_ADDR_W+4-1:4];

    assign fb_dout_A_raw = fmap_b_dout_b;
    assign fb_dout_B_raw = 128'd0;

endmodule
