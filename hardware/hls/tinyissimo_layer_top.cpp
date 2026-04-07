// tinyissimo_layer_top.cpp
//
// Synthesis entry point for Vitis HLS.
//
// Walks all NUM_LAYERS layers in a single ap_start invocation, reading the
// per-layer parameters from the compile-time LAYER_CFG[] array in
// layer_config.h.  Both ping-pong feature-map URAMs (fmap_a, fmap_b) are
// exposed as separate ap_memory interfaces; the layer loop branches on
// LAYER_CFG[L].pp_buf_sel and calls the inner kernel with the in/out
// arrays swapped accordingly — no top-level ping-pong mux is needed.
//
// curr_layer_out is an ap_vld scalar output that the wrapper module
// (inference_hls.sv) latches into a 5-bit status register read back via
// AXI-Lite.
//
// syn.top = tinyissimo_layer_top
#include "tinyissimo_layer.h"

void tinyissimo_layer_top(
    ap_uint<128>       fmap_a   [FMAP_DEPTH],
    ap_uint<128>       fmap_b   [FMAP_DEPTH],
    const ap_uint<128> wt_mem   [WT_DEPTH],
    const ap_uint<128> qp_mem   [QP_DEPTH],
    const ap_uint<8>   silu_mem [SILU_LUT_DEPTH],
    volatile int      *curr_layer_out
)
{
    // ── Block-level handshake: bare ap_start/ap_done/ap_idle/ap_ready ──
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return

    // ── Memory interfaces ───────────────────────────────────────────────
#pragma HLS INTERFACE mode=bram port=fmap_a
#pragma HLS INTERFACE mode=bram port=fmap_b
#pragma HLS INTERFACE mode=bram port=wt_mem
#pragma HLS INTERFACE mode=bram port=qp_mem
#pragma HLS INTERFACE mode=bram port=silu_mem

    // ── Status sideband ────────────────────────────────────────────────
#pragma HLS INTERFACE mode=ap_vld port=curr_layer_out

    LAYER_LOOP:
    for (int L = 0; L < NUM_LAYERS; L++) {
#pragma HLS LOOP_TRIPCOUNT min=NUM_LAYERS max=NUM_LAYERS

        // Publish current layer index for AXI status readback.
        *curr_layer_out = L;

        const LayerCfg cfg = LAYER_CFG[L];

        const bool packed_rgb = (L == 0);  // model layer 0 only

        if (cfg.pp_buf_sel == 0) {
            // fmap_a is the OUTPUT buffer, fmap_b is the INPUT buffer.
            tinyissimo_layer(
                cfg.h_in, cfg.w_in, cfg.cin, cfg.cout,
                cfg.kh,   cfg.kw,   cfg.pad_h, cfg.pad_w,
                /*use_maxpool=*/ (bool)cfg.use_maxpool,
                /*use_silu   =*/ (bool)cfg.use_silu,
                /*layer_idx  =*/ L,
                /*zp_in      =*/ (ap_int<8>)cfg.zp_in,
                /*zp_out     =*/ (ap_int<8>)cfg.zp_out,
                /*wt_base    =*/ (int)cfg.wt_base,
                /*qp_base    =*/ (int)cfg.qp_base,
                /*fmap_rd_off=*/ (int)cfg.pp_rd_offset,
                /*fmap_wr_off=*/ (int)cfg.pp_wr_offset,
                /*packed_rgb=*/ packed_rgb,
                /*fmap_in    =*/ fmap_b,
                /*fmap_out   =*/ fmap_a,
                wt_mem, qp_mem, silu_mem);
        } else {
            // fmap_b is the OUTPUT buffer, fmap_a is the INPUT buffer.
            tinyissimo_layer(
                cfg.h_in, cfg.w_in, cfg.cin, cfg.cout,
                cfg.kh,   cfg.kw,   cfg.pad_h, cfg.pad_w,
                /*use_maxpool=*/ (bool)cfg.use_maxpool,
                /*use_silu   =*/ (bool)cfg.use_silu,
                /*layer_idx  =*/ L,
                /*zp_in      =*/ (ap_int<8>)cfg.zp_in,
                /*zp_out     =*/ (ap_int<8>)cfg.zp_out,
                /*wt_base    =*/ (int)cfg.wt_base,
                /*qp_base    =*/ (int)cfg.qp_base,
                /*fmap_rd_off=*/ (int)cfg.pp_rd_offset,
                /*fmap_wr_off=*/ (int)cfg.pp_wr_offset,
                /*packed_rgb=*/ packed_rgb,
                /*fmap_in    =*/ fmap_a,
                /*fmap_out   =*/ fmap_b,
                wt_mem, qp_mem, silu_mem);
        }
    }

    // Sentinel: layer == NUM_LAYERS signals "done" to the status reader.
    *curr_layer_out = NUM_LAYERS;
}
