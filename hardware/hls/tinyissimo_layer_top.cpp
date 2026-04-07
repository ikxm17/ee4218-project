// tinyissimo_layer_top.cpp
//
// Synthesis entry point for Vitis HLS.
// Places all interface pragmas and delegates to tinyissimo_layer().
//
// syn.top = tinyissimo_layer_top
#include "tinyissimo_layer.h"

void tinyissimo_layer_top(
    int in_h, int in_w, int in_c,
    int out_c,
    int kh, int kw,
    int pad_h, int pad_w,
    bool use_maxpool,
    bool use_silu,
    int layer_idx,
    ap_int<8> zp_in,
    ap_int<8> zp_out,
    int wt_base,
    int qp_base,
    const ap_uint<128> fmap_in  [FMAP_DEPTH],
    ap_uint<128>       fmap_out [FMAP_DEPTH],
    const ap_uint<128> wt_mem   [WT_DEPTH],
    const ap_uint<72>  qp_mem   [QP_DEPTH],
    const ap_uint<8>   silu_mem [SILU_LUT_DEPTH]
)
{
    // ── Memory interface pragmas ────────────────────────────────────────
#pragma HLS INTERFACE bram port=fmap_in
#pragma HLS INTERFACE bram port=fmap_out
#pragma HLS INTERFACE bram port=wt_mem
#pragma HLS INTERFACE bram port=qp_mem
#pragma HLS INTERFACE bram port=silu_mem

    // ── Scalar parameters via AXI-Lite ──────────────────────────────────
#pragma HLS INTERFACE s_axilite port=in_h       bundle=control
#pragma HLS INTERFACE s_axilite port=in_w       bundle=control
#pragma HLS INTERFACE s_axilite port=in_c       bundle=control
#pragma HLS INTERFACE s_axilite port=out_c      bundle=control
#pragma HLS INTERFACE s_axilite port=kh         bundle=control
#pragma HLS INTERFACE s_axilite port=kw         bundle=control
#pragma HLS INTERFACE s_axilite port=pad_h      bundle=control
#pragma HLS INTERFACE s_axilite port=pad_w      bundle=control
#pragma HLS INTERFACE s_axilite port=use_maxpool bundle=control
#pragma HLS INTERFACE s_axilite port=use_silu   bundle=control
#pragma HLS INTERFACE s_axilite port=layer_idx  bundle=control
#pragma HLS INTERFACE s_axilite port=zp_in      bundle=control
#pragma HLS INTERFACE s_axilite port=zp_out     bundle=control
#pragma HLS INTERFACE s_axilite port=wt_base    bundle=control
#pragma HLS INTERFACE s_axilite port=qp_base    bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // ── Delegate to core compute function (inlined) ─────────────────────
    tinyissimo_layer(
        in_h, in_w, in_c, out_c,
        kh, kw, pad_h, pad_w,
        use_maxpool, use_silu, layer_idx,
        zp_in, zp_out, wt_base, qp_base,
        fmap_in, fmap_out, wt_mem, qp_mem, silu_mem
    );
}
