// tinyissimo_layer.h
//
// Runtime-parameterized HLS convolution layer for Tinyissimo CNN.
// Parallelizes TILE_IC=16 input channels and TILE_OC=8 output channels
// (128 INT8 MACs per cycle).
//
// Targets Kria KV260 (xck26-sfvc784-2LV-c) at 100 MHz.
// Memory interfaces match the RTL sdp_ram instances in top.sv.
#pragma once
#include <ap_int.h>
#include "layer_config.h"

// ── Tile parallelism (TILE_IC=16 for IC, TILE_OC=16 for OC) ─────────────
// TILE_OC=16 ⇒ 16 × 8-bit = 128-bit, exactly one fmap word, no half-RMW.
static const int TILE_IC = 16;
static const int TILE_OC = 16;

// ── Compile-time maximum dimensions ─────────────────────────────────────
static const int MAX_H        = 256;
static const int MAX_W        = 256;
static const int MAX_IC_TILES = 8;    // max IN_C=128 -> 128/16
static const int MAX_OC_TILES = 16;   // max OUT_C=128 -> 128/8
static const int MAX_KH       = 3;
static const int MAX_KW       = 3;
static const int MAX_KSQ      = MAX_KH * MAX_KW;

// ── Memory depths (matching RTL sdp_ram instances in top.sv) ────────────
static const int FMAP_DEPTH     = 16384;  // 128-bit x 16384  URAM
static const int WT_DEPTH       = 32768;  // 128-bit x 32768  URAM
static const int QP_DEPTH       = 1024;   // 72-bit  x 1024   BRAM
static const int SILU_LUT_DEPTH = 4352;   // 8-bit   x 4352   distributed
static const int SILU_SLICE     = 256;    // entries per layer in SiLU LUT

// ── Core compute function ───────────────────────────────────────────────
//
// Processes ONE layer.  fmap_in and fmap_out are non-overlapping URAM
// regions; ping-pong selection happens in the caller (tinyissimo_layer_top).
//
// qp_mem is declared as ap_uint<128> because Vitis HLS pads odd-width BRAM
// interfaces to a power-of-two byte width.  The packed bias/m0/n_shift
// fields still live in the low 70 bits — see lines 123-126 in the .cpp.
void tinyissimo_layer(
    // Runtime layer configuration
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
    // Ping-pong buffer offsets (word-level, into FMAP_DEPTH arrays)
    // Enables detection-head branches: layers 11-13 and 14-16 both
    // read from layer 10's output at different URAM regions.
    int fmap_rd_offset,
    int fmap_wr_offset,
    // When true, fmap_in is interpreted as 4 packed RGB pixels per
    // 128-bit word (matching the HDL camera preload format).  Used
    // ONLY by model layer 0; all other layers expect the standard
    // 16-channel-packed layout (one word per (h,w) position).
    bool packed_rgb_input,
    // External memories (ap_memory interfaces in synthesis)
    const ap_uint<128> fmap_in  [FMAP_DEPTH],
    ap_uint<128>       fmap_out [FMAP_DEPTH],
    const ap_uint<128> wt_mem   [WT_DEPTH],
    const ap_uint<128> qp_mem   [QP_DEPTH],
    const ap_uint<8>   silu_mem [SILU_LUT_DEPTH]
);

// ── Synthesis top-level wrapper ─────────────────────────────────────────
//
// Walks all NUM_LAYERS layers in a single ap_start invocation, reading the
// per-layer parameters from the compile-time LAYER_CFG[] array in
// layer_config.h.  Both fmap_a and fmap_b are exposed as separate ap_memory
// interfaces; the layer loop branches on LAYER_CFG[L].pp_buf_sel and calls
// tinyissimo_layer() with the in/out arrays swapped accordingly.
//
// curr_layer_out is an ap_vld scalar output: the wrapper module latches
// each pulse into a 5-bit status register read back via AXI-Lite.
void tinyissimo_layer_top(
    ap_uint<128>       fmap_a   [FMAP_DEPTH],
    ap_uint<128>       fmap_b   [FMAP_DEPTH],
    const ap_uint<128> wt_mem   [WT_DEPTH],
    const ap_uint<128> qp_mem   [QP_DEPTH],
    const ap_uint<8>   silu_mem [SILU_LUT_DEPTH],
    volatile int      *curr_layer_out
);
