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

// ── Tile parallelism (matching RTL C_PAR=16 for IC, 8 for OC) ───────────
static const int TILE_IC = 16;
static const int TILE_OC = 8;

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
    // External memories (ap_memory interfaces in synthesis)
    const ap_uint<128> fmap_in  [FMAP_DEPTH],
    ap_uint<128>       fmap_out [FMAP_DEPTH],
    const ap_uint<128> wt_mem   [WT_DEPTH],
    const ap_uint<72>  qp_mem   [QP_DEPTH],
    const ap_uint<8>   silu_mem [SILU_LUT_DEPTH]
);

// ── Synthesis top-level wrapper ─────────────────────────────────────────
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
);
