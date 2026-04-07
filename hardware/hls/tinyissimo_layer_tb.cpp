// tinyissimo_layer_tb.cpp
//
// C-simulation testbench for the runtime-parameterized tinyissimo_layer.
//
// Tests:
//   1. 4x4, IC=16->OC=16, 3x3 conv, no pool, SiLU
//   2. 4x4, IC=16->OC=16, 3x3 conv + MaxPool 2x2, SiLU
//   3. 4x4, IC=3->OC=8,   3x3 conv, no pool, SiLU   (non-multiple channels)
//   4. Chained: L1(4x4, 16->16) -> L2(4x4, 16->8), no pool
//   5. 8x8, IC=32->OC=24, 1x1 conv, no pool, SiLU    (CONV1)
//   6. 8x8, IC=24->OC=8,  1x1 conv, no pool, linear  (CONV1_LIN)
//   7. Branching offsets: two branches from shared fmap, split output
//
// Compile standalone (no HLS toolchain needed):
//   g++ -std=c++17 -I<HLS_INCLUDE_PATH> -O2 \
//       tinyissimo_layer_tb.cpp tinyissimo_layer.cpp -o tb && ./tb
//
// For Vitis 2025.2 C-simulation the same file acts as the testbench.
// ────────────��────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "ap_int.h"
#include "tinyissimo_layer.h"

// ════════════════════════════════════════��════════════════════════════════
// Utility helpers
// ══════════════════════════��══════════════════════════════════════════════

static int8_t clip8(int32_t x) {
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

// Truncating rescale — returns full int32, matches HLS rescale().
// The caller adds zp_out and clips ONCE (single-clip) to match the
// HDL conv3d.v requantization order exactly.
static int32_t ref_rescale(int32_t acc, uint32_t m0, uint8_t n_shift) {
    int64_t product = (int64_t)acc * (int64_t)m0;
    int64_t shifted = product >> n_shift;
    return (int32_t)shifted;
}

// ═════════════════════════════════════════════════════════════════════════
// Identity SiLU LUT (output == input for easy verification)
// Uses XOR-0x80 addressing to match RTL: {~val[7], val[6:0]}
// ═════════════════════════════════════════════════════════════════════════

static void build_identity_lut(int8_t lut[256]) {
    for (int i = 0; i < 256; i++) {
        // Index i maps to signed value (int8_t)(i ^ 0x80)
        // Identity: lut[i] should return that same value
        lut[i] = (int8_t)((uint8_t)i ^ 0x80);
    }
}

// ═══════════════════════════════════════════════════���═════════════════════
// Packing: flat int8 [h][w][c] -> tiled fmap layout
//   word_addr = ic_tile * h * w + row * w + col
//   each 128-bit word = 16 int8 channels
// ═���═══════════════════��═════════════════════════════��═════════════════════

static void pack_input(const int8_t* flat,
                       ap_uint<128> fmap[],
                       int h, int w, int c)
{
    int ic_tiles = (c + TILE_IC - 1) / TILE_IC;
    for (int ict = 0; ict < ic_tiles; ict++)
    for (int row = 0; row < h; row++)
    for (int col = 0; col < w; col++) {
        ap_uint<128> word = 0;
        for (int i = 0; i < TILE_IC; i++) {
            int ch = ict * TILE_IC + i;
            int8_t val = (ch < c) ? flat[(row * w + col) * c + ch] : 0;
            word.range(i*8+7, i*8) = (uint8_t)val;
        }
        fmap[ict * h * w + row * w + col] = word;
    }
}

// ═���══════════════════════════════════════���════════════════════════════════
// Packing: flat weights [oc][kh][kw][ic] -> weight ROM layout
//   addr = wt_base + oc * ic_tiles * ksq + ict * ksq + kh*kw_dim + kw
// ════════════════════════════════════════════════════════════��════════════

static void pack_weights_rom(const int8_t* flat,
                             ap_uint<128> wt_mem[],
                             int wt_base,
                             int out_c, int in_c,
                             int kh, int kw)
{
    int ic_tiles = (in_c + TILE_IC - 1) / TILE_IC;
    int ksq = kh * kw;
    for (int oc = 0; oc < out_c; oc++)
    for (int ict = 0; ict < ic_tiles; ict++)
    for (int kh_idx = 0; kh_idx < kh; kh_idx++)
    for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
        ap_uint<128> word = 0;
        for (int i = 0; i < TILE_IC; i++) {
            int ic = ict * TILE_IC + i;
            int8_t val = (ic < in_c)
                ? flat[((oc * kh + kh_idx) * kw + kw_idx) * in_c + ic]
                : 0;
            word.range(i*8+7, i*8) = (uint8_t)val;
        }
        int addr = wt_base + oc * ic_tiles * ksq
                   + ict * ksq + kh_idx * kw + kw_idx;
        wt_mem[addr] = word;
    }
}

// ═══��══════════════════════��══════════════════════════════════════════════
// Packing: per-channel QP into packed 72-bit ROM
//   [31:0]  = bias (signed)
//   [63:32] = m0 (unsigned)
//   [69:64] = n_shift
// ══���════════════════════════════════════════════════════════════════��═════

static void pack_qp_rom(const int32_t* bias,
                         const uint32_t* m0,
                         const uint8_t* n_shift,
                         ap_uint<128> qp_mem[],
                         int qp_base, int out_c)
{
    for (int oc = 0; oc < out_c; oc++) {
        ap_uint<128> word = 0;
        word.range(31,  0) = (uint32_t)bias[oc];
        word.range(63, 32) = m0[oc];
        word.range(69, 64) = n_shift[oc];
        qp_mem[qp_base + oc] = word;
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Fill SiLU memory slice for a given layer_idx
// ═══════════════���═════════════════════════════════════════════════════════

static void fill_silu_mem(const int8_t lut[256],
                           ap_uint<8> silu_mem[],
                           int layer_idx)
{
    int base = layer_idx * SILU_SLICE;
    for (int i = 0; i < SILU_SLICE; i++) {
        silu_mem[base + i] = (ap_uint<8>)((uint8_t)lut[i]);
    }
}

// ═════════════════════��═══════════════════════════════════════════════════
// Unpack: tiled fmap -> flat int8 [h][w][c]
// ════════════════════════════��════════════════════════════���═══════════════

static void unpack_output(const ap_uint<128> fmap[],
                           int8_t* flat,
                           int out_h, int out_w, int out_c,
                           bool use_maxpool,
                           int fmap_offset = 0)
{
    int oc_tiles = (out_c + TILE_OC - 1) / TILE_OC;
    int ph = use_maxpool ? out_h / 2 : out_h;
    int pw = use_maxpool ? out_w / 2 : out_w;
    int spatial = ph * pw;

    for (int oct = 0; oct < oc_tiles; oct++) {
        int pair   = oct / 2;
        int bit_lo = (oct % 2) * 64;
        for (int row = 0; row < ph; row++)
        for (int col = 0; col < pw; col++) {
            ap_uint<128> word = fmap[fmap_offset + pair * spatial + row * pw + col];
            for (int t = 0; t < TILE_OC; t++) {
                int ch = oct * TILE_OC + t;
                if (ch >= out_c) continue;
                uint8_t raw = (uint8_t)word.range(bit_lo + t*8+7,
                                                   bit_lo + t*8);
                flat[(row * pw + col) * out_c + ch] = (int8_t)raw;
            }
        }
    }
}

// ════════════════════���══════════════════════════════���═════════════════════
// Golden reference: conv + rescale + zp_out + LUT (per-channel QP)
// ════════════���══════════════════════════════��═════════════════════════════

static int8_t ref_pixel(const int8_t* ifmap,
                         const int8_t* weights,  // [oc][kh][kw][ic]
                         const int32_t* bias,
                         const uint32_t* m0,
                         const uint8_t* n_shift,
                         const int8_t* lut,       // [256]
                         int8_t zp_in, int8_t zp_out,
                         bool use_silu,
                         int in_h, int in_w, int in_c,
                         int kh, int kw,
                         int pad_h, int pad_w,
                         int oh, int ow, int oc)
{
    int32_t acc = bias[oc];
    for (int kh_idx = 0; kh_idx < kh; kh_idx++)
    for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
        int h = oh + kh_idx - pad_h;
        int w = ow + kw_idx - pad_w;
        for (int ic = 0; ic < in_c; ic++) {
            int8_t pix = (h < 0 || h >= in_h || w < 0 || w >= in_w)
                         ? zp_in
                         : ifmap[(h * in_w + w) * in_c + ic];
            int wt_idx = ((oc * kh + kh_idx) * kw + kw_idx) * in_c + ic;
            acc += (int32_t)pix * (int32_t)weights[wt_idx];
        }
    }
    int32_t shifted = ref_rescale(acc, m0[oc], n_shift[oc]);
    int32_t with_zp = shifted + (int32_t)zp_out;
    int8_t  clamped = clip8(with_zp);
    if (use_silu) {
        // XOR-0x80 addressing to match RTL
        uint8_t lut_idx = (uint8_t)clamped ^ 0x80;
        return lut[lut_idx];
    }
    return clamped;
}

// ═══════════════════════════════════════════════════���═════════════════════
// Simple pseudo-random generator (LCG, reproducible)
// ════════════��════════════════════════════════════════════════════════════

static uint32_t g_seed = 0xDEADBEEF;
static int8_t randi8() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (int8_t)(g_seed >> 24);
}

// ═════════════════════════════════════════════════════════════════════════
// Result checking
// ═════════════════════════════════════════════════════════════════════════

static int g_pass = 0, g_fail = 0;

static void check(const char* tag, int8_t got, int8_t ref,
                   int h, int w, int c)
{
    if (got != ref) {
        printf("  FAIL [%s] h=%d w=%d c=%d  got=%d  ref=%d\n",
               tag, h, w, c, (int)got, (int)ref);
        g_fail++;
    } else {
        g_pass++;
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Shared memory arrays (static to avoid stack overflow)
// ═══════════════════════════════════��═════════════════════════════════════

static ap_uint<128> g_fmap_in [FMAP_DEPTH];
static ap_uint<128> g_fmap_out[FMAP_DEPTH];
static ap_uint<128> g_wt_mem  [WT_DEPTH];
static ap_uint<128>  g_qp_mem  [QP_DEPTH];
static ap_uint<8>   g_silu_mem[SILU_LUT_DEPTH];

// ═══════════════════════════════════���═════════════════════════════════════
// Test 1: 4x4, IC=16 -> OC=16, 3x3, pad=1, no pool, SiLU
// ═══════���════════════════════════════════════════════════════��════════════

static void test1_no_pool() {
    printf("=== Test 1: 4x4 IC=16 OC=16 3x3 no-pool SiLU ===\n");

    const int IH=4, IW=4, IC=16, OC=16;
    const int KH=3, KW=3, PH=1, PW=1;
    const int OH=IH, OW=IW;
    const int WT_BASE=0, QP_BASE=0, LAYER_IDX=0;

    int8_t  ifmap  [IH*IW*IC];
    int8_t  weights[OC*KH*KW*IC];
    int32_t bias   [OC];
    uint32_t m0    [OC];
    uint8_t  nshift [OC];
    int8_t   lut   [256];

    g_seed = 1;
    for (int i = 0; i < IH*IW*IC; i++) ifmap[i] = randi8();
    for (int i = 0; i < OC*KH*KW*IC; i++) weights[i] = randi8();
    for (int i = 0; i < OC; i++) {
        bias[i]   = (int32_t)(randi8()) * 128;
        m0[i]     = 0x40000000u + (uint32_t)(randi8() & 0x1F) * 0x01000000u;
        nshift[i] = 30;
    }
    build_identity_lut(lut);

    int8_t zp_in = -10, zp_out = 5;

    // Pack into shared memories
    memset(g_fmap_out, 0, sizeof(g_fmap_out));
    pack_input(ifmap, g_fmap_in, IH, IW, IC);
    pack_weights_rom(weights, g_wt_mem, WT_BASE, OC, IC, KH, KW);
    pack_qp_rom(bias, m0, nshift, g_qp_mem, QP_BASE, OC);
    fill_silu_mem(lut, g_silu_mem, LAYER_IDX);

    // Run DUT
    tinyissimo_layer(
        IH, IW, IC, OC, KH, KW, PH, PW,
        false, true, LAYER_IDX,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE, QP_BASE, 0, 0,
        /*packed_rgb=*/ false,
        g_fmap_in, g_fmap_out, g_wt_mem, g_qp_mem, g_silu_mem);

    // Unpack and compare
    int8_t dut_out[OH*OW*OC];
    unpack_output(g_fmap_out, dut_out, OH, OW, OC, false);

    int errors = 0;
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
    for (int oc = 0; oc < OC; oc++) {
        int8_t ref = ref_pixel(ifmap, weights, bias, m0, nshift, lut,
                               zp_in, zp_out, true,
                               IH, IW, IC, KH, KW, PH, PW,
                               oh, ow, oc);
        int8_t got = dut_out[(oh*OW+ow)*OC + oc];
        check("T1", got, ref, oh, ow, oc);
        if (got != ref) errors++;
    }
    printf("  Errors: %d / %d\n\n", errors, OH*OW*OC);
}

// ═══��══════════���══════════════════════════════════════════════════════════
// Test 2: 4x4, IC=16 -> OC=16, 3x3 + MaxPool 2x2, SiLU
// ════���═════���══════════════════════════���═══════════════════════════════════

static void test2_maxpool() {
    printf("=== Test 2: 4x4 IC=16 OC=16 3x3 + MaxPool2x2 SiLU ===\n");

    const int IH=4, IW=4, IC=16, OC=16;
    const int KH=3, KW=3, PH=1, PW=1;
    const int OH=IH, OW=IW;
    const int PH2=OH/2, PW2=OW/2;
    const int WT_BASE=0, QP_BASE=0, LAYER_IDX=1;

    int8_t  ifmap  [IH*IW*IC];
    int8_t  weights[OC*KH*KW*IC];
    int32_t bias   [OC];
    uint32_t m0    [OC];
    uint8_t  nshift [OC];
    int8_t   lut   [256];

    g_seed = 2;
    for (int i = 0; i < IH*IW*IC; i++) ifmap[i] = randi8();
    for (int i = 0; i < OC*KH*KW*IC; i++) weights[i] = randi8();
    for (int i = 0; i < OC; i++) {
        bias[i]   = (int32_t)(randi8()) * 64;
        m0[i]     = 0x80000000u;
        nshift[i] = 31;
    }
    build_identity_lut(lut);

    int8_t zp_in = 0, zp_out = 0;

    memset(g_fmap_out, 0, sizeof(g_fmap_out));
    pack_input(ifmap, g_fmap_in, IH, IW, IC);
    pack_weights_rom(weights, g_wt_mem, WT_BASE, OC, IC, KH, KW);
    pack_qp_rom(bias, m0, nshift, g_qp_mem, QP_BASE, OC);
    fill_silu_mem(lut, g_silu_mem, LAYER_IDX);

    tinyissimo_layer(
        IH, IW, IC, OC, KH, KW, PH, PW,
        true, true, LAYER_IDX,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE, QP_BASE, 0, 0,
        /*packed_rgb=*/ false,
        g_fmap_in, g_fmap_out, g_wt_mem, g_qp_mem, g_silu_mem);

    int8_t dut_out[PH2*PW2*OC];
    unpack_output(g_fmap_out, dut_out, OH, OW, OC, true);

    // Reference: conv first, then 2x2 maxpool
    int8_t ref_conv[OH*OW*OC];
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
    for (int oc = 0; oc < OC; oc++)
        ref_conv[(oh*OW+ow)*OC+oc] =
            ref_pixel(ifmap, weights, bias, m0, nshift, lut,
                      zp_in, zp_out, true,
                      IH, IW, IC, KH, KW, PH, PW,
                      oh, ow, oc);

    int errors = 0;
    for (int ph = 0; ph < PH2; ph++)
    for (int pw = 0; pw < PW2; pw++)
    for (int oc = 0; oc < OC; oc++) {
        int8_t mx = ref_conv[((ph*2  )*OW + pw*2  )*OC+oc];
        int8_t v;
        v = ref_conv[((ph*2  )*OW + pw*2+1)*OC+oc]; if (v>mx) mx=v;
        v = ref_conv[((ph*2+1)*OW + pw*2  )*OC+oc]; if (v>mx) mx=v;
        v = ref_conv[((ph*2+1)*OW + pw*2+1)*OC+oc]; if (v>mx) mx=v;
        int8_t got = dut_out[(ph*PW2+pw)*OC+oc];
        check("T2", got, mx, ph, pw, oc);
        if (got != mx) errors++;
    }
    printf("  Errors: %d / %d\n\n", errors, PH2*PW2*OC);
}

// ═══════════════════════════════════════════════════════════��═════════════
// Test 3: 4x4, IC=3 -> OC=8, 3x3, no pool, SiLU (non-multiple channels)
// ═════���════════════════════════════════════════════════════════════════���══

static void test3_nonmultiple() {
    printf("=== Test 3: 4x4 IC=3 OC=8 3x3 no-pool SiLU ===\n");

    const int IH=4, IW=4, IC=3, OC=8;
    const int KH=3, KW=3, PH=1, PW=1;
    const int OH=IH, OW=IW;
    const int WT_BASE=0, QP_BASE=0, LAYER_IDX=2;

    int8_t  ifmap  [IH*IW*IC];
    int8_t  weights[OC*KH*KW*IC];
    int32_t bias   [OC];
    uint32_t m0    [OC];
    uint8_t  nshift [OC];
    int8_t   lut   [256];

    g_seed = 3;
    for (int i = 0; i < IH*IW*IC; i++) ifmap[i] = randi8();
    for (int i = 0; i < OC*KH*KW*IC; i++) weights[i] = randi8();
    for (int i = 0; i < OC; i++) {
        bias[i]   = (int32_t)(randi8()) * 32;
        m0[i]     = 0x60000000u;
        nshift[i] = 30;
    }
    build_identity_lut(lut);

    int8_t zp_in = 3, zp_out = -3;

    memset(g_fmap_out, 0, sizeof(g_fmap_out));
    pack_input(ifmap, g_fmap_in, IH, IW, IC);
    pack_weights_rom(weights, g_wt_mem, WT_BASE, OC, IC, KH, KW);
    pack_qp_rom(bias, m0, nshift, g_qp_mem, QP_BASE, OC);
    fill_silu_mem(lut, g_silu_mem, LAYER_IDX);

    tinyissimo_layer(
        IH, IW, IC, OC, KH, KW, PH, PW,
        false, true, LAYER_IDX,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE, QP_BASE, 0, 0,
        /*packed_rgb=*/ false,
        g_fmap_in, g_fmap_out, g_wt_mem, g_qp_mem, g_silu_mem);

    int8_t dut_out[OH*OW*OC];
    unpack_output(g_fmap_out, dut_out, OH, OW, OC, false);

    int errors = 0;
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
    for (int oc = 0; oc < OC; oc++) {
        int8_t ref = ref_pixel(ifmap, weights, bias, m0, nshift, lut,
                               zp_in, zp_out, true,
                               IH, IW, IC, KH, KW, PH, PW,
                               oh, ow, oc);
        int8_t got = dut_out[(oh*OW+ow)*OC + oc];
        check("T3", got, ref, oh, ow, oc);
        if (got != ref) errors++;
    }
    printf("  Errors: %d / %d\n\n", errors, OH*OW*OC);
}

// ═════════════════���════════════════════════════��══════════════════════════
// Test 4: Chained layers — L1(16->16) -> L2(16->8), no pool
// Verifies output layout compatibility with next layer's input
// ═════════════════════════════════════════════════════════════════════════

static void test4_chained() {
    printf("=== Test 4: 4x4 chained L1(16->16) -> L2(16->8) no-pool ===\n");

    const int IH=4, IW=4;
    const int L1_IC=16, L1_OC=16;
    const int L2_OC=8;
    const int KH=3, KW=3, PH=1, PW=1;
    const int WT_BASE_L1=0;
    const int WT_BASE_L2 = L1_OC * ((L1_IC+TILE_IC-1)/TILE_IC) * KH * KW;
    const int QP_BASE_L1=0, QP_BASE_L2=L1_OC;
    const int LI_1=3, LI_2=4;  // layer indices for SiLU

    int8_t   ifmap[IH*IW*L1_IC];
    int8_t   w1[L1_OC*KH*KW*L1_IC];
    int8_t   w2[L2_OC*KH*KW*L1_OC];
    int32_t  b1[L1_OC], b2[L2_OC];
    uint32_t m0_1[L1_OC], m0_2[L2_OC];
    uint8_t  ns1[L1_OC], ns2[L2_OC];
    int8_t   lut[256];

    g_seed = 4;
    for (int i = 0; i < IH*IW*L1_IC; i++) ifmap[i] = randi8();
    for (int i = 0; i < L1_OC*KH*KW*L1_IC; i++) w1[i] = randi8();
    for (int i = 0; i < L2_OC*KH*KW*L1_OC; i++) w2[i] = randi8();
    for (int i = 0; i < L1_OC; i++) {
        b1[i] = (int32_t)(randi8())*64;
        m0_1[i] = 0x80000000u;
        ns1[i] = 31;
    }
    for (int i = 0; i < L2_OC; i++) {
        b2[i] = (int32_t)(randi8())*64;
        m0_2[i] = 0x80000000u;
        ns2[i] = 31;
    }
    build_identity_lut(lut);

    int8_t zp_in = 0, zp_out = 0;

    // Pack into shared memories
    memset(g_fmap_in, 0, sizeof(g_fmap_in));
    memset(g_fmap_out, 0, sizeof(g_fmap_out));
    pack_input(ifmap, g_fmap_in, IH, IW, L1_IC);
    pack_weights_rom(w1, g_wt_mem, WT_BASE_L1, L1_OC, L1_IC, KH, KW);
    pack_weights_rom(w2, g_wt_mem, WT_BASE_L2, L2_OC, L1_OC, KH, KW);
    pack_qp_rom(b1, m0_1, ns1, g_qp_mem, QP_BASE_L1, L1_OC);
    pack_qp_rom(b2, m0_2, ns2, g_qp_mem, QP_BASE_L2, L2_OC);
    fill_silu_mem(lut, g_silu_mem, LI_1);
    fill_silu_mem(lut, g_silu_mem, LI_2);

    // Run L1: fmap_in -> fmap_out
    tinyissimo_layer(
        IH, IW, L1_IC, L1_OC, KH, KW, PH, PW,
        false, true, LI_1,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE_L1, QP_BASE_L1, 0, 0,
        /*packed_rgb=*/ false,
        g_fmap_in, g_fmap_out, g_wt_mem, g_qp_mem, g_silu_mem);

    // Run L2: fmap_out (from L1) -> fmap_in (reuse as output buffer)
    // This proves output layout is compatible with next layer's input
    static ap_uint<128> g_fmap_final[FMAP_DEPTH];
    memset(g_fmap_final, 0, sizeof(g_fmap_final));

    tinyissimo_layer(
        IH, IW, L1_OC, L2_OC, KH, KW, PH, PW,
        false, true, LI_2,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE_L2, QP_BASE_L2, 0, 0,
        /*packed_rgb=*/ false,
        g_fmap_out, g_fmap_final, g_wt_mem, g_qp_mem, g_silu_mem);

    // Reference: L1 then L2
    int8_t ref_mid[IH*IW*L1_OC];
    for (int oh=0; oh<IH; oh++)
    for (int ow=0; ow<IW; ow++)
    for (int oc=0; oc<L1_OC; oc++)
        ref_mid[(oh*IW+ow)*L1_OC+oc] =
            ref_pixel(ifmap, w1, b1, m0_1, ns1, lut,
                      zp_in, zp_out, true,
                      IH, IW, L1_IC, KH, KW, PH, PW,
                      oh, ow, oc);

    int8_t ref_out[IH*IW*L2_OC];
    for (int oh=0; oh<IH; oh++)
    for (int ow=0; ow<IW; ow++)
    for (int oc=0; oc<L2_OC; oc++)
        ref_out[(oh*IW+ow)*L2_OC+oc] =
            ref_pixel(ref_mid, w2, b2, m0_2, ns2, lut,
                      zp_in, zp_out, true,
                      IH, IW, L1_OC, KH, KW, PH, PW,
                      oh, ow, oc);

    int8_t dut_out[IH*IW*L2_OC];
    unpack_output(g_fmap_final, dut_out, IH, IW, L2_OC, false);

    int errors = 0;
    for (int oh=0; oh<IH; oh++)
    for (int ow=0; ow<IW; ow++)
    for (int oc=0; oc<L2_OC; oc++) {
        int8_t ref = ref_out[(oh*IW+ow)*L2_OC+oc];
        int8_t got = dut_out[(oh*IW+ow)*L2_OC+oc];
        check("T4", got, ref, oh, ow, oc);
        if (got != ref) errors++;
    }
    printf("  Errors: %d / %d\n\n", errors, IH*IW*L2_OC);
}

// ══════════���══════════════════════════════════════════════════════════════
// Test 5: 8x8, IC=32 -> OC=24, 1x1, no pool, SiLU (CONV1)
// ═════════════════��════════════════════════════════════��══════════════════

static void test5_conv1() {
    printf("=== Test 5: 8x8 IC=32 OC=24 1x1 no-pool SiLU (CONV1) ===\n");

    const int IH=8, IW=8, IC=32, OC=24;
    const int KH=1, KW=1, PH=0, PW=0;
    const int OH=IH, OW=IW;
    const int WT_BASE=0, QP_BASE=0, LAYER_IDX=5;

    int8_t  ifmap  [IH*IW*IC];
    int8_t  weights[OC*KH*KW*IC];
    int32_t bias   [OC];
    uint32_t m0    [OC];
    uint8_t  nshift [OC];
    int8_t   lut   [256];

    g_seed = 5;
    for (int i = 0; i < IH*IW*IC; i++) ifmap[i] = randi8();
    for (int i = 0; i < OC*KH*KW*IC; i++) weights[i] = randi8();
    for (int i = 0; i < OC; i++) {
        bias[i]   = (int32_t)(randi8()) * 64;
        m0[i]     = 0x50000000u + (uint32_t)(randi8() & 0x0F) * 0x01000000u;
        nshift[i] = 30;
    }
    build_identity_lut(lut);

    int8_t zp_in = -5, zp_out = 3;

    memset(g_fmap_out, 0, sizeof(g_fmap_out));
    pack_input(ifmap, g_fmap_in, IH, IW, IC);
    pack_weights_rom(weights, g_wt_mem, WT_BASE, OC, IC, KH, KW);
    pack_qp_rom(bias, m0, nshift, g_qp_mem, QP_BASE, OC);
    fill_silu_mem(lut, g_silu_mem, LAYER_IDX);

    tinyissimo_layer(
        IH, IW, IC, OC, KH, KW, PH, PW,
        false, true, LAYER_IDX,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE, QP_BASE, 0, 0,
        /*packed_rgb=*/ false,
        g_fmap_in, g_fmap_out, g_wt_mem, g_qp_mem, g_silu_mem);

    int8_t dut_out[OH*OW*OC];
    unpack_output(g_fmap_out, dut_out, OH, OW, OC, false);

    int errors = 0;
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
    for (int oc = 0; oc < OC; oc++) {
        int8_t ref = ref_pixel(ifmap, weights, bias, m0, nshift, lut,
                               zp_in, zp_out, true,
                               IH, IW, IC, KH, KW, PH, PW,
                               oh, ow, oc);
        int8_t got = dut_out[(oh*OW+ow)*OC + oc];
        check("T5", got, ref, oh, ow, oc);
        if (got != ref) errors++;
    }
    printf("  Errors: %d / %d\n\n", errors, OH*OW*OC);
}

// ═════════════════��════════════════════════════════���══════════════════════
// Test 6: 8x8, IC=24 -> OC=8, 1x1, no pool, linear (CONV1_LIN)
// ��════════════���═══════════════════════════════════════════════════════════

static void test6_conv1_lin() {
    printf("=== Test 6: 8x8 IC=24 OC=8 1x1 no-pool linear (CONV1_LIN) ===\n");

    const int IH=8, IW=8, IC=24, OC=8;
    const int KH=1, KW=1, PH=0, PW=0;
    const int OH=IH, OW=IW;
    const int WT_BASE=0, QP_BASE=0, LAYER_IDX=6;

    int8_t  ifmap  [IH*IW*IC];
    int8_t  weights[OC*KH*KW*IC];
    int32_t bias   [OC];
    uint32_t m0    [OC];
    uint8_t  nshift [OC];
    int8_t   lut   [256];  // unused but filled for completeness

    g_seed = 6;
    for (int i = 0; i < IH*IW*IC; i++) ifmap[i] = randi8();
    for (int i = 0; i < OC*KH*KW*IC; i++) weights[i] = randi8();
    for (int i = 0; i < OC; i++) {
        bias[i]   = (int32_t)(randi8()) * 32;
        m0[i]     = 0x70000000u;
        nshift[i] = 31;
    }
    build_identity_lut(lut);

    int8_t zp_in = -2, zp_out = 10;

    memset(g_fmap_out, 0, sizeof(g_fmap_out));
    pack_input(ifmap, g_fmap_in, IH, IW, IC);
    pack_weights_rom(weights, g_wt_mem, WT_BASE, OC, IC, KH, KW);
    pack_qp_rom(bias, m0, nshift, g_qp_mem, QP_BASE, OC);
    fill_silu_mem(lut, g_silu_mem, LAYER_IDX);

    // use_silu = false for CONV1_LIN
    tinyissimo_layer(
        IH, IW, IC, OC, KH, KW, PH, PW,
        false, false, LAYER_IDX,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE, QP_BASE, 0, 0,
        /*packed_rgb=*/ false,
        g_fmap_in, g_fmap_out, g_wt_mem, g_qp_mem, g_silu_mem);

    int8_t dut_out[OH*OW*OC];
    unpack_output(g_fmap_out, dut_out, OH, OW, OC, false);

    int errors = 0;
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++)
    for (int oc = 0; oc < OC; oc++) {
        int8_t ref = ref_pixel(ifmap, weights, bias, m0, nshift, lut,
                               zp_in, zp_out, false,
                               IH, IW, IC, KH, KW, PH, PW,
                               oh, ow, oc);
        int8_t got = dut_out[(oh*OW+ow)*OC + oc];
        check("T6", got, ref, oh, ow, oc);
        if (got != ref) errors++;
    }
    printf("  Errors: %d / %d\n\n", errors, OH*OW*OC);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 7: Branching detection head -- two branches from shared input
//   Mimics layers 10->11 and 10->14: both read from the same fmap
//   region (offset 0), but write to different output regions (64, 128).
//   Uses a single fmap buffer for both in and out (like RTL ping-pong).
// ═══════════════════════════════════════════════════════════════════════

static void test7_branch_offsets() {
    printf("=== Test 7: 4x4 branching offsets (shared input, split output) ===\n");

    const int IH=4, IW=4, IC=16;
    const int BR_A_OC=8, BR_B_OC=8;
    const int KH=3, KW=3, PH=1, PW=1;
    const int OH=IH, OW=IW;
    // Branch A writes at fmap offset 64, Branch B at 128
    // (oc_tiles=1 for OC=8, so each branch uses (oc_tiles/2)*OH*OW=8 words)
    const int WR_OFF_A = 64, WR_OFF_B = 128;
    const int RD_OFF = 0;  // both read from offset 0

    const int WT_BASE_A=0;
    const int WT_BASE_B = BR_A_OC * ((IC+TILE_IC-1)/TILE_IC) * KH * KW;
    const int QP_BASE_A=0, QP_BASE_B=BR_A_OC;
    const int LI_A=7, LI_B=8;

    int8_t   ifmap[IH*IW*IC];
    int8_t   wA[BR_A_OC*KH*KW*IC], wB[BR_B_OC*KH*KW*IC];
    int32_t  bA[BR_A_OC], bB[BR_B_OC];
    uint32_t m0A[BR_A_OC], m0B[BR_B_OC];
    uint8_t  nsA[BR_A_OC], nsB[BR_B_OC];
    int8_t   lut[256];

    g_seed = 7;
    for (int i = 0; i < IH*IW*IC; i++) ifmap[i] = randi8();
    for (int i = 0; i < BR_A_OC*KH*KW*IC; i++) wA[i] = randi8();
    for (int i = 0; i < BR_B_OC*KH*KW*IC; i++) wB[i] = randi8();
    for (int i = 0; i < BR_A_OC; i++) {
        bA[i] = (int32_t)(randi8())*64;
        m0A[i] = 0x80000000u;
        nsA[i] = 31;
    }
    for (int i = 0; i < BR_B_OC; i++) {
        bB[i] = (int32_t)(randi8())*64;
        m0B[i] = 0x80000000u;
        nsB[i] = 31;
    }
    build_identity_lut(lut);

    int8_t zp_in = 0, zp_out = 0;

    // Use a single shared fmap buffer for both read and write
    static ap_uint<128> g_fmap_shared[FMAP_DEPTH];
    memset(g_fmap_shared, 0, sizeof(g_fmap_shared));

    // Pack input at offset 0
    pack_input(ifmap, g_fmap_shared, IH, IW, IC);
    pack_weights_rom(wA, g_wt_mem, WT_BASE_A, BR_A_OC, IC, KH, KW);
    pack_weights_rom(wB, g_wt_mem, WT_BASE_B, BR_B_OC, IC, KH, KW);
    pack_qp_rom(bA, m0A, nsA, g_qp_mem, QP_BASE_A, BR_A_OC);
    pack_qp_rom(bB, m0B, nsB, g_qp_mem, QP_BASE_B, BR_B_OC);
    fill_silu_mem(lut, g_silu_mem, LI_A);
    fill_silu_mem(lut, g_silu_mem, LI_B);

    // Branch A: read from offset 0, write to WR_OFF_A
    tinyissimo_layer(
        IH, IW, IC, BR_A_OC, KH, KW, PH, PW,
        false, true, LI_A,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE_A, QP_BASE_A, RD_OFF, WR_OFF_A,
        /*packed_rgb=*/ false,
        g_fmap_shared, g_fmap_shared, g_wt_mem, g_qp_mem, g_silu_mem);

    // Branch B: read from same offset 0, write to WR_OFF_B
    tinyissimo_layer(
        IH, IW, IC, BR_B_OC, KH, KW, PH, PW,
        false, true, LI_B,
        (ap_int<8>)zp_in, (ap_int<8>)zp_out,
        WT_BASE_B, QP_BASE_B, RD_OFF, WR_OFF_B,
        /*packed_rgb=*/ false,
        g_fmap_shared, g_fmap_shared, g_wt_mem, g_qp_mem, g_silu_mem);

    // Verify Branch A output at WR_OFF_A
    int8_t dut_a[OH*OW*BR_A_OC];
    unpack_output(g_fmap_shared, dut_a, OH, OW, BR_A_OC, false, WR_OFF_A);

    int errors_a = 0;
    for (int oh=0; oh<OH; oh++)
    for (int ow=0; ow<OW; ow++)
    for (int oc=0; oc<BR_A_OC; oc++) {
        int8_t ref = ref_pixel(ifmap, wA, bA, m0A, nsA, lut,
                               zp_in, zp_out, true,
                               IH, IW, IC, KH, KW, PH, PW,
                               oh, ow, oc);
        int8_t got = dut_a[(oh*OW+ow)*BR_A_OC + oc];
        check("T7a", got, ref, oh, ow, oc);
        if (got != ref) errors_a++;
    }

    // Verify Branch B output at WR_OFF_B
    int8_t dut_b[OH*OW*BR_B_OC];
    unpack_output(g_fmap_shared, dut_b, OH, OW, BR_B_OC, false, WR_OFF_B);

    int errors_b = 0;
    for (int oh=0; oh<OH; oh++)
    for (int ow=0; ow<OW; ow++)
    for (int oc=0; oc<BR_B_OC; oc++) {
        int8_t ref = ref_pixel(ifmap, wB, bB, m0B, nsB, lut,
                               zp_in, zp_out, true,
                               IH, IW, IC, KH, KW, PH, PW,
                               oh, ow, oc);
        int8_t got = dut_b[(oh*OW+ow)*BR_B_OC + oc];
        check("T7b", got, ref, oh, ow, oc);
        if (got != ref) errors_b++;
    }

    // Verify input at offset 0 was NOT corrupted by writes
    int8_t readback[IH*IW*IC];
    unpack_output(g_fmap_shared, readback, IH, IW, IC, false, 0);
    int corrupted = 0;
    for (int i = 0; i < IH*IW*IC; i++) {
        if (readback[i] != ifmap[i]) corrupted++;
    }

    printf("  Branch A errors: %d / %d\n", errors_a, OH*OW*BR_A_OC);
    printf("  Branch B errors: %d / %d\n", errors_b, OH*OW*BR_B_OC);
    printf("  Input corruption: %d / %d\n\n", corrupted, IH*IW*IC);
}

// ═════════════════════════════════════════════════════════════════════════
// Test 8: Full 17-layer model end-to-end
//
// Loads the production ROMs (weight_rom.mem, qp_packed_rom.mem, silu_lut.mem)
// and the camera frame (pixels_layer0.mem) from the project's weights/
// directory, packs the frame into fmap_b in the 4-pixels-per-128-bit-word
// format that the HDL AXI-Stream preload uses, then calls the new
// tinyissimo_layer_top() once to walk all 17 layers.  The cv2 (layer 13)
// and cv3 (layer 16) detection-head outputs are compared byte-for-byte
// against golden_layer{13,16}_uram.mem produced by
// hardware/scripts/generate_conv3d_golden.py — which itself matches the
// HDL conv3d arithmetic exactly.  Both engines must therefore produce
// byte-identical output for the runtime A/B mux to be useful.
//
// This test is what makes Vitis HLS cosim pass — without it, no call to
// tinyissimo_layer_top() exists in the testbench and cosim aborts.
// ═════════════════════════════════════════════════════════════════════════

#ifndef TB_DATA_DIR
// Absolute path to the project root.  csim, cosim, and any other Vitis
// HLS tool flow run from different working directories deep inside the
// solution tree, so a relative path is fragile.  If you move the project,
// update this define or override at compile time:
//   syn.cflags = -DTB_DATA_DIR=\"/your/path/to/ee4218-project/\"
//   tb.cflags  = -DTB_DATA_DIR=\"/your/path/to/ee4218-project/\"
#define TB_DATA_DIR "/home/leeey/Downloads/ee4218-project/"
#endif

#include <fstream>
#include <string>
#include <cstring>

static int load_hex_mem(const char *path, ap_uint<128> *out, int max_words,
                        int word_bits)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        printf("  ERROR: cannot open %s\n", path);
        return -1;
    }
    int n = 0;
    std::string line;
    while (std::getline(f, line) && n < max_words) {
        // Strip leading whitespace
        size_t s = line.find_first_not_of(" \t");
        if (s == std::string::npos) continue;
        // Skip comment lines
        if (line.compare(s, 2, "//") == 0) continue;
        // Trim trailing whitespace / CR
        size_t e = line.find_last_not_of(" \t\r\n");
        std::string hex = line.substr(s, e - s + 1);
        if (hex.empty()) continue;

        ap_uint<128> word = 0;
        // Parse hex string into 128-bit word, low-nibble last
        // hex chars are big-endian (MSB first)
        int nibbles = (int)hex.size();
        for (int i = 0; i < nibbles; i++) {
            char c = hex[i];
            int v;
            if      (c >= '0' && c <= '9') v = c - '0';
            else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
            else continue;
            int bit = (nibbles - 1 - i) * 4;
            if (bit + 3 < word_bits)
                word.range(bit + 3, bit) = v;
        }
        out[n++] = word;
    }
    return n;
}

static int load_hex_mem_8(const char *path, ap_uint<8> *out, int max_words)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        printf("  ERROR: cannot open %s\n", path);
        return -1;
    }
    int n = 0;
    std::string line;
    while (std::getline(f, line) && n < max_words) {
        size_t s = line.find_first_not_of(" \t");
        if (s == std::string::npos) continue;
        if (line.compare(s, 2, "//") == 0) continue;
        unsigned int v = 0;
        if (sscanf(line.c_str() + s, "%x", &v) == 1)
            out[n++] = (ap_uint<8>)(v & 0xFF);
    }
    return n;
}

// Read pixels_layer0.mem ([ch][row][col] one int8 byte per line, 196608 lines)
// and pack into fmap as 4 RGB pixels per 128-bit word matching the HDL AXIS
// preload accumulator format (inference_top.sv:324-336).
static int load_packed_rgb(const char *path, ap_uint<128> *fmap)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        printf("  ERROR: cannot open %s\n", path);
        return -1;
    }
    static int8_t pix[3][256][256];
    int total = 0;
    std::string line;
    int ch = 0, row = 0, col = 0;
    while (std::getline(f, line) && total < 3 * 256 * 256) {
        size_t s = line.find_first_not_of(" \t");
        if (s == std::string::npos) continue;
        if (line.compare(s, 2, "//") == 0) continue;
        unsigned int v = 0;
        if (sscanf(line.c_str() + s, "%x", &v) != 1) continue;
        pix[ch][row][col] = (int8_t)(v & 0xFF);
        total++;
        if (++col == 256) {
            col = 0;
            if (++row == 256) {
                row = 0;
                ch++;
            }
        }
    }
    if (total != 3 * 256 * 256) {
        printf("  ERROR: expected %d pixel bytes, got %d\n",
               3 * 256 * 256, total);
        return -1;
    }
    // Pack: linear pixel index N -> word N/4, lane N%4
    for (int r = 0; r < 256; r++) {
        for (int c = 0; c < 256; c++) {
            int linear = r * 256 + c;
            int word_addr = linear >> 2;
            int lane     = linear & 3;
            uint32_t w32 =
                  ((uint32_t)(uint8_t)pix[2][r][c] << 16)
                | ((uint32_t)(uint8_t)pix[1][r][c] <<  8)
                | ((uint32_t)(uint8_t)pix[0][r][c] <<  0);
            ap_uint<128> w = fmap[word_addr];
            w.range(lane * 32 + 31, lane * 32) = w32;
            fmap[word_addr] = w;
        }
    }
    return total;
}

// Compare a slice of an fmap buffer against a golden .mem file.
//
// valid_channels: how many of the 16 channels in each 128-bit word
//   actually carry meaningful output for this layer.  Channels above
//   this index hold residual data from earlier layers (which is fine
//   — host code reads only the first cout bytes per word).  Set to 16
//   when all channels are valid; for layer 16 (cv3, cout=3) set to 3.
static int compare_uram_region(const ap_uint<128> *got, const char *golden_path,
                                int base_addr, int n_words, int valid_channels)
{
    static ap_uint<128> golden[16384];
    int loaded = load_hex_mem(golden_path, golden, n_words, 128);
    if (loaded != n_words) {
        printf("  WARN: %s loaded %d / %d words\n",
               golden_path, loaded, n_words);
    }
    // Build a mask covering only the valid_channels low bytes of each word.
    ap_uint<128> mask = 0;
    int valid_bits = valid_channels * 8;
    if (valid_bits > 0)
        mask.range(valid_bits - 1, 0) = ((ap_uint<128>)1 << valid_bits) - 1;

    int errors = 0;
    for (int i = 0; i < n_words && i < loaded; i++) {
        ap_uint<128> g = got[base_addr + i] & mask;
        ap_uint<128> r = golden[i]          & mask;
        if (g != r) {
            if (errors < 4) {
                printf("    [FAIL] @%d  got=%016lx%016lx  ref=%016lx%016lx\n",
                       base_addr + i,
                       (unsigned long)g.range(127, 64),
                       (unsigned long)g.range(63, 0),
                       (unsigned long)r.range(127, 64),
                       (unsigned long)r.range(63, 0));
            }
            errors++;
        }
    }
    return errors;
}

static void test8_full_model() {
    printf("=== Test 8: full 17-layer model end-to-end ===\n");

    // Persistent buffers (heap-sized to avoid blowing the stack)
    static ap_uint<128> g_fmap_a   [FMAP_DEPTH];
    static ap_uint<128> g_fmap_b   [FMAP_DEPTH];
    static ap_uint<128> g_wt_full  [WT_DEPTH];
    static ap_uint<128> g_qp_full  [QP_DEPTH];
    static ap_uint<8>   g_silu_full[SILU_LUT_DEPTH];

    memset(g_fmap_a,   0, sizeof(g_fmap_a));
    memset(g_fmap_b,   0, sizeof(g_fmap_b));
    memset(g_wt_full,  0, sizeof(g_wt_full));
    memset(g_qp_full,  0, sizeof(g_qp_full));
    memset(g_silu_full, 0, sizeof(g_silu_full));

    // ── Load production ROMs and the camera frame ──
    const char *wt_path  = TB_DATA_DIR "hardware/weights/hdl/weight_rom.mem";
    const char *qp_path  = TB_DATA_DIR "hardware/weights/hdl/qp_packed_rom.mem";
    const char *lut_path = TB_DATA_DIR "hardware/weights/hdl/silu_lut.mem";
    const char *pix_path = TB_DATA_DIR "hardware/testbench/inference_hdl/pixels_layer0.mem";
    const char *g13_path = TB_DATA_DIR "hardware/testbench/inference_hdl/golden_layer13_uram.mem";
    const char *g16_path = TB_DATA_DIR "hardware/testbench/inference_hdl/golden_layer16_uram.mem";

    int wt_n  = load_hex_mem (wt_path,  g_wt_full, WT_DEPTH, 128);
    int qp_n  = load_hex_mem (qp_path,  g_qp_full, QP_DEPTH, 72);
    int lut_n = load_hex_mem_8(lut_path, g_silu_full, SILU_LUT_DEPTH);
    int pix_n = load_packed_rgb(pix_path, g_fmap_b);

    if (wt_n <= 0 || qp_n <= 0 || lut_n <= 0 || pix_n <= 0) {
        printf("  SKIP: data files not found (set TB_DATA_DIR or copy mem files)\n\n");
        return;
    }
    printf("  Loaded: %d wt, %d qp, %d silu, %d pixels\n",
           wt_n, qp_n, lut_n, pix_n);

    // ── Run the full top function ──
    volatile int curr_layer = -1;
    tinyissimo_layer_top(g_fmap_a, g_fmap_b,
                         g_wt_full, g_qp_full, g_silu_full,
                         &curr_layer);
    printf("  Top function returned, curr_layer=%d (expect 17)\n", curr_layer);

    if (curr_layer != NUM_LAYERS) {
        printf("  [FAIL] curr_layer != NUM_LAYERS (%d)\n", NUM_LAYERS);
        g_fail++;
        return;
    }

    // ── Compare detection-head outputs against golden ──
    // Layer 13 (cv2): pp_buf_sel=1, pp_wr_offset=256, 64 channels x 8x8
    //   = 4 channel groups x 64 spatial = 256 words at fmap_b[256..511]
    // Layer 16 (cv3): pp_buf_sel=1, pp_wr_offset=512, 3 channels x 8x8
    //   = 1 channel group x 64 spatial = 64 words at fmap_b[512..575]
    // Layer 13 (cv2): cout=64 — full 16 channels per word, all valid.
    printf("  Verifying layer 13 (cv2, fmap_b[256..511])...\n");
    int e13 = compare_uram_region(g_fmap_b, g13_path, 256, 256, 16);

    // Layer 16 (cv3): cout=3 — only the low 3 bytes per word carry data.
    // The high bits hold residual data from layer 14 (which writes 24
    // channels to fmap_b[512..639]) and are intentionally ignored, just
    // like the host driver only reads the first cout valid bytes.
    printf("  Verifying layer 16 (cv3, fmap_b[512..575])...\n");
    int e16 = compare_uram_region(g_fmap_b, g16_path, 512, 64, 3);

    int total = 256 + 64;
    int errors = e13 + e16;
    printf("  Layer 13 errors: %d / 256\n", e13);
    printf("  Layer 16 errors: %d / 64\n", e16);
    printf("  Total errors:    %d / %d\n\n", errors, total);

    g_pass += (total - errors);
    g_fail += errors;
}

// ═════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════

int main() {
    printf("tinyissimo_layer testbench (runtime-parameterized)\n");
    printf("TILE_IC=%d  TILE_OC=%d\n\n", TILE_IC, TILE_OC);

    test1_no_pool();
    test2_maxpool();
    test3_nonmultiple();
    test4_chained();
    test5_conv1();
    test6_conv1_lin();
    test7_branch_offsets();
    test8_full_model();

    printf("==========================================\n");
    printf("TOTAL  pass=%d  fail=%d\n", g_pass, g_fail);
    printf("==========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
