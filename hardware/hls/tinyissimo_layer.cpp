// tinyissimo_layer.cpp
//
// Runtime-parameterized HLS convolution layer for Tinyissimo CNN.
//
// Architecture: Two-phase LOAD/COMPUTE per output-channel tile.
//   LOAD  – Sequential reads from weight ROM + QP ROM into local buffers
//   COMPUTE – Stream through input feature map with 128 parallel MACs
//
// Supports: CONV3 (3x3+SiLU), CONV3_POOL (3x3+SiLU+MaxPool2x2),
//           CONV1 (1x1+SiLU), CONV1_LIN (1x1+linear)
#include "tinyissimo_layer.h"

// ─────────────────────────────────────────────────────────────────────────
// Helper: Clip a 32-bit value to int8 range [-128, 127]
// ─────────────────────────────────────────────────────────────────────────
static ap_int<8> clip_int8(ap_int<32> x) {
#pragma HLS INLINE
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (ap_int<8>)x;
}

// ─────────────────────────────────────────────────────────────────────────
// Helper: Fixed-point rescale  (matches RTL conv3d quantisation)
//   round( acc * m0 / 2^n_shift )  then clip to int8
// ─────────────────────────────────────────────────────────────────────────
static ap_int<8> rescale(ap_int<32> acc, ap_uint<32> m0, ap_uint<8> n_shift) {
#pragma HLS INLINE
    ap_int<64> product = (ap_int<64>)acc * (ap_int<64>)m0;
    ap_int<64> half    = (ap_int<64>)1 << (n_shift - 1);
    ap_int<64> rounded = (product + half) >> n_shift;
    return clip_int8((ap_int<32>)rounded);
}

// ─────────────────────────────────────────────────────────────────────────
// Main layer function
// ─────────────────────────────────────────────────────────────────────────
void tinyissimo_layer(
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
#pragma HLS INLINE

    // ── Derived runtime constants ───────────────────────────────────────
    const int ic_tiles    = (in_c + TILE_IC - 1) / TILE_IC;
    const int oc_tiles    = (out_c + TILE_OC - 1) / TILE_OC;
    const int oc_tail     = (out_c % TILE_OC == 0) ? TILE_OC : (out_c % TILE_OC);
    const int out_h       = in_h;  // stride is always 1 in this model
    const int out_w       = in_w;
    const int ksq         = kh * kw;
    const int pool_h      = use_maxpool ? (out_h >> 1) : out_h;
    const int pool_w      = use_maxpool ? (out_w >> 1) : out_w;
    const int spatial_out = pool_h * pool_w;

    // ── Local SiLU LUT slice (256 entries for current layer) ────────────
    ap_int<8> local_silu[SILU_SLICE];
#pragma HLS ARRAY_PARTITION variable=local_silu complete dim=1

    if (use_silu) {
        const int silu_base = layer_idx * SILU_SLICE;
        SILU_COPY:
        for (int i = 0; i < SILU_SLICE; i++) {
#pragma HLS PIPELINE II=1
            local_silu[i] = (ap_int<8>)silu_mem[silu_base + i];
        }
    }

    // ── Weight buffer (partitioned for 8-way parallel OC read) ──────────
    ap_uint<128> w_buf[TILE_OC][MAX_IC_TILES][MAX_KH][MAX_KW];
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1
#pragma HLS BIND_STORAGE variable=w_buf type=ram_2p impl=bram

    // ── Per-channel quantisation parameter buffers ──────────────────────
    ap_int<32>  bias_buf  [TILE_OC];
    ap_uint<32> m0_buf    [TILE_OC];
    ap_uint<8>  nshift_buf[TILE_OC];
#pragma HLS ARRAY_PARTITION variable=bias_buf   complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_buf     complete dim=1
#pragma HLS ARRAY_PARTITION variable=nshift_buf complete dim=1

    // ── Accumulator and activation registers ────────────────────────────
    ap_int<32> acc     [TILE_OC];
    ap_int<8>  act_tile[TILE_OC];
#pragma HLS ARRAY_PARTITION variable=acc      complete dim=1
#pragma HLS ARRAY_PARTITION variable=act_tile complete dim=1

    // ── MaxPool row buffer: 2 rows x MAX_W x TILE_OC ───────────────────
    ap_int<8> pool_buf[2][MAX_W][TILE_OC];
#pragma HLS ARRAY_PARTITION variable=pool_buf complete dim=3

    // ════════════════════════════════════════════════════════════════════
    // Main processing: iterate over OC tiles
    // ════════════════════════════════════════════════════════════════════
    OC_TILE_LOOP:
    for (int oct = 0; oct < oc_tiles; oct++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16

        const int oc_valid = (oct == oc_tiles - 1) ? oc_tail : TILE_OC;
        const int oc_base  = oct * TILE_OC;

        // ── LOAD PHASE 1: Quantisation parameters ──────────────────────
        LOAD_QP:
        for (int t = 0; t < TILE_OC; t++) {
#pragma HLS PIPELINE II=1
            if (t < oc_valid) {
                ap_uint<72> qp_word = qp_mem[qp_base + oc_base + t];
                bias_buf[t]   = (ap_int<32>)  qp_word.range(31,  0);
                m0_buf[t]     = (ap_uint<32>) qp_word.range(63, 32);
                nshift_buf[t] = (ap_uint<8>)  qp_word.range(69, 64);
            } else {
                bias_buf[t]   = 0;
                m0_buf[t]     = 0;
                nshift_buf[t] = 1;  // avoid shift-by-0 in rescale
            }
        }

        // ── LOAD PHASE 2: Weights into local buffer ────────────────────
        //
        // Weight ROM layout (matches generate_hdl_weights.py):
        //   For each oc, for each ic_tile, for each kh*kw:
        //     one 128-bit word = 16 x int8 weights
        //
        // Address: wt_base + oc * ic_tiles * ksq + ict * ksq + kh*kw + kw
        const int wt_words_per_oc = ic_tiles * ksq;
        const int wt_load_total   = TILE_OC * wt_words_per_oc;
        const int wt_load_base    = wt_base + oc_base * wt_words_per_oc;

        int wt_t = 0, wt_ict = 0, wt_kh = 0, wt_kw = 0;

        LOAD_WT:
        for (int i = 0; i < wt_load_total; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=576

            w_buf[wt_t][wt_ict][wt_kh][wt_kw] = wt_mem[wt_load_base + i];

            // Advance counters: kw -> kh -> ict -> t
            if (++wt_kw >= kw) {
                wt_kw = 0;
                if (++wt_kh >= kh) {
                    wt_kh = 0;
                    if (++wt_ict >= ic_tiles) {
                        wt_ict = 0;
                        wt_t++;
                    }
                }
            }
        }

        // ── COMPUTE PHASE: stream through spatial positions ────────────
        const int conv_total = ic_tiles * ksq;

        OUT_ROW_LOOP:
        for (int oh = 0; oh < out_h; oh++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        OUT_COL_LOOP:
        for (int ow = 0; ow < out_w; ow++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256

            // ── Step 0: Initialise accumulators with folded biases ─────
            INIT_ACC:
            for (int t = 0; t < TILE_OC; t++) {
#pragma HLS UNROLL
                acc[t] = bias_buf[t];
            }

            // ── Step 1: Convolution MAC ────────────────────────────────
            // Manually flattened ic_tiles x kh x kw loop for clean II=1.
            int c_ict = 0, c_kh = 0, c_kw = 0;

            CONV_LOOP:
            for (int iter = 0; iter < conv_total; iter++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=72

                // Compute input coordinates (padded space)
                int h = oh + c_kh - pad_h;
                int w = ow + c_kw - pad_w;
                bool is_pad = (h < 0 || h >= in_h || w < 0 || w >= in_w);

                // Safe address for speculative reads when padding
                int h_safe = (h < 0) ? 0 : ((h >= in_h) ? 0 : h);
                int w_safe = (w < 0) ? 0 : ((w >= in_w) ? 0 : w);
                ap_uint<128> raw = fmap_in[c_ict * in_h * in_w
                                           + h_safe * in_w + w_safe];

                // Unpack 16 int8 pixel values (pad channels use zp_in)
                ap_int<8> pix[TILE_IC];
#pragma HLS ARRAY_PARTITION variable=pix complete dim=1

                UNPACK_PIX:
                for (int i = 0; i < TILE_IC; i++) {
#pragma HLS UNROLL
                    pix[i] = is_pad ? zp_in
                                    : (ap_int<8>)raw.range(i*8+7, i*8);
                }

                // MAC across all TILE_OC output lanes
                MAC_OC:
                for (int t = 0; t < TILE_OC; t++) {
#pragma HLS UNROLL

                    ap_uint<128> ww = w_buf[t][c_ict][c_kh][c_kw];
                    ap_int<32> mac = 0;

                    MAC_IC:
                    for (int i = 0; i < TILE_IC; i++) {
#pragma HLS UNROLL
                        ap_int<8> w_val = (ap_int<8>)ww.range(i*8+7, i*8);
                        mac += (ap_int<32>)pix[i] * (ap_int<32>)w_val;
                    }
                    acc[t] += mac;
                }

                // Advance counters: kw -> kh -> ict
                if (++c_kw >= kw) {
                    c_kw = 0;
                    if (++c_kh >= kh) {
                        c_kh = 0;
                        c_ict++;
                    }
                }
            } // CONV_LOOP

            // ── Step 2: Post-processing (rescale, zp_out, SiLU) ────────
            POST_PROC:
            for (int t = 0; t < TILE_OC; t++) {
#pragma HLS UNROLL
                if (t < oc_valid) {
                    // Rescale 32-bit accumulator to int8
                    ap_int<8> scaled = rescale(acc[t], m0_buf[t],
                                               nshift_buf[t]);
                    // Add output zero-point and clip
                    ap_int<32> with_zp = (ap_int<32>)scaled
                                       + (ap_int<32>)zp_out;
                    ap_int<8>  clamped = clip_int8(with_zp);

                    // SiLU activation via LUT (or linear bypass)
                    // Addressing: XOR MSB to map signed [-128..127] -> [0..255]
                    // Matches RTL: {~val[7], val[6:0]}
                    if (use_silu) {
                        int lut_idx = (int)((ap_uint<8>)clamped
                                          ^ (ap_uint<8>)0x80);
                        act_tile[t] = local_silu[lut_idx];
                    } else {
                        act_tile[t] = clamped;
                    }
                } else {
                    act_tile[t] = 0;
                }
            }

            // ── Step 3: Write output (with optional MaxPool 2x2) ───────
            if (use_maxpool) {
                int buf_row = oh & 1;

                POOL_STORE:
                for (int t = 0; t < TILE_OC; t++) {
#pragma HLS UNROLL
                    pool_buf[buf_row][ow][t] = act_tile[t];
                }

                if ((oh & 1) && (ow & 1)) {
                    int pool_oh = oh >> 1;
                    int pool_ow = ow >> 1;

                    POOL_MAX:
                    for (int t = 0; t < TILE_OC; t++) {
#pragma HLS UNROLL
                        ap_int<8> v00 = pool_buf[0][ow-1][t];
                        ap_int<8> v01 = pool_buf[0][ow  ][t];
                        ap_int<8> v10 = pool_buf[1][ow-1][t];
                        ap_int<8> v11 = pool_buf[1][ow  ][t];

                        ap_int<8> mx = v00;
                        if (v01 > mx) mx = v01;
                        if (v10 > mx) mx = v10;
                        if (v11 > mx) mx = v11;
                        act_tile[t] = mx;
                    }

                    // Pack into 128-bit output word (RMW)
                    // Build 64-bit half with compile-time constant ranges
                    int out_idx = (oct / 2) * spatial_out
                                + pool_oh * pool_w + pool_ow;

                    ap_uint<64> half = 0;
                    PACK_POOL:
                    for (int t = 0; t < TILE_OC; t++) {
#pragma HLS UNROLL
                        if (t < oc_valid)
                            half.range(t*8+7, t*8) = (ap_uint<8>)act_tile[t];
                    }
                    ap_uint<128> out_word = fmap_out[out_idx];
                    if (oct & 1)
                        out_word.range(127, 64) = half;
                    else
                        out_word.range(63, 0) = half;
                    fmap_out[out_idx] = out_word;
                }

            } else {
                // No pooling: pack and write directly (RMW)
                // Build 64-bit half with compile-time constant ranges
                int out_idx = (oct / 2) * out_h * out_w
                            + oh * out_w + ow;

                ap_uint<64> half = 0;
                PACK_DIRECT:
                for (int t = 0; t < TILE_OC; t++) {
#pragma HLS UNROLL
                    if (t < oc_valid)
                        half.range(t*8+7, t*8) = (ap_uint<8>)act_tile[t];
                }
                ap_uint<128> out_word = fmap_out[out_idx];
                if (oct & 1)
                    out_word.range(127, 64) = half;
                else
                    out_word.range(63, 0) = half;
                fmap_out[out_idx] = out_word;
            }

        } // OUT_COL_LOOP
        } // OUT_ROW_LOOP
    } // OC_TILE_LOOP
}
