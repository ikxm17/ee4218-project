`timescale 1ns / 1ps
`include "layer_config.svh"

module inference_hdl #(
    parameter C_PAR        = C_PARALLEL,
    parameter MAX_COUT_PAR = 4,  // physical replica count for conv3d's downstream pipeline
    parameter K            = 3,
    parameter N_BITS       = 8,
    parameter ACC_BITS     = 32,
    parameter DEPTH_BITS   = 16,
    parameter KSQ          = K * K,
    parameter WT_DATA_W    = C_PAR * N_BITS,
    parameter WT_ADDR_W    = $clog2(32768),
    parameter QP_ADDR_W    = $clog2(1024),
    parameter LUT_ADDR_W   = $clog2(ACT_LUT_DEPTH),
    parameter FMAP_DATA_W  = C_PAR * N_BITS,
    parameter FMAP_ADDR_W  = $clog2(16384),
    parameter ACC_DATA_W   = MAX_COUT_PAR * ACC_BITS  // packed per-lane ACC word
)(
    input  logic                             aclk,
    input  logic                             aresetn,
    input  logic                             start,
    output logic                             done,

    /* Weight ROM read port (inference_top u_wt_mem port B) */
    output logic                             wt_mem_en_b,
    output logic [WT_ADDR_W-1:0]             wt_mem_addr_b,
    input  logic [WT_DATA_W-1:0]             wt_mem_dout_b,

    /* QP ROM read port (inference_top u_qp_mem port B) */
    output logic                             qp_mem_en_b,
    output logic [QP_ADDR_W-1:0]             qp_mem_addr_b,
    input  logic [71:0]                      qp_mem_dout_b,

    /* SiLU LUT ROM read port — one BRAM replica per Cout lane.
       inference_top instantiates MAX_COUT_PAR replicas of u_silu_mem and
       routes each lane independently. Lanes beyond cout_par_active stay
       idle (en_b=0) — masked by conv3d's per-lane cout_par_active gating. */
    output logic [MAX_COUT_PAR-1:0]                            silu_mem_en_b,
    output logic [MAX_COUT_PAR-1:0][LUT_ADDR_W-1:0]            silu_mem_addr_b,
    input  logic [MAX_COUT_PAR-1:0][N_BITS-1:0]                silu_mem_dout_b,

    /* Input feature-map read port (inference_top routes to fmap_a/b) */
    output logic [DEPTH_BITS-1:0]            in_buf_rd_addr,
    output logic                             in_buf_rd_en,
    input  logic [C_PAR*N_BITS-1:0]   in_buf_rd_data,

    /* Output feature-map RMW read-back port (inference_top routes to fmap_a/b) */
    output logic                             out_buf_rd_en,
    output logic [FMAP_ADDR_W-1:0]           out_buf_rd_addr,
    input  logic [FMAP_DATA_W-1:0]           out_buf_rd_data,

    /* Output feature-map RMW write port (inference_top routes to fmap_a/b) */
    output logic                             out_buf_wr_en,
    output logic [FMAP_ADDR_W-1:0]           out_buf_wr_addr,
    output logic [FMAP_DATA_W-1:0]           out_buf_wr_data,

    /* Status (for AXI register file) */
    output logic [4:0]                       curr_layer_idx,

    /* Sub-pingpong config (for inference_top URAM routing) */
    output logic                             curr_pp_buf_sel,
    output logic [13:0]                      curr_pp_rd_offset,

    /* Inference loop bound — host can stop inference early to read
     * intermediate fmap_a/b state for layer-by-layer debugging.
     * Default 5'd17 = full network. Tied to 5'd17 in TB_MODE=1; driven
     * by axil_regs reg_max_layers in TB_MODE=0. */
    input  logic [4:0]                       max_layers_run,

    /* Debug capture — first 4 conv3d/conv1d RES write addresses during
     * layer 0. Used to localize the silicon-only +1 URAM shift bug.
     * If silicon shows addrs 0, 1, 2, 3 here, conv3d is writing to
     * correct RES addresses and the shift is downstream (rmw writer or
     * URAM primitive). If it shows 16383, 0, 1, 2 (or similar shifted
     * sequence), the bug is in conv3d. */
    output logic [13:0]                      dbg_conv_res_addr_0,
    output logic [13:0]                      dbg_conv_res_addr_1,
    output logic [13:0]                      dbg_conv_res_addr_2,
    output logic [13:0]                      dbg_conv_res_addr_3,
    output logic [13:0]                      dbg_out_wr_addr_0,
    output logic [13:0]                      dbg_out_wr_addr_1,
    output logic [13:0]                      dbg_out_wr_addr_2,
    output logic [13:0]                      dbg_out_wr_addr_3,
    output logic [13:0]                      dbg_pool_out_addr_0,
    output logic [13:0]                      dbg_pool_out_addr_1,
    output logic [13:0]                      dbg_pool_out_addr_2,
    output logic [13:0]                      dbg_pool_out_addr_3,
    output logic [13:0]                      dbg_rmw_s0_addr_0,
    output logic [13:0]                      dbg_rmw_s0_addr_1,
    output logic [13:0]                      dbg_rmw_s0_addr_2,
    output logic [13:0]                      dbg_rmw_s0_addr_3,
    /* Captures rmw_base_addr (combinational) snapshotted at each
     * rmw_s0_valid fire. If silicon shows 16383 but sim shows 0, the
     * issue is in rmw_base_addr computation (pp_wr_offset, ch_out>>4,
     * or h_out). */
    output logic [13:0]                      dbg_rmw_base_0,
    output logic [13:0]                      dbg_rmw_base_1,
    output logic [13:0]                      dbg_rmw_base_2,
    output logic [13:0]                      dbg_rmw_base_3,
    /* Snapshot of the full r_cfg.pp_wr_offset + curr_ch_out at capture 0.
     * Helps disambiguate where the 16383 is coming from. */
    output logic [13:0]                      dbg_pp_wr_offset,
    output logic [7:0]                       dbg_ch_out,
    output logic [8:0]                       dbg_h_out,
    output logic [3:0]                       dbg_capture_count
);

    /* ================================================================
     *  FSM State Encoding
     * ================================================================ */
    typedef enum logic [2:0] {
        S_IDLE       = 3'd0,
        S_LOAD       = 3'd1,
        S_COMPUTE    = 3'd2,
        S_NEXT_CHOUT = 3'd3,
        S_NEXT_LAYER = 3'd4
    } state_t;

    state_t state;
    state_t next_state;

    /* ================================================================
     *  Shadow Registers
     * ================================================================ */

    /* Layer config (from compile-time LAYER_CFG LUT) */
    layer_cfg_t r_cfg;

    /* Layer index (for activation LUT addressing) */
    logic [4:0] r_layer_idx;

    /* QP shadow registers (per output channel, from QP ROM) */
    /* Per-Cout-lane quant params — sequentially loaded over cout_par_active
       cycles in S_LOAD from the single-port shared QP ROM.  Lanes beyond
       cout_par_active are don't-care (conv3d masks them via cout_par_active). */
    logic signed [MAX_COUT_PAR-1:0][31:0] r_bias;
    logic        [MAX_COUT_PAR-1:0][31:0] r_m0;
    logic        [MAX_COUT_PAR-1:0][ 5:0] r_nshift;

    /* Runtime Cin/Cout parallelism split, sourced from per-layer config.
       log2_cgs=4 → cin_group_size=16, cout_par_active=1 (legacy scalar).
       log2_cgs=2 → cin_group_size=4,  cout_par_active=4 (Cout-parallel). */
    localparam int LOG2_CPAR        = $clog2(C_PAR);     // 4 for C_PAR=16
    wire [2:0]     log2_cgs_curr    = r_cfg.log2_cin_group_size;
    wire [4:0]     cout_par_active  = C_PAR[4:0] >> log2_cgs_curr;
    wire [2:0]     log2_cout_par    = LOG2_CPAR[2:0] - log2_cgs_curr;

    /* ================================================================
     *  Runtime Layer-Type Derived Signals
     * ================================================================ */
    wire        is_conv1  = r_cfg.layer_type[1];   // 1 for CONV1/CONV1_LIN
    wire  [3:0] wt_words  = is_conv1 ? 4'd1 : 4'd9; // weight entries per (ch_out, cin_group)

    /* ================================================================
     *  Double-Buffered Weight Register File
     * ================================================================ */

    /* Two banks: 9 rows x 16 cols x 8 bits each (CONV1 uses only row 0) */
    logic signed [N_BITS-1:0] wt_bank_a [0:KSQ-1][0:C_PAR-1];
    logic signed [N_BITS-1:0] wt_bank_b [0:KSQ-1][0:C_PAR-1];
    logic                     wt_sel;   // 0 = bank A active, 1 = bank B active

    /* Transpose + mux: ROM [kp][channel] -> conv3d [channel][kp] */
    logic [C_PAR*KSQ*N_BITS-1:0] weights_flat;

    generate
        for (genvar slot = 0; slot < C_PAR; slot++) begin : g_slot
            for (genvar kp = 0; kp < KSQ; kp++) begin : g_kp
                assign weights_flat[slot*KSQ*N_BITS + kp*N_BITS +: N_BITS] =
                    wt_sel ? wt_bank_b[kp][slot] : wt_bank_a[kp][slot];
            end
        end
    endgenerate

    /* Conv1x1 weight bus: only row 0 of each bank (K=1, one weight per channel) */
    logic [C_PAR*N_BITS-1:0] weights_1x1;

    generate
        for (genvar s = 0; s < C_PAR; s++) begin : g_w1
            assign weights_1x1[s*N_BITS +: N_BITS] =
                wt_sel ? wt_bank_b[0][s] : wt_bank_a[0][s];
        end
    endgenerate

    /* ================================================================
     *  Counters & Control
     * ================================================================ */
    logic [7:0]  ch_out;
    logic [4:0]  layer_idx;
    wire  [4:0]  next_layer_idx  = layer_idx + 5'd1;

    /* Combinational lookahead into LAYER_CFG.
     * Hoisted out of always_ff blocks so Vivado does not infer phantom
     * sequential elements for a local _next struct (one phantom FF per
     * packed field × two always_ff blocks = 26 spurious 8-6014 warnings).
     * The whole-struct assignment form sidesteps the XSim variable-indexed
     * struct field extraction bug (commits b799984, b271749) — that bug
     * only fires on LAYER_CFG[var].field; copying the whole element to a
     * local first and then field-selecting from the local is safe. */
    layer_cfg_t next_layer_cfg;
    always_comb next_layer_cfg = LAYER_CFG[next_layer_idx];

    wire  [8:0]  rt_act_size     = r_cfg.h_in;
    wire  [7:0]  rt_cin          = r_cfg.cin;
    wire  [7:0]  rt_cout         = r_cfg.cout;
    wire  [3:0]  rt_cin_grp      = r_cfg.cin_grp;
    wire  [3:0]  rt_total_rounds = r_cfg.cin_grp;

    logic [WT_ADDR_W-1:0]                       wt_addr_reg;
    logic [3:0]                                 load_cnt;       // 0..wt_words-1 in S_LOAD
    logic [3:0]                                 round_loaded;   // how many rounds loaded so far

    /* Background preload (runs inside S_COMPUTE) */
    logic       preload_active;
    logic       preload_done;
    logic [3:0] preload_cnt;    // 0..wt_words in preload sequence

    /* Compute engine interface signals */
    logic conv3d_start, conv1_start;
    logic conv3d_done,  conv1_done;
    logic conv3d_req_weights, conv1_req_weights;
    logic conv3d_weights_ready, conv1_weights_ready;

    /* Muxed compute signals */
    wire compute_done   = is_conv1 ? conv1_done        : conv3d_done;
    wire compute_req_wt = is_conv1 ? conv1_req_weights  : conv3d_req_weights;

    /* Per-engine pixel interface */
    logic [DEPTH_BITS-1:0] conv3d_pixel_addr, conv1_pixel_addr;
    logic                  conv3d_pixel_en,   conv1_pixel_en;

    /* Per-engine ACC interface (packed MAX_COUT_PAR-wide) */
    logic                                 conv3d_acc_wr_en,   conv1_acc_wr_en;
    logic [DEPTH_BITS-1:0]                conv3d_acc_wr_addr, conv1_acc_wr_addr;
    logic signed [ACC_DATA_W-1:0]         conv3d_acc_wr_data, conv1_acc_wr_data;
    logic                                 conv3d_acc_rd_en,   conv1_acc_rd_en;
    logic [DEPTH_BITS-1:0]                conv3d_acc_rd_addr, conv1_acc_rd_addr;

    /* Per-engine RES interface (packed MAX_COUT_PAR-wide; per-lane write enable) */
    logic [MAX_COUT_PAR-1:0]              conv3d_res_en,   conv1_res_en;
    logic [DEPTH_BITS-1:0]                conv3d_res_addr, conv1_res_addr;
    logic signed [MAX_COUT_PAR*N_BITS-1:0] conv3d_res_data, conv1_res_data;

    /* Muxed conv RES output — vectorized per-Cout-lane.
       The selected engine drives all MAX_COUT_PAR lanes; lanes beyond
       cout_par_active have res_en=0 (gated inside conv3d/conv1d). */
    logic [MAX_COUT_PAR-1:0]                          conv_res_en;
    logic [DEPTH_BITS-1:0]                            conv_res_addr;
    logic signed [MAX_COUT_PAR-1:0][N_BITS-1:0]       conv_res_data;

    /* Muxed ACC URAM signals (packed MAX_COUT_PAR-wide) */
    logic                                 acc_wr_en;
    logic [DEPTH_BITS-1:0]                acc_wr_addr;
    logic signed [ACC_DATA_W-1:0]         acc_wr_data;
    logic                                 acc_rd_en;
    logic [DEPTH_BITS-1:0]                acc_rd_addr;
    logic signed [ACC_DATA_W-1:0]         acc_rd_data;

    assign in_buf_rd_addr = is_conv1 ? conv1_pixel_addr : conv3d_pixel_addr;
    assign in_buf_rd_en   = is_conv1 ? conv1_pixel_en   : conv3d_pixel_en;

    assign acc_wr_en   = is_conv1 ? conv1_acc_wr_en   : conv3d_acc_wr_en;
    assign acc_wr_addr = is_conv1 ? conv1_acc_wr_addr  : conv3d_acc_wr_addr;
    assign acc_wr_data = is_conv1 ? conv1_acc_wr_data  : conv3d_acc_wr_data;
    assign acc_rd_en   = is_conv1 ? conv1_acc_rd_en    : conv3d_acc_rd_en;
    assign acc_rd_addr = is_conv1 ? conv1_acc_rd_addr  : conv3d_acc_rd_addr;

    assign conv_res_addr = is_conv1 ? conv1_res_addr : conv3d_res_addr;
    generate for (genvar c = 0; c < MAX_COUT_PAR; c = c + 1) begin : g_conv_res_mux
        assign conv_res_en[c]   = is_conv1 ? conv1_res_en[c]
                                           : conv3d_res_en[c];
        assign conv_res_data[c] = is_conv1 ? conv1_res_data[c*N_BITS +: N_BITS]
                                           : conv3d_res_data[c*N_BITS +: N_BITS];
    end endgenerate

    /* ================================================================
     *  Next State Logic
     * ================================================================ */
    always_comb begin : next_state_logic
        next_state = state;
        case (state)
            S_IDLE:
                if (start)
                    next_state = S_LOAD;
            S_LOAD:
                if (load_cnt >= wt_words - 1)
                    next_state = S_COMPUTE;
            S_COMPUTE:
                if (compute_done)
                    next_state = S_NEXT_CHOUT;
            S_NEXT_CHOUT:
                if (ch_out + {3'd0, cout_par_active} < r_cfg.cout)
                    next_state = S_LOAD;
                else
                    next_state = S_NEXT_LAYER;
            S_NEXT_LAYER:
                if (next_layer_idx < max_layers_run)
                    next_state = S_LOAD;
                else
                    next_state = S_IDLE;
            default:
                next_state = S_IDLE;
        endcase
    end

    /* ================================================================
     *  State Memory
     *
     *  Synchronous reset: see UG901 p. 85, UG949 pp. 50-55.
     *  FDRE primitives support sync reset natively and avoid the
     *  [Synth 8-7137] "Set and reset with same priority" warning
     *  class that async-sensitivity-list blocks are prone to when
     *  some registers are assigned outside the reset branch.
     * ================================================================ */
    always_ff @(posedge aclk) begin : state_memory
        if (!aresetn)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    /* ================================================================
     *  Counters
     *
     *  Synchronous reset (see state_memory header). ch_out and
     *  layer_idx are explicitly listed in the reset branch — without
     *  them, [Synth 8-7137] fires because they are assigned only
     *  inside case branches and Vivado cannot map the implicit hold
     *  to a single FDRE/FDSE primitive.
     * ================================================================ */
    always_ff @(posedge aclk) begin : counters
        if (!aresetn) begin
            load_cnt       <= '0;
            preload_cnt    <= '0;
            wt_addr_reg    <= '0;
            round_loaded   <= '0;
            r_layer_idx    <= '0;
            preload_active <= 1'b0;
            preload_done   <= 1'b0;
            ch_out         <= 8'd0;
            layer_idx      <= 5'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        r_layer_idx    <= 5'd0;
                        ch_out         <= 8'd0;
                        layer_idx      <= 5'd0;
                        wt_addr_reg    <= LAYER_CFG[0].wt_base + 1;
                        load_cnt       <= 4'd0;
                        round_loaded   <= 0;
                        preload_active <= 1'b0;
                        preload_done   <= 1'b0;
                    end
                end

                S_LOAD: begin
                    if (load_cnt < wt_words - 1) begin
                        wt_addr_reg <= wt_addr_reg + 1;
                        load_cnt    <= load_cnt + 1;
                    end else begin
                        round_loaded <= round_loaded + 1;
                        if (round_loaded + 1 < rt_total_rounds) begin
                            preload_active <= 1'b1;
                            preload_cnt    <= 4'd0;
                            preload_done   <= 1'b0;
                        end
                    end
                end

                S_COMPUTE: begin
                    /* Preload counter management — must come BEFORE compute_req_wt
                       so that compute_req_wt's preload_active<=1 wins when both
                       preload completion and weight request fire on the same cycle. */
                    if (preload_active && !preload_done) begin
                        if (preload_cnt < wt_words) begin
                            wt_addr_reg <= wt_addr_reg + 1;
                            preload_cnt <= preload_cnt + 1;
                        end else begin
                            preload_done   <= 1'b1;
                            preload_active <= 1'b0;
                            round_loaded   <= round_loaded + 1;
                        end
                    end

                    if (compute_req_wt) begin
                        if (round_loaded < rt_total_rounds) begin
                            preload_active <= 1'b1;
                            preload_cnt    <= 4'd0;
                            preload_done   <= 1'b0;
                        end
                    end
                end

                S_NEXT_CHOUT: begin
                    if (ch_out + {3'd0, cout_par_active} < r_cfg.cout) begin
                        ch_out         <= ch_out + {3'd0, cout_par_active};
                        round_loaded   <= 0;
                        preload_active <= 1'b0;
                        preload_done   <= 1'b0;
                        /* Each Cout-pass consumes cin_grp*wt_words ROM words
                           (regardless of how many Couts are interleaved per
                           pass).  pass_idx = next_ch_out >> log2_cout_par. */
                        wt_addr_reg    <= r_cfg.wt_base
                                       + ((ch_out + {3'd0, cout_par_active}) >> log2_cout_par)
                                         * r_cfg.cin_grp * wt_words
                                       + 1;
                        load_cnt       <= 4'd0;
                    end
                end

                S_NEXT_LAYER: begin
                    if (next_layer_idx < max_layers_run) begin
                        wt_addr_reg    <= next_layer_cfg.wt_base + 1;
                        layer_idx      <= next_layer_idx;
                        r_layer_idx    <= next_layer_idx;
                        ch_out         <= 8'd0;
                        round_loaded   <= 0;
                        preload_active <= 1'b0;
                        preload_done   <= 1'b0;
                        load_cnt       <= 4'd0;
                    end
                end

                default: ;
            endcase
        end
    end

    /* ================================================================
     *  Weight Banks
     *
     *  Synchronous reset (see state_memory header). wt_bank_a and
     *  wt_bank_b are intentionally NOT reset here — they are
     *  data-path registers initialized by GSR at bitstream load
     *  (UG949 p. 50: "resets are generally less necessary on the
     *  data path logic"). They are fully written in S_LOAD before
     *  being read in S_COMPUTE, so their post-reset contents do
     *  not matter.
     * ================================================================ */
    always_ff @(posedge aclk) begin : weight_banks
        if (!aresetn) begin
            wt_sel <= 1'b0;
        end else begin
            case (state)
                S_LOAD: begin
                    /* Latch weight row into shadow bank */
                    for (int c = 0; c < C_PAR; c++) begin
                        if (wt_sel)   // shadow = A
                            wt_bank_a[load_cnt][c] <= $signed(wt_mem_dout_b[c*N_BITS +: N_BITS]);
                        else          // shadow = B
                            wt_bank_b[load_cnt][c] <= $signed(wt_mem_dout_b[c*N_BITS +: N_BITS]);
                    end

                    /* Swap active bank on load complete */
                    if (load_cnt >= wt_words - 1)
                        wt_sel <= ~wt_sel;
                end

                S_COMPUTE: begin
                    /* Background preload writes into shadow bank */
                    if (preload_active && !preload_done) begin
                        if (preload_cnt > 0) begin
                            for (int c = 0; c < C_PAR; c++) begin
                                if (wt_sel)   // shadow = A (active = B)
                                    wt_bank_a[preload_cnt - 1][c] <= $signed(wt_mem_dout_b[c*N_BITS +: N_BITS]);
                                else          // shadow = B (active = A)
                                    wt_bank_b[preload_cnt - 1][c] <= $signed(wt_mem_dout_b[c*N_BITS +: N_BITS]);
                            end
                        end
                    end

                    /* Swap on weight request from compute engine */
                    if (compute_req_wt)
                        wt_sel <= ~wt_sel;
                end

                default: ;
            endcase
        end
    end

    /* ================================================================
     *  Shadow Registers (layer config + quantization parameters)
     * ================================================================ */
    always_ff @(posedge aclk) begin : shadow_registers
        case (state)
            S_IDLE: begin
                if (start)
                    r_cfg <= LAYER_CFG[0];
            end

            S_LOAD: begin
                /* Sequential per-lane QP latch.  Cycle k of S_LOAD captures
                   qp_mem_dout_b for lane k (where k = load_cnt < cout_par_active).
                   For cout_par_active=1, only lane 0 is latched on cycle 0 —
                   identical to legacy scalar behavior. */
                if (load_cnt < cout_par_active) begin
                    r_bias[load_cnt[2:0]]   <= $signed(qp_mem_dout_b[31:0]);
                    r_m0[load_cnt[2:0]]     <= qp_mem_dout_b[63:32];
                    r_nshift[load_cnt[2:0]] <= qp_mem_dout_b[69:64];
                end
            end

            S_NEXT_LAYER: begin
                if (next_layer_idx < max_layers_run) begin
                    r_cfg <= next_layer_cfg;
                end
            end

            default: ;
        endcase
    end

    /* ================================================================
     *  Output Registers (single-cycle pulses)
     *
     *  Synchronous reset (see state_memory header).
     * ================================================================ */
    always_ff @(posedge aclk) begin : output_registers
        if (!aresetn) begin
            conv3d_start         <= 1'b0;
            conv1_start          <= 1'b0;
            conv3d_weights_ready <= 1'b0;
            conv1_weights_ready  <= 1'b0;
            done                 <= 1'b0;
        end else begin
            conv3d_start         <= 1'b0;
            conv1_start          <= 1'b0;
            conv3d_weights_ready <= 1'b0;
            conv1_weights_ready  <= 1'b0;
            done                 <= 1'b0;

            case (state)
                S_LOAD: begin
                    if (load_cnt >= wt_words - 1) begin
                        if (!is_conv1)
                            conv3d_start <= 1'b1;
                        else
                            conv1_start  <= 1'b1;
                    end
                end

                S_COMPUTE: begin
                    if (compute_req_wt) begin
                        if (!is_conv1)
                            conv3d_weights_ready <= 1'b1;
                        else
                            conv1_weights_ready  <= 1'b1;
                    end
                end

                S_NEXT_LAYER: begin
                    if (!(next_layer_idx < max_layers_run))
                        done <= 1'b1;
                end

                default: ;
            endcase
        end
    end

    /* ================================================================
     *  Combinational ROM Drive
     * ================================================================ */
    always_comb begin : rom_drive
        /* XSim workaround: LAYER_CFG[variable].field silently returns 0.
           Assigning the whole struct to a local first works around this. */
        layer_cfg_t next_layer_cfg;
        next_layer_cfg = LAYER_CFG[next_layer_idx];

        /* Defaults -- ROM ports idle */
        wt_mem_en_b   = 1'b0;
        wt_mem_addr_b = '0;
        qp_mem_en_b   = 1'b0;
        qp_mem_addr_b = '0;

        case (state)
            S_IDLE: begin
                if (start) begin
                    /* Drive QP + first weight address simultaneously */
                    qp_mem_en_b   = 1'b1;
                    qp_mem_addr_b = LAYER_CFG[0].qp_base;
                    wt_mem_en_b   = 1'b1;
                    wt_mem_addr_b = LAYER_CFG[0].wt_base;
                end
            end

            S_LOAD: begin
                if (load_cnt < wt_words - 1) begin
                    wt_mem_en_b   = 1'b1;
                    wt_mem_addr_b = wt_addr_reg;
                end
                /* Sequential QP fetch for lanes 1..cout_par_active-1.
                   Cycle k of S_LOAD requests qp_addr = qp_base + ch_out + k + 1,
                   so the data lands at cycle k+1 to be latched into lane k+1.
                   For cout_par_active=1 this never fires (load_cnt < 0). */
                if (load_cnt < cout_par_active - 1) begin
                    qp_mem_en_b   = 1'b1;
                    qp_mem_addr_b = r_cfg.qp_base + ch_out + {6'd0, load_cnt} + 10'd1;
                end
            end

            S_COMPUTE: begin
                if (preload_active && !preload_done && preload_cnt < wt_words) begin
                    wt_mem_en_b   = 1'b1;
                    wt_mem_addr_b = wt_addr_reg;
                end
            end

            S_NEXT_CHOUT: begin
                if (ch_out + {3'd0, cout_par_active} < r_cfg.cout) begin
                    /* Drive QP addr for the FIRST lane of the next pass.
                       S_LOAD then drives addrs for lanes 1..cout_par_active-1
                       on its own cycles 0..cout_par_active-2. */
                    qp_mem_en_b   = 1'b1;
                    qp_mem_addr_b = r_cfg.qp_base + ch_out + {5'd0, cout_par_active};
                    wt_mem_en_b   = 1'b1;
                    /* First word of next pass: same pass_idx shift as wt_addr_reg
                       above so the layout matches generate_hdl_weights.py packing. */
                    wt_mem_addr_b = r_cfg.wt_base
                                 + ((ch_out + {3'd0, cout_par_active}) >> log2_cout_par)
                                   * r_cfg.cin_grp * wt_words;
                end
            end

            S_NEXT_LAYER: begin
                if (next_layer_idx < max_layers_run) begin
                    qp_mem_en_b   = 1'b1;
                    qp_mem_addr_b = next_layer_cfg.qp_base;
                    wt_mem_en_b   = 1'b1;
                    wt_mem_addr_b = next_layer_cfg.wt_base;
                end
            end

            default: ;
        endcase
    end

    /* ================================================================
     *  Current Layer Info
     *
     *  curr_layer_idx, curr_pp_buf_sel, curr_pp_rd_offset are exposed
     *  to inference_top (status + URAM routing).
     *  The remainder are internal — consumed by the activation, max_pool,
     *  and RMW writer stages instantiated below.
     * ================================================================ */
    logic [1:0]  curr_layer_type;
    logic [8:0]  curr_act_size;
    logic [7:0]  curr_ch_out;
    logic [13:0] curr_pp_wr_offset;

    assign curr_layer_type   = r_cfg.layer_type;
    assign curr_layer_idx    = r_layer_idx;
    assign curr_act_size     = r_cfg.h_in;
    assign curr_ch_out       = ch_out;
    assign curr_pp_buf_sel   = r_cfg.pp_buf_sel;
    assign curr_pp_rd_offset = r_cfg.pp_rd_offset;
    assign curr_pp_wr_offset = r_cfg.pp_wr_offset;

    /* ================================================================
     *  Conv3d Instance (K=3 convolution)
     * ================================================================ */
    /* Per-lane QP vectors are latched directly into packed r_bias/r_m0/r_nshift
       by the sequential S_LOAD fetch.  Lane packing matches conv3d/conv1d's
       expected bit layout: lane c occupies bits [c*W +: W]. */
    wire signed [MAX_COUT_PAR*ACC_BITS-1:0] bias_packed   = r_bias;
    wire signed [MAX_COUT_PAR*32-1:0]       m0_packed     = r_m0;
    wire        [MAX_COUT_PAR*6-1:0]        nshift_packed = r_nshift;

    conv3d #(
        .K            (K),
        .STRIDE       (1),
        .C_PAR        (C_PAR),
        .MAX_COUT_PAR (MAX_COUT_PAR),
        .N_BITS       (N_BITS),
        .ACC_BITS     (ACC_BITS),
        .M0_BITS      (32),
        .SHIFT_BITS   (6),
        .DEPTH_BITS   (DEPTH_BITS)
    ) u_conv3d (
        .clk                  (aclk),
        .rst                  (!aresetn),
        .start                (conv3d_start),
        .act_size             (rt_act_size),
        .cin                  (rt_cin),
        .log2_cin_group_size  (r_cfg.log2_cin_group_size),
        .zp_in                (r_cfg.zp_in),
        .zp_out               (r_cfg.zp_out),
        .bias                 (bias_packed),
        .m0                   (m0_packed),
        .n_shift              (nshift_packed),
        .pixel_bram_addr      (conv3d_pixel_addr),
        .pixel_bram_en        (conv3d_pixel_en),
        .pixel_bram_data      (in_buf_rd_data),
        .weights_all_channels (weights_flat),
        .ACC_write_en         (conv3d_acc_wr_en),
        .ACC_write_address    (conv3d_acc_wr_addr),
        .ACC_write_data_in    (conv3d_acc_wr_data),
        .ACC_read_en          (conv3d_acc_rd_en),
        .ACC_read_address     (conv3d_acc_rd_addr),
        .ACC_read_data_out    (acc_rd_data),
        .RES_write_en         (conv3d_res_en),
        .RES_write_address    (conv3d_res_addr),
        .RES_write_data_in    (conv3d_res_data),
        .req_weights          (conv3d_req_weights),
        .weights_ready        (conv3d_weights_ready),
        .done                 (conv3d_done)
    );

    /* ================================================================
     *  Conv1x1 Instance (K=1 pointwise convolution)
     * ================================================================ */
    conv1d #(
        .C_PAR        (C_PAR),
        .MAX_COUT_PAR (MAX_COUT_PAR),
        .N_BITS       (N_BITS),
        .ACC_BITS     (ACC_BITS),
        .M0_BITS      (32),
        .SHIFT_BITS   (6),
        .DEPTH_BITS   (DEPTH_BITS)
    ) u_conv1d (
        .clk                  (aclk),
        .rst                  (!aresetn),
        .start                (conv1_start),
        .act_size             (rt_act_size),
        .cin                  (rt_cin),
        .zp_in                (r_cfg.zp_in),
        .zp_out               (r_cfg.zp_out),
        .bias                 (bias_packed),
        .m0                   (m0_packed),
        .n_shift              (nshift_packed),
        .pixel_bram_addr      (conv1_pixel_addr),
        .pixel_bram_en        (conv1_pixel_en),
        .pixel_bram_data      (in_buf_rd_data),
        .weights_all_channels (weights_1x1),
        .ACC_write_en         (conv1_acc_wr_en),
        .ACC_write_address    (conv1_acc_wr_addr),
        .ACC_write_data_in    (conv1_acc_wr_data),
        .ACC_read_en          (conv1_acc_rd_en),
        .ACC_read_address     (conv1_acc_rd_addr),
        .ACC_read_data_out    (acc_rd_data),
        .RES_write_en         (conv1_res_en),
        .RES_write_address    (conv1_res_addr),
        .RES_write_data_in    (conv1_res_data),
        .req_weights          (conv1_req_weights),
        .weights_ready        (conv1_weights_ready),
        .done                 (conv1_done)
    );

    /* ================================================================
     *  ACC Scratch BRAM (intermediate accumulation across rounds)
     *
     *  Depth sized for the largest multi-round layer: layer 4 at
     *  64x64 = 4096 entries.  Single-round layers (Cin <= C_PARALLEL)
     *  write to ACC but never read back (quantizer uses the
     *  combinational wire), so address aliasing is harmless.
     * ================================================================ */
    localparam ACC_DEPTH  = 4096;
    localparam ACC_ADDR_W = $clog2(ACC_DEPTH);  // 12

    sdp_ram #(
        .DATA_WIDTH (ACC_DATA_W),    // MAX_COUT_PAR * ACC_BITS (128 for MAX_COUT_PAR=4)
        .DEPTH      (ACC_DEPTH),
        .RAM_STYLE  ("ultra")         // URAM: frees ~4 BRAM36; packed 4 x 32-bit per word
    ) u_acc_mem (
        .clk    (aclk),
        .en_a   (acc_wr_en),
        .we_a   (acc_wr_en),
        .addr_a (acc_wr_addr[ACC_ADDR_W-1:0]),
        .din_a  (acc_wr_data),
        .en_b   (acc_rd_en),
        .addr_b (acc_rd_addr[ACC_ADDR_W-1:0]),
        .dout_b (acc_rd_data)
    );

    /* ================================================================
     *  Activation + Max-Pool Stages — replicated MAX_COUT_PAR× per Cout lane
     *
     *  Each lane gets its own activation (with private silu_mem replica)
     *  and max_pool instance. Lanes beyond cout_par_active receive
     *  conv_res_en=0 from conv3d/conv1d and therefore produce no output —
     *  behavior is bit-identical to the legacy scalar path when
     *  cout_par_active=1.
     * ================================================================ */
    logic [MAX_COUT_PAR-1:0]                          act_out_valid;
    logic [MAX_COUT_PAR-1:0][DEPTH_BITS-1:0]          act_out_addr;
    logic signed [MAX_COUT_PAR-1:0][N_BITS-1:0]       act_out_data;

    logic [MAX_COUT_PAR-1:0]                          pool_out_valid;
    logic [MAX_COUT_PAR-1:0][DEPTH_BITS-1:0]          pool_out_addr;
    logic signed [MAX_COUT_PAR-1:0][N_BITS-1:0]       pool_out_data;

    generate for (genvar c = 0; c < MAX_COUT_PAR; c = c + 1) begin : g_downstream
        activation #(
            .N_BITS     (N_BITS),
            .DEPTH_BITS (DEPTH_BITS),
            .LUT_ADDR_W (LUT_ADDR_W)
        ) u_activation (
            .clk        (aclk),
            .rst_n      (aresetn),
            .layer_type (curr_layer_type),
            .layer_idx  (curr_layer_idx),
            .in_valid   (conv_res_en[c]),
            .in_addr    (conv_res_addr),
            .in_data    (conv_res_data[c]),
            .lut_en     (silu_mem_en_b[c]),
            .lut_addr   (silu_mem_addr_b[c]),
            .lut_rdata  (silu_mem_dout_b[c]),
            .out_valid  (act_out_valid[c]),
            .out_addr   (act_out_addr[c]),
            .out_data   (act_out_data[c])
        );

        max_pool #(
            .N_BITS     (N_BITS),
            .DEPTH_BITS (DEPTH_BITS)
        ) u_max_pool (
            .clk        (aclk),
            .rst_n      (aresetn),
            .layer_type (curr_layer_type),
            .act_size   (curr_act_size),
            .in_valid   (act_out_valid[c]),
            .in_addr    (act_out_addr[c]),
            .in_data    (act_out_data[c]),
            .out_valid  (pool_out_valid[c]),
            .out_addr   (pool_out_addr[c]),
            .out_data   (pool_out_data[c])
        );
    end endgenerate

    /* ================================================================
     *  RMW Output Writer
     *
     *  Splices the per-pixel pool output (8-bit) into the appropriate
     *  byte lane of a 128-bit URAM word, writing back via the
     *  out_buf_wr_* port.  Read-back of the existing word comes from
     *  out_buf_rd_data.  inference_top routes both ports to whichever
     *  fmap_a/b URAM is the current output buffer.
     * ================================================================ */
    /* Per-lane RMW stage-0 — each Cout lane registers its own valid/data.
       All lanes share a single URAM word address (same spatial pixel).
       Lane 0 drives the shared addr/byte_pos_base (h_out/w_out identical
       across lanes in lockstep). */
    logic [MAX_COUT_PAR-1:0]                          rmw_s0_valid;
    logic [FMAP_ADDR_W-1:0]                           rmw_s0_addr;
    logic signed [MAX_COUT_PAR-1:0][N_BITS-1:0]       rmw_s0_data;
    logic [3:0]                                       rmw_s0_byte_pos_base;

    wire [8:0] h_out = (curr_layer_type == CONV3_POOL) ? (curr_act_size >> 1) : curr_act_size;
    wire [FMAP_ADDR_W-1:0] rmw_base_addr = curr_pp_wr_offset + (curr_ch_out >> 4) * h_out * h_out;

    /* Synchronous reset (see state_memory header). */
    always_ff @(posedge aclk) begin : rmw_s0_pipeline
        if (!aresetn) begin
            rmw_s0_valid <= '0;
        end else begin
            for (int c = 0; c < MAX_COUT_PAR; c++) begin
                rmw_s0_valid[c] <= pool_out_valid[c];
                rmw_s0_data[c]  <= pool_out_data[c];
            end
            rmw_s0_addr          <= rmw_base_addr + pool_out_addr[0][FMAP_ADDR_W-1:0];
            rmw_s0_byte_pos_base <= curr_ch_out[3:0];
        end
    end

    /* Read-back port — fires whenever any lane produces output. */
    assign out_buf_rd_en   = |pool_out_valid;
    assign out_buf_rd_addr = rmw_base_addr + pool_out_addr[0][FMAP_ADDR_W-1:0];

    /* Multi-byte splice: each active lane c writes its byte at
       (byte_pos_base + c). When byte_pos_base == 0 the word starts fresh
       (no read-back dependency), matching the legacy semantics where the
       first byte into a word zero-initializes the rest. */
    logic [FMAP_DATA_W-1:0] spliced_word;
    always_comb begin
        spliced_word = (rmw_s0_byte_pos_base == 4'd0) ? {FMAP_DATA_W{1'b0}} : out_buf_rd_data;
        for (int c = 0; c < MAX_COUT_PAR; c++) begin
            if (rmw_s0_valid[c]) begin
                /* 4-bit (byte_pos_base) + 3-bit lane index = 5-bit byte index,
                   bounded by cout_par_active placement so it never exceeds 15. */
                spliced_word[({1'b0, rmw_s0_byte_pos_base} + 5'(c)) * 8 +: 8] = rmw_s0_data[c];
            end
        end
    end

    /* Synchronous reset (see state_memory header). */
    always_ff @(posedge aclk) begin : rmw_write_pipeline
        if (!aresetn) begin
            out_buf_wr_en <= 1'b0;
        end else begin
            out_buf_wr_en   <= |rmw_s0_valid;
            out_buf_wr_addr <= rmw_s0_addr;
            out_buf_wr_data <= spliced_word;
        end
    end

    /* ================================================================
     *  Debug Capture — first 4 conv RES / out_buf writes during layer 0
     *
     *  Two capture points, independently indexed:
     *    conv_res_*: captures the first 4 conv3d/conv1d RES_write_address
     *                values (mux output) while curr_layer_idx == 0 AND
     *                conv_res_en == 1.
     *    out_wr_*:   captures the first 4 out_buf_wr_addr values while
     *                curr_layer_idx == 0 AND out_buf_wr_en == 1.
     *
     *  These let us pinpoint where the silicon-only +1 URAM shift is
     *  introduced. Expected values in both capture sets for layer 0:
     *    0, 1, 2, 3  (strictly correct, each cycle increments by 1)
     *
     *  NOTE: the conv_res capture is at the PRE-activation/pool/rmw
     *  output of conv3d (before the 2x maxpool), so for layer 0
     *  (CONV3_POOL, act_size=256) it counts 0, 1, 2, ... in conv space.
     *  The out_wr capture is post-rmw so it counts 0, 1, 2, ... in
     *  pool space (after max_pool's divide-by-4).
     * ================================================================ */
    logic [1:0] conv_res_cap_idx;
    logic [1:0] pool_cap_idx;
    logic [1:0] rmw_s0_cap_idx;
    logic [1:0] out_wr_cap_idx;
    logic       conv_res_done;
    logic       pool_done;
    logic       rmw_s0_done;
    logic       out_wr_done;

    always_ff @(posedge aclk or negedge aresetn) begin : dbg_capture
        if (!aresetn) begin
            dbg_conv_res_addr_0 <= '0;
            dbg_conv_res_addr_1 <= '0;
            dbg_conv_res_addr_2 <= '0;
            dbg_conv_res_addr_3 <= '0;
            dbg_pool_out_addr_0 <= '0;
            dbg_pool_out_addr_1 <= '0;
            dbg_pool_out_addr_2 <= '0;
            dbg_pool_out_addr_3 <= '0;
            dbg_rmw_s0_addr_0   <= '0;
            dbg_rmw_s0_addr_1   <= '0;
            dbg_rmw_s0_addr_2   <= '0;
            dbg_rmw_s0_addr_3   <= '0;
            dbg_rmw_base_0      <= '0;
            dbg_rmw_base_1      <= '0;
            dbg_rmw_base_2      <= '0;
            dbg_rmw_base_3      <= '0;
            dbg_pp_wr_offset    <= '0;
            dbg_ch_out          <= '0;
            dbg_h_out           <= '0;
            dbg_out_wr_addr_0   <= '0;
            dbg_out_wr_addr_1   <= '0;
            dbg_out_wr_addr_2   <= '0;
            dbg_out_wr_addr_3   <= '0;
            conv_res_cap_idx    <= 2'd0;
            pool_cap_idx        <= 2'd0;
            rmw_s0_cap_idx      <= 2'd0;
            out_wr_cap_idx      <= 2'd0;
            conv_res_done       <= 1'b0;
            pool_done           <= 1'b0;
            rmw_s0_done         <= 1'b0;
            out_wr_done         <= 1'b0;
        end else if (start) begin
            // Reset on new inference start
            conv_res_cap_idx <= 2'd0;
            pool_cap_idx     <= 2'd0;
            rmw_s0_cap_idx   <= 2'd0;
            out_wr_cap_idx   <= 2'd0;
            conv_res_done    <= 1'b0;
            pool_done        <= 1'b0;
            rmw_s0_done      <= 1'b0;
            out_wr_done      <= 1'b0;
        end else begin
            /* Layer-0 debug captures use lane 0 only — bit-identical to the
               legacy scalar pipeline when cout_par_active=1.  When cout_par
               engages later, these still show lane 0's behavior. */
            if (conv_res_en[0] && !conv_res_done && (curr_layer_idx == 5'd0)) begin
                case (conv_res_cap_idx)
                    2'd0: dbg_conv_res_addr_0 <= conv_res_addr;
                    2'd1: dbg_conv_res_addr_1 <= conv_res_addr;
                    2'd2: dbg_conv_res_addr_2 <= conv_res_addr;
                    2'd3: dbg_conv_res_addr_3 <= conv_res_addr;
                endcase
                if (conv_res_cap_idx == 2'd3)
                    conv_res_done <= 1'b1;
                else
                    conv_res_cap_idx <= conv_res_cap_idx + 2'd1;
            end

            // Capture pool_out_addr (layer 0 only, first 4)
            if (pool_out_valid[0] && !pool_done && (curr_layer_idx == 5'd0)) begin
                case (pool_cap_idx)
                    2'd0: dbg_pool_out_addr_0 <= pool_out_addr[0][13:0];
                    2'd1: dbg_pool_out_addr_1 <= pool_out_addr[0][13:0];
                    2'd2: dbg_pool_out_addr_2 <= pool_out_addr[0][13:0];
                    2'd3: dbg_pool_out_addr_3 <= pool_out_addr[0][13:0];
                endcase
                if (pool_cap_idx == 2'd3)
                    pool_done <= 1'b1;
                else
                    pool_cap_idx <= pool_cap_idx + 2'd1;
            end

            // Capture rmw_s0_addr + rmw_base_addr at rmw_s0_valid transitions.
            // Also snapshot pp_wr_offset/ch_out/h_out at the first capture
            // so we can see what the buggy computation inputs are.
            if (rmw_s0_valid[0] && !rmw_s0_done && (curr_layer_idx == 5'd0)) begin
                case (rmw_s0_cap_idx)
                    2'd0: begin
                        dbg_rmw_s0_addr_0 <= rmw_s0_addr;
                        dbg_rmw_base_0    <= rmw_base_addr;
                        dbg_pp_wr_offset  <= curr_pp_wr_offset;
                        dbg_ch_out        <= curr_ch_out;
                        dbg_h_out         <= h_out;
                    end
                    2'd1: begin
                        dbg_rmw_s0_addr_1 <= rmw_s0_addr;
                        dbg_rmw_base_1    <= rmw_base_addr;
                    end
                    2'd2: begin
                        dbg_rmw_s0_addr_2 <= rmw_s0_addr;
                        dbg_rmw_base_2    <= rmw_base_addr;
                    end
                    2'd3: begin
                        dbg_rmw_s0_addr_3 <= rmw_s0_addr;
                        dbg_rmw_base_3    <= rmw_base_addr;
                    end
                endcase
                if (rmw_s0_cap_idx == 2'd3)
                    rmw_s0_done <= 1'b1;
                else
                    rmw_s0_cap_idx <= rmw_s0_cap_idx + 2'd1;
            end

            // Capture out_buf_wr_addr (post-rmw, layer 0 only, first 4)
            if (out_buf_wr_en && !out_wr_done && (curr_layer_idx == 5'd0)) begin
                case (out_wr_cap_idx)
                    2'd0: dbg_out_wr_addr_0 <= out_buf_wr_addr;
                    2'd1: dbg_out_wr_addr_1 <= out_buf_wr_addr;
                    2'd2: dbg_out_wr_addr_2 <= out_buf_wr_addr;
                    2'd3: dbg_out_wr_addr_3 <= out_buf_wr_addr;
                endcase
                if (out_wr_cap_idx == 2'd3)
                    out_wr_done <= 1'b1;
                else
                    out_wr_cap_idx <= out_wr_cap_idx + 2'd1;
            end
        end
    end

    assign dbg_capture_count = {out_wr_cap_idx, conv_res_cap_idx};

endmodule
