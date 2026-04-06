`timescale 1ns / 1ps
`include "layer_config.svh"

module inference_hdl #(
    parameter MAX_PARALLEL = C_PARALLEL,
    parameter K            = 3,
    parameter N_BITS       = 8,
    parameter ACC_BITS     = 32,
    parameter DEPTH_BITS   = 16,
    parameter KSQ          = K * K,
    parameter WT_DATA_W    = MAX_PARALLEL * N_BITS,
    parameter WT_ADDR_W    = $clog2(32768),
    parameter QP_ADDR_W    = $clog2(1024)
)(
    input  logic                             aclk,
    input  logic                             aresetn,
    input  logic                             start,
    output logic                             done,

    /* Weight ROM read port (top.sv u_wt_mem port B) */
    output logic                             wt_mem_en_b,
    output logic [WT_ADDR_W-1:0]             wt_mem_addr_b,
    input  logic [WT_DATA_W-1:0]             wt_mem_dout_b,

    /* QP ROM read port (top.sv u_qp_mem port B) */
    output logic                             qp_mem_en_b,
    output logic [QP_ADDR_W-1:0]             qp_mem_addr_b,
    input  logic [71:0]                      qp_mem_dout_b,

    /* Pixel BRAM interface (testbench drives) */
    output logic [DEPTH_BITS-1:0]            pixel_bram_addr,
    output logic                             pixel_bram_en,
    input  logic [MAX_PARALLEL*N_BITS-1:0]   pixel_bram_data,

    /* RES output interface (testbench captures) */
    output logic                             res_write_en,
    output logic [DEPTH_BITS-1:0]            res_write_addr,
    output logic signed [N_BITS-1:0]         res_write_data,

    /* Current layer info (for activation + pool stages) */
    output logic [1:0]                       curr_layer_type,
    output logic [4:0]                       curr_layer_idx,
    output logic [8:0]                       curr_act_size,
    output logic [7:0]                       curr_ch_out,

    /* Sub-pingpong config (for top.sv URAM routing) */
    output logic                             curr_pp_buf_sel,
    output logic [13:0]                      curr_pp_rd_offset,
    output logic [13:0]                      curr_pp_wr_offset
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
    logic signed [31:0] r_bias;
    logic        [31:0] r_m0;
    logic         [5:0] r_nshift;

    /* ================================================================
     *  Runtime Layer-Type Derived Signals
     * ================================================================ */
    wire        is_conv1  = r_cfg.layer_type[1];   // 1 for CONV1/CONV1_LIN
    wire  [3:0] wt_words  = is_conv1 ? 4'd1 : 4'd9; // weight entries per (ch_out, cin_group)

    /* ================================================================
     *  Double-Buffered Weight Register File
     * ================================================================ */

    /* Two banks: 9 rows x 16 cols x 8 bits each (CONV1 uses only row 0) */
    logic signed [N_BITS-1:0] wt_bank_a [0:KSQ-1][0:MAX_PARALLEL-1];
    logic signed [N_BITS-1:0] wt_bank_b [0:KSQ-1][0:MAX_PARALLEL-1];
    logic                     wt_sel;   // 0 = bank A active, 1 = bank B active

    /* Transpose + mux: ROM [kp][channel] -> conv3d [channel][kp] */
    logic [MAX_PARALLEL*KSQ*N_BITS-1:0] weights_flat;

    generate
        for (genvar slot = 0; slot < MAX_PARALLEL; slot++) begin : g_slot
            for (genvar kp = 0; kp < KSQ; kp++) begin : g_kp
                assign weights_flat[slot*KSQ*N_BITS + kp*N_BITS +: N_BITS] =
                    wt_sel ? wt_bank_b[kp][slot] : wt_bank_a[kp][slot];
            end
        end
    endgenerate

    /* Conv1x1 weight bus: only row 0 of each bank (K=1, one weight per channel) */
    logic [MAX_PARALLEL*N_BITS-1:0] weights_1x1;

    generate
        for (genvar s = 0; s < MAX_PARALLEL; s++) begin : g_w1
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

    /* Per-engine ACC interface */
    logic                      conv3d_acc_wr_en,   conv1_acc_wr_en;
    logic [DEPTH_BITS-1:0]     conv3d_acc_wr_addr, conv1_acc_wr_addr;
    logic signed [ACC_BITS-1:0] conv3d_acc_wr_data, conv1_acc_wr_data;
    logic                      conv3d_acc_rd_en,   conv1_acc_rd_en;
    logic [DEPTH_BITS-1:0]     conv3d_acc_rd_addr, conv1_acc_rd_addr;

    /* Per-engine RES interface */
    logic                      conv3d_res_en,   conv1_res_en;
    logic [DEPTH_BITS-1:0]     conv3d_res_addr, conv1_res_addr;
    logic signed [N_BITS-1:0]  conv3d_res_data, conv1_res_data;

    /* Muxed ACC BRAM signals */
    logic                      acc_wr_en;
    logic [DEPTH_BITS-1:0]     acc_wr_addr;
    logic signed [ACC_BITS-1:0] acc_wr_data;
    logic                      acc_rd_en;
    logic [DEPTH_BITS-1:0]     acc_rd_addr;
    logic signed [ACC_BITS-1:0] acc_rd_data;

    assign pixel_bram_addr = is_conv1 ? conv1_pixel_addr : conv3d_pixel_addr;
    assign pixel_bram_en   = is_conv1 ? conv1_pixel_en   : conv3d_pixel_en;

    assign acc_wr_en   = is_conv1 ? conv1_acc_wr_en   : conv3d_acc_wr_en;
    assign acc_wr_addr = is_conv1 ? conv1_acc_wr_addr  : conv3d_acc_wr_addr;
    assign acc_wr_data = is_conv1 ? conv1_acc_wr_data  : conv3d_acc_wr_data;
    assign acc_rd_en   = is_conv1 ? conv1_acc_rd_en    : conv3d_acc_rd_en;
    assign acc_rd_addr = is_conv1 ? conv1_acc_rd_addr  : conv3d_acc_rd_addr;

    assign res_write_en   = is_conv1 ? conv1_res_en   : conv3d_res_en;
    assign res_write_addr = is_conv1 ? conv1_res_addr  : conv3d_res_addr;
    assign res_write_data = is_conv1 ? conv1_res_data  : conv3d_res_data;

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
                if (ch_out + 8'd1 < r_cfg.cout)
                    next_state = S_LOAD;
                else
                    next_state = S_NEXT_LAYER;
            S_NEXT_LAYER:
                if (next_layer_idx < NUM_LAYERS)
                    next_state = S_LOAD;
                else
                    next_state = S_IDLE;
            default:
                next_state = S_IDLE;
        endcase
    end

    /* ================================================================
     *  State Memory
     * ================================================================ */
    always_ff @(posedge aclk or negedge aresetn) begin : state_memory
        if (!aresetn)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    /* ================================================================
     *  Counters
     * ================================================================ */
    always_ff @(posedge aclk or negedge aresetn) begin : counters
        if (!aresetn) begin
            load_cnt       <= '0;
            preload_cnt    <= '0;
            wt_addr_reg    <= '0;
            round_loaded   <= '0;
            r_layer_idx    <= '0;
            preload_active <= 1'b0;
            preload_done   <= 1'b0;
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
                    if (ch_out + 8'd1 < r_cfg.cout) begin
                        ch_out         <= ch_out + 8'd1;
                        round_loaded   <= 0;
                        preload_active <= 1'b0;
                        preload_done   <= 1'b0;
                        wt_addr_reg    <= r_cfg.wt_base
                                       + (ch_out + 8'd1) * r_cfg.cin_grp * wt_words
                                       + 1;
                        load_cnt       <= 4'd0;
                    end
                end

                S_NEXT_LAYER: begin
                    if (next_layer_idx < NUM_LAYERS) begin
                        layer_cfg_t _next;
                        _next = LAYER_CFG[next_layer_idx];
                        wt_addr_reg    <= _next.wt_base + 1;
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
     * ================================================================ */
    always_ff @(posedge aclk or negedge aresetn) begin : weight_banks
        if (!aresetn) begin
            wt_sel <= 1'b0;
        end else begin
            case (state)
                S_LOAD: begin
                    /* Latch weight row into shadow bank */
                    for (int c = 0; c < MAX_PARALLEL; c++) begin
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
                            for (int c = 0; c < MAX_PARALLEL; c++) begin
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
                if (load_cnt == 0) begin
                    r_bias   <= $signed(qp_mem_dout_b[31:0]);
                    r_m0     <= qp_mem_dout_b[63:32];
                    r_nshift <= qp_mem_dout_b[69:64];
                end
            end

            S_NEXT_LAYER: begin
                if (next_layer_idx < NUM_LAYERS) begin
                    layer_cfg_t _next;
                    _next = LAYER_CFG[next_layer_idx];
                    r_cfg <= _next;
                end
            end

            default: ;
        endcase
    end

    /* ================================================================
     *  Output Registers (single-cycle pulses)
     * ================================================================ */
    always_ff @(posedge aclk or negedge aresetn) begin : output_registers
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
                    if (!(next_layer_idx < NUM_LAYERS))
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
            end

            S_COMPUTE: begin
                if (preload_active && !preload_done && preload_cnt < wt_words) begin
                    wt_mem_en_b   = 1'b1;
                    wt_mem_addr_b = wt_addr_reg;
                end
            end

            S_NEXT_CHOUT: begin
                if (ch_out + 8'd1 < r_cfg.cout) begin
                    qp_mem_en_b   = 1'b1;
                    qp_mem_addr_b = r_cfg.qp_base + ch_out + 10'd1;
                    wt_mem_en_b   = 1'b1;
                    wt_mem_addr_b = r_cfg.wt_base
                                 + (ch_out + 8'd1) * r_cfg.cin_grp * wt_words;
                end
            end

            S_NEXT_LAYER: begin
                if (next_layer_idx < NUM_LAYERS) begin
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
     *  Current Layer Info (for activation stage)
     * ================================================================ */
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
    conv3d #(
        .K            (K),
        .STRIDE       (1),
        .MAX_PARALLEL (MAX_PARALLEL),
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
        .zp_in                (r_cfg.zp_in),
        .zp_out               (r_cfg.zp_out),
        .bias                 (r_bias),
        .m0                   (r_m0),
        .n_shift              (r_nshift),
        .pixel_bram_addr      (conv3d_pixel_addr),
        .pixel_bram_en        (conv3d_pixel_en),
        .pixel_bram_data      (pixel_bram_data),
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
        .MAX_PARALLEL (MAX_PARALLEL),
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
        .bias                 (r_bias),
        .m0                   (r_m0),
        .n_shift              (r_nshift),
        .pixel_bram_addr      (conv1_pixel_addr),
        .pixel_bram_en        (conv1_pixel_en),
        .pixel_bram_data      (pixel_bram_data),
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
        .DATA_WIDTH (ACC_BITS),
        .DEPTH      (ACC_DEPTH),
        .RAM_STYLE  ("block")
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

endmodule
