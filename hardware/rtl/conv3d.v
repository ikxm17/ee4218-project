`timescale 1ns / 1ps

// call on (ch_IN) convolvers and feed in zero-point padded input
// padding is fixed to value of 1
// neccessary inputs:
    // zero-point values (zp_in, zp_out) and weights + bias and rescaling factors and frame input
// return 1 INT8 feature map (rescaled and zero centered)

// streaming to conv should be:
// zp * n+2
// zp INPUT_RAM[:n] zp
// zp INPUT_RAM[n:2n] zp
// ...
// zp * n+2

module conv3d #(
    parameter K            = 3,
    parameter STRIDE       = 1,
    parameter C_PAR        = 16,
    parameter MAX_COUT_PAR = 1,
    parameter MAX_LOG2_CGS = $clog2(C_PAR),
    parameter N_BITS       = 8,
    parameter ACC_BITS     = 32,
    parameter M0_BITS      = 32,
    parameter SHIFT_BITS   = 6,
    parameter DEPTH_BITS   = 16
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               start,

    // Runtime layer dimensions
    input  wire [8:0]                         act_size,
    input  wire [7:0]                         cin,
    // Runtime Cin/Cout parallelism split: log2(cin_group_size).
    // 4 (default) = cin_group_size 16, cout_par 1 (legacy behavior).
    // 2 = cin_group_size 4,  cout_par 4 (layer 0 Cout-parallel mode).
    input  wire [2:0]                         log2_cin_group_size,

    // Quantisation scalars (per-layer, scalar)
    input  wire signed [N_BITS-1:0]                   zp_in,
    input  wire signed [N_BITS-1:0]                   zp_out,
    // Quantisation scalars (per-cout, packed MAX_COUT_PAR-wide)
    input  wire signed [MAX_COUT_PAR*ACC_BITS-1:0]    bias,
    input  wire signed [MAX_COUT_PAR*M0_BITS-1:0]     m0,
    input  wire        [MAX_COUT_PAR*SHIFT_BITS-1:0]  n_shift,

    // -------------------------------------------------------------------------
    // Pixel BRAM interface  (one BRAM that outputs C_PAR pixels per cycle)
    //   pixel_bram_addr – read address issued to input BRAM
    //   pixel_bram_en   – enb for each channel BRAM (0 on padding pixels)
    //   pixel_bram_data – registered read data from each channel BRAM
    // -------------------------------------------------------------------------
    output reg  [DEPTH_BITS-1:0]                                                pixel_bram_addr,
    output reg                                                                  pixel_bram_en,
    input  wire signed [C_PAR*N_BITS-1:0]                                pixel_bram_data,

    // Weight BRAM read ports (one per slot, addressed by global channel index)
    input  wire signed [C_PAR*K*K*N_BITS-1:0]   weights_all_channels,

    // ACC RAM interface (packed MAX_COUT_PAR-wide; 128b for MAX_COUT_PAR=4)
    // One shared enable/address; data is packed so lane c occupies bits
    // [c*ACC_BITS +: ACC_BITS].
    output reg                                       ACC_write_en,
    output reg        [DEPTH_BITS-1:0]               ACC_write_address,
    output reg signed [MAX_COUT_PAR*ACC_BITS-1:0]    ACC_write_data_in,
    output reg                                       ACC_read_en,
    output reg         [DEPTH_BITS-1:0]              ACC_read_address,
    input  signed      [MAX_COUT_PAR*ACC_BITS-1:0]   ACC_read_data_out,

    // RES output (packed MAX_COUT_PAR-wide; per-lane write enable)
    output reg [MAX_COUT_PAR-1:0]                    RES_write_en,
    output reg [DEPTH_BITS-1:0]                      RES_write_address,
    output reg signed [MAX_COUT_PAR*N_BITS-1:0]      RES_write_data_in,

    // Control signals for weight loading if rounds > 1
    output reg                              req_weights,   // signal to request next round's weights
    input  wire                              weights_ready, // signal indicating requested weights are ready

    output reg                               done
);

    // Runtime-derived constants
    wire [8:0]  pad_size       = act_size + 9'd2;
    wire [3:0]  total_rounds   = cin[7:4] + (|cin[3:0]);  // ceil(cin / 16)
    wire [4:0]  last_round_cin = (cin[3:0] == 4'd0) ? 5'd16 : {1'b0, cin[3:0]};

    // Pre-compute act_size^2 for pixel addressing (registered)
    reg [17:0] act_size_sq;
    always @(posedge clk) begin
        if (rst)
            act_size_sq <= 0;
        else if (start)
            act_size_sq <= act_size * act_size;
    end

    // Quantization parameters registered at layer start.
    // zp_in/zp_out are per-layer (scalar); bias/m0/n_shift are per-cout
    // (packed MAX_COUT_PAR-wide).
    reg signed [N_BITS-1:0]                        r_zp_in, r_zp_out;
    reg signed [MAX_COUT_PAR*ACC_BITS-1:0]         r_bias;
    reg signed [MAX_COUT_PAR*M0_BITS-1:0]          r_m0;
    reg        [MAX_COUT_PAR*SHIFT_BITS-1:0]       r_n_shift;

    // FSM
    localparam S_IDLE           = 2'd0;
    localparam S_RUNNING        = 2'd1;
    localparam S_WAIT_ROUND     = 2'd2;
    localparam S_DRAIN          = 2'd3;     // Cone A pipeline flush (post requantize fix)

    reg [1:0]                              state;
    reg [1:0]                              drain_count;        // counts to 2 in S_DRAIN
    reg [3:0]                              round;
    reg                                    conv_running;        // needed as condition to enable convolver
    reg                                    rst_convolvers;      // set after every round of convolutions

    // Input addressing and padding
    reg [8:0]                                  input_row;
    reg [8:0]                                  input_col;
    reg                                        is_padding;
    reg                                        is_padded_act; // lags one cycle behind is_padding

    // Track number of convolvers running in parallel (aka slots)
    wire [$clog2(C_PAR+1)-1:0] active_slots;
    assign active_slots = (round == total_rounds - 1)
                        ? last_round_cin[$clog2(C_PAR+1)-1:0]
                        : C_PAR[$clog2(C_PAR+1)-1:0];

    // Runtime Cin/Cout parallelism split:
    //   cin_group_size  = slots per Cout group (power of 2)
    //   cout_par_active = C_PAR / cin_group_size (parallel Couts per pass)
    wire [4:0] cin_group_size  = 5'd1 << log2_cin_group_size;
    wire [3:0] cin_group_mask  = cin_group_size[3:0] - 4'd1;
    wire [4:0] cout_par_active = C_PAR[4:0] >> log2_cin_group_size;

    // Outputs of convolvers and pad streamers running in parallel
    wire [ACC_BITS-1:0]     slot_conv_out [0:C_PAR-1];
    wire                    slot_valid    [0:C_PAR-1];
    wire                    slot_end      [0:C_PAR-1];

    // Per-slot (cout_idx, slot_active) hoisted out of SLOT generate for use
    // by the per-lane pix_acc adder tree below. The generate block below
    // drives these via `assign` from its local wires.
    wire [3:0] slot_cout_idx_arr [0:C_PAR-1];
    wire       slot_active_arr   [0:C_PAR-1];

    // Per-lane accumulators and requantize pipeline (indexed by cout group)
    reg signed [ACC_BITS-1:0]      pix_acc        [0:MAX_COUT_PAR-1];
    reg signed [ACC_BITS-1:0]      r_pix_acc      [0:MAX_COUT_PAR-1];
    reg signed [64-1:0]            scaled_pix_acc [0:MAX_COUT_PAR-1];
    reg signed [ACC_BITS-1:0]      q_pix_wide     [0:MAX_COUT_PAR-1];
    reg signed [N_BITS-1:0]        q_pix          [0:MAX_COUT_PAR-1];

    // Per-lane slices of the packed QP registers and ACC_read_data_out.
    wire signed [ACC_BITS-1:0]     r_bias_lane    [0:MAX_COUT_PAR-1];
    wire signed [M0_BITS-1:0]      r_m0_lane      [0:MAX_COUT_PAR-1];
    wire        [SHIFT_BITS-1:0]   r_n_shift_lane [0:MAX_COUT_PAR-1];
    wire signed [ACC_BITS-1:0]     acc_rd_lane    [0:MAX_COUT_PAR-1];

    genvar lc;
    generate
        for (lc = 0; lc < MAX_COUT_PAR; lc = lc + 1) begin : QP_LANE
            assign r_bias_lane[lc]    = $signed(r_bias   [lc*ACC_BITS   +: ACC_BITS]);
            assign r_m0_lane[lc]      = $signed(r_m0     [lc*M0_BITS    +: M0_BITS]);
            assign r_n_shift_lane[lc] =          r_n_shift[lc*SHIFT_BITS +: SHIFT_BITS];
            assign acc_rd_lane[lc]    = $signed(ACC_read_data_out[lc*ACC_BITS +: ACC_BITS]);
        end
    endgenerate

    // padding logic
    always @(*) begin
        is_padding = (input_row == 0) || (input_row == act_size + 9'd1)
                  || (input_col == 0) || (input_col == act_size + 9'd1);
    end

    // used to track if act fed into activation is a padded value or not
    always @(posedge clk) begin
        is_padded_act <= is_padding;
    end

    // Input row and col counters
    always @(posedge clk) begin
        if (rst || rst_convolvers) begin
            input_row <= 0;
            input_col <= 0;
        end else if (state == S_RUNNING) begin
            // advance pixel stream position every cycle while convolvers running
            if (input_col == act_size + 9'd1) begin
                input_col <= 0;
                if (input_row == act_size + 9'd1) begin
                    input_row <= 0;
                end else begin
                    input_row <= input_row + 1;
                end
            end else begin
                input_col <= input_col + 1;
            end
        end
    end

    // BRAM enable logic: only enable read for real pixels, not padding
    // data has one cycle latency and will be available in pixel_bram_data one cycle after its address is used
    always @(*) begin
        pixel_bram_en = 0;
        pixel_bram_addr = 0;
        if (state == S_RUNNING) begin
            // enable BRAM read for real pixels only (not padding)
            if (!is_padding) begin
                pixel_bram_en = 1;
                pixel_bram_addr = (round * act_size_sq) + ((input_row - 1) * act_size + (input_col - 1));
            end
            else                pixel_bram_en = 0;
        end
        end

    genvar gi;
    generate
        for (gi = 0; gi < C_PAR; gi = gi + 1) begin : SLOT

            // Slot -> (Cout, Cin) mapping via shift/mask (free hardware).
            // For log2_cin_group_size=4 (default): cout_idx=0 for all gi, cin_idx=gi → legacy behavior.
            wire [3:0] slot_cout_idx   = gi >> log2_cin_group_size;
            wire [3:0] slot_cin_idx    = gi & cin_group_mask;
            wire [4:0] slot_cout_idx_w = {1'b0, slot_cout_idx};
            wire slot_active = conv_running
                            && (slot_cout_idx_w < cout_par_active)
                            && ({1'b0, slot_cin_idx} < active_slots);

            // Hoist out of generate so the per-lane pix_acc adder can reach them.
            assign slot_cout_idx_arr[gi] = slot_cout_idx;
            assign slot_active_arr[gi]   = slot_active;

            // ------------------------------------------------------------------
            // convolver  (sees pad_size-wide stream)
            // ------------------------------------------------------------------
            // act_zp is the zero-point subtracted and padded activation value fed into the convolver
            reg signed [N_BITS-1:0] act_zp;

            always @(*) begin
                if (!is_padded_act) begin
                    // Slot gi reads channel slot_cin_idx of the loaded
                    // cin_group. In legacy mode (log2_cgs=4) slot_cin_idx==gi
                    // → identical to historic behaviour. In Cout-parallel
                    // mode (log2_cgs=2) slots in different cout lanes share
                    // the same Cin channels (slot_cin_idx = gi & 3).
                    act_zp = pixel_bram_data[slot_cin_idx * N_BITS +: N_BITS];
                end
                else begin
                    // zp_in subtraction: padding→0, real pixel→(pixel-zp_in)
                    // now pad with zp_in instead of 0 since bias has the baked in values
                    act_zp = r_zp_in;
                end
            end

            convolver #(
                .k (K),
                .s (STRIDE),
                .N (N_BITS),
                .ACC_BITS(ACC_BITS)
            ) u_conv (
                .clk        (clk),
                .ce         (slot_active),
                .global_rst (rst || rst_convolvers),
                .n          (pad_size),
                .activation (act_zp),
                .weight1    (weights_all_channels[gi*K*K*N_BITS +: K*K*N_BITS]),
                .conv_op    (slot_conv_out[gi]),
                .valid_conv (slot_valid[gi]),
                .end_conv   (slot_end[gi])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst)
        begin
            state          <= S_IDLE;
            done           <= 0;
            round          <= 0;
            conv_running   <= 0;
            rst_convolvers <= 1;
            r_zp_in        <= 0;
            r_zp_out       <= 0;
            r_bias         <= 0;
            r_m0           <= 0;
            r_n_shift      <= 0;
            req_weights    <= 0;
            drain_count    <= 0;
        end
        else
        begin
            rst_convolvers <= 0;
            done           <= 0;
            case (state)

            S_IDLE: begin
                if (start) begin
                    r_zp_in   <= zp_in;
                    r_zp_out  <= zp_out;
                    r_bias    <= bias;
                    r_m0      <= m0;
                    r_n_shift <= n_shift;
                    round          <= 0;
                    rst_convolvers <= 1;
                    req_weights    <= 0;
                    state          <= S_RUNNING;
                end
            end

            S_RUNNING: begin
                if (input_row == 0 && input_col == 0 && !rst_convolvers) begin
                    // start convolvers only when padded activations are ready
                    conv_running <= 1;
                end
                if (slot_end[0]) begin
                    // Enter S_DRAIN to let the new pipeline stages
                    // (r_scaled_pix_acc + RES_write_data_in_reg + control delay)
                    // flush the last accumulated value before we reset the
                    // convolvers and fire done. r_pix_acc holds across
                    // S_DRAIN because the "r_pix_acc <= pix_acc" assignment
                    // lives in the S_RUNNING case of the ACC always_ff.
                    state       <= S_DRAIN;
                    drain_count <= 0;
                end
            end

            S_WAIT_ROUND: begin
                rst_convolvers <= 0;
                // only keep req_weights high for one cycle
                req_weights <= 0;
                if (round == total_rounds - 1) begin
                    // all ch_in convolutions completed
                    state <= S_IDLE;
                    done <= 1;
                end else if (weights_ready) begin
                    // start another batch of convolutions
                    round <= round + 1;
                    rst_convolvers <= 1;
                    state <= S_RUNNING;
                end else begin
                    // wait in this state until weights for next round are loaded
                end
            end

            S_DRAIN: begin
                // Wait for r_scaled_pix_acc, RES_write_data_in_reg, and the
                // control-path delay flops to flush the last accumulated
                // value through the new pipeline. drain_count==2 gives 3
                // cycles in S_DRAIN, which is 1 cycle of margin past the
                // last RES BRAM write. r_pix_acc is not reassigned here
                // (no S_DRAIN case in the ACC always_ff), so the last
                // valid pix_acc holds and feeds the requantize chain
                // through the drain.
                if (drain_count == 2'd2) begin
                    state          <= S_WAIT_ROUND;
                    rst_convolvers <= 1;
                    conv_running   <= 0;
                    if (round != total_rounds - 1)
                        req_weights <= 1;
                end else begin
                    drain_count <= drain_count + 1;
                end
            end

            endcase


        end

    end

    // ------------------------------------------------------------------
    // conv_op accumulates (per-lane)
    // ------------------------------------------------------------------
    //
    // Per-lane adder trees: each lane sums slots assigned to its cout group.
    // For log2_cin_group_size=4 (default): all slots route to lane 0, so
    // pix_acc[0] matches the legacy scalar sum and pix_acc[1..] hold 0.
    integer vi_p, c_p;
    always @(*) begin
        for (c_p = 0; c_p < MAX_COUT_PAR; c_p = c_p + 1) begin
            pix_acc[c_p] = 0;
            for (vi_p = 0; vi_p < C_PAR; vi_p = vi_p + 1) begin
                if (slot_active_arr[vi_p] && (slot_cout_idx_arr[vi_p] == c_p[3:0]))
                    pix_acc[c_p] = pix_acc[c_p] + $signed(slot_conv_out[vi_p]);
            end
        end
    end

    // accumulator data to ACC RAM (packed, per-lane)
    integer cw_a;
    always @(*) begin
        // default to 0 so every code path drives the signal — prevents
        // latch inference. Safe because ACC_write_en is gated to 0
        // outside S_RUNNING by the FSM block above.
        ACC_write_data_in = 0;
        if (state == S_RUNNING) begin
            for (cw_a = 0; cw_a < MAX_COUT_PAR; cw_a = cw_a + 1) begin
                // TODO: my slot_valid does not align with the q_out value, is one cycle earlier
                // currently removed slot_valid[0] condition, but see if i rlly needed it?
                ACC_write_data_in[cw_a*ACC_BITS +: ACC_BITS] = (round == 0)
                    ? r_pix_acc[cw_a] + r_bias_lane[cw_a]
                    : acc_rd_lane[cw_a] + r_pix_acc[cw_a];
            end
        end
    end

    // only valid accumulates will occur during S_RUNNING state
    // acc values updated 2 cycles after valid conv_op output
    always @(posedge clk) begin : acc_ff
        integer c_ff;
        ACC_read_en <= 1;
        case(state)
        S_IDLE: begin
            if (start) begin
                ACC_write_en <= 0;
                ACC_write_address <= 0;
                ACC_read_address <= 0;
                for (c_ff = 0; c_ff < MAX_COUT_PAR; c_ff = c_ff + 1)
                    r_pix_acc[c_ff] <= 0;
            end
        end

        S_RUNNING: begin
            for (c_ff = 0; c_ff < MAX_COUT_PAR; c_ff = c_ff + 1)
                r_pix_acc[c_ff] <= pix_acc[c_ff]; // latch current acc value for use in next cycle's calculation

            // slot_end[0] takes priority: on the cycle the FSM transitions
            // to S_WAIT_ROUND, force the write strobe low so a stale write
            // does not bleed into the next state. Without this priority,
            // a same-cycle slot_valid[0] would re-arm ACC_write_en and the
            // (default-zero) ACC_write_data_in would zero out a real slot.
            if (slot_end[0]) begin
                ACC_write_en <= 0;
            end else if (slot_valid[0]) begin
                ACC_read_address <= ACC_read_address + 1;
                ACC_write_en <= 1;
            end else begin
                ACC_write_en <= 0;
            end
            if (ACC_write_en) begin
                ACC_write_address <= ACC_write_address + 1;
            end
        end

        S_WAIT_ROUND: begin
            if (round != total_rounds - 1) begin
                // start another batch of convolutions, reset addresses for new acc. values
                ACC_write_address <= 0;
                ACC_read_address <= 0;
                for (c_ff = 0; c_ff < MAX_COUT_PAR; c_ff = c_ff + 1)
                    r_pix_acc[c_ff] <= 0;
            end
        end

        endcase
    end

    // ------------------------------------------------------------------
    // re-scaling — per-lane scaled_pix_acc registered to break Cone A at DSP output
    // ------------------------------------------------------------------
    always @(posedge clk) begin : rescale_ff
        integer cs;
        if (rst) begin
            for (cs = 0; cs < MAX_COUT_PAR; cs = cs + 1)
                scaled_pix_acc[cs] <= 0;
        end else begin
            for (cs = 0; cs < MAX_COUT_PAR; cs = cs + 1)
                scaled_pix_acc[cs] <= $signed(ACC_write_data_in[cs*ACC_BITS +: ACC_BITS]) * r_m0_lane[cs];
        end
    end

    // Round-half-up nudge for the requantize shift: add 2^(r_n_shift-1)
    // before the arithmetic right-shift so `(x + nudge) >>> n` performs
    // round-nearest instead of truncation. Matches TFLite's gemmlowp
    // RoundingDivideByPOT for positive products (the common case post-SiLU).
    // Guard against r_n_shift == 0 to avoid `<<< -1` underflow.
    // Computed per-lane so each cout uses its own shift amount.
    wire signed [63:0] requant_nudge [0:MAX_COUT_PAR-1];
    generate
        for (lc = 0; lc < MAX_COUT_PAR; lc = lc + 1) begin : NUDGE
            assign requant_nudge[lc] =
                (r_n_shift_lane[lc] != {SHIFT_BITS{1'b0}}) ?
                    (64'sd1 <<< (r_n_shift_lane[lc] - {{(SHIFT_BITS-1){1'b0}}, 1'b1})) :
                    64'sd0;
        end
    endgenerate

    // Per-lane saturate to int8 [-128, 127]
    always @(*) begin : requant_comb
        integer cq;
        for (cq = 0; cq < MAX_COUT_PAR; cq = cq + 1) begin
            q_pix_wide[cq] = ((scaled_pix_acc[cq] + requant_nudge[cq]) >>> r_n_shift_lane[cq]) + r_zp_out;
            if (q_pix_wide[cq] > 127)
                q_pix[cq] = 127;
            else if (q_pix_wide[cq] < -128)
                q_pix[cq] = -128;
            else
                q_pix[cq] = q_pix_wide[cq][N_BITS-1:0];
        end
    end

    // ------------------------------------------------------------------
    // Control-path delay: 1-cycle pipeline of address/enable to align with
    // the new scaled_pix_acc data flop. Without this, q_pix would lag the
    // address by 1 cycle and writes would land at the wrong address.
    // ------------------------------------------------------------------
    reg [DEPTH_BITS-1:0]   ACC_write_address_d;
    reg                    ACC_write_en_d;

    always @(posedge clk) begin
        if (rst) begin
            ACC_write_address_d <= 0;
            ACC_write_en_d      <= 0;
        end else begin
            ACC_write_address_d <= ACC_write_address;
            ACC_write_en_d      <= ACC_write_en;
        end
    end

    // ------------------------------------------------------------------
    // write outputs — per-lane q_pix packed into RES_write_data_in, with
    // per-lane write strobes gated by cout_par_active so lanes beyond the
    // active Cout count never commit.
    // ------------------------------------------------------------------

    // in LAST ROUND can write to RES_RAM after every conv_valid output

    always @(posedge clk) begin : res_ff
        integer cr;
        if (rst) begin
            RES_write_en      <= {MAX_COUT_PAR{1'b0}};
            RES_write_address <= 0;
            RES_write_data_in <= 0;
        end else begin
            RES_write_address <= ACC_write_address_d;
            for (cr = 0; cr < MAX_COUT_PAR; cr = cr + 1) begin
                RES_write_data_in[cr*N_BITS +: N_BITS] <= q_pix[cr];
                if ((cr[4:0] < cout_par_active) && (round == total_rounds - 1))
                    RES_write_en[cr] <= ACC_write_en_d;
                else
                    RES_write_en[cr] <= 1'b0;
            end
        end
    end


endmodule
