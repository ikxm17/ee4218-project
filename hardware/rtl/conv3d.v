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
    parameter ACT_SIZE     = 4, // test with 100
    parameter K            = 3,
    parameter STRIDE       = 1,
    parameter C_IN         = 3,
    parameter MAX_PARALLEL = 4,     // test with 4, original design 16
    parameter N_BITS       = 8,
    parameter ACC_BITS     = 32,
    parameter M0_BITS      = 32,
    parameter SHIFT_BITS   = 6,
    parameter DEPTH_BITS   = 16   // address bits: 2^DEPTH_BITS >= ACT_SIZE²
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               start,

    // Quantisation scalars
    input  wire signed [N_BITS-1:0]           zp_in,
    input  wire signed [N_BITS-1:0]           zp_out,
    input  wire signed [ACC_BITS-1:0]         bias,
    input  wire signed [M0_BITS-1:0]          m0,
    input  wire        [SHIFT_BITS-1:0]       n_shift,

    // -------------------------------------------------------------------------
    // Pixel BRAM interface  (one BRAM that outputs MAX_PARALLEL pixels per cycle)
    //   pixel_bram_addr – read address issued to input BRAM
    //   pixel_bram_en   – enb for each channel BRAM (0 on padding pixels)
    //   pixel_bram_data – registered read data from each channel BRAM
    // -------------------------------------------------------------------------
    output reg [DEPTH_BITS-1:0]                     pixel_bram_addr,
    output reg                                      pixel_bram_en,
    input  wire signed [MAX_PARALLEL*N_BITS-1:0]    pixel_bram_data,

    // Weight BRAM read ports (one per slot, addressed by global channel index)
    input  wire signed [MAX_PARALLEL*K*K*N_BITS-1:0]   weights_all_channels,

    // Acc BRAM interface
    // Store intermediate acc for each pixel here, store to RES after all channels and rescaling
    output reg                          ACC_write_en, 							
    output reg        [DEPTH_BITS-1:0]  ACC_write_address,
    output reg signed [ACC_BITS-1:0]    ACC_write_data_in,
    output reg                          ACC_read_en, 							
    output reg         [DEPTH_BITS-1:0] ACC_read_address,
    input signed       [ACC_BITS-1:0]   ACC_read_data_out,

    // Result BRAM interface 
    output reg                         RES_write_en, 							
    output reg [DEPTH_BITS-1:0]        RES_write_address,
    output reg signed [N_BITS-1:0]     RES_write_data_in,

    // output wire signed [N_BITS-1:0]           conv3d_out,
    // output wire                               valid_out,
    output reg                               done
);

    // Constants
    localparam PAD_SIZE       = ACT_SIZE + 2;                      // 258 for 256
    localparam OUT_SIZE       = (PAD_SIZE - K) / STRIDE + 1;       // 256 for K=3,S=1
    localparam OUT_PIXELS     = OUT_SIZE * OUT_SIZE;
    localparam TOTAL_ROUNDS   = (C_IN + MAX_PARALLEL - 1) / MAX_PARALLEL; // round up
    localparam LAST_ROUND_CIN = C_IN - (TOTAL_ROUNDS - 1) * MAX_PARALLEL;

    // Quantization parameters to be stored in registers
    reg signed [N_BITS-1:0]         r_zp_in, r_zp_out;
    reg signed [ACC_BITS-1:0]       r_bias; 
    reg signed [M0_BITS-1:0]        r_m0;
    reg        [SHIFT_BITS-1:0]     r_n_shift;

    // FSM
    localparam S_IDLE           = 2'd0;
    localparam S_RUNNING        = 2'd1;
    localparam S_WAIT_ROUND     = 2'd2;

    reg [1:0]                              state;
    reg [$clog2(TOTAL_ROUNDS+1)-1:0]       round;
    reg                                    conv_running;        // needed as condition to enable convolver
    reg                                    rst_convolvers;      // set after every round of convolutions

    // Input addressing and padding
    reg [$clog2(ACT_SIZE +2)-1:0]              input_row;         
    reg [$clog2(ACT_SIZE +2)-1:0]              input_col;
    reg                                        is_padding;
    reg                                        is_padded_act; // lags one cycle behind is_padding
       
    // Track number of convolvers running in parallel (aka slots)
    wire [$clog2(MAX_PARALLEL+1)-1:0] active_slots;
    assign active_slots = (round == TOTAL_ROUNDS - 1)
                        ? LAST_ROUND_CIN[$clog2(MAX_PARALLEL+1)-1:0]
                        : MAX_PARALLEL[$clog2(MAX_PARALLEL+1)-1:0];

    // Outputs of convolvers and pad streamers running in parallel
    wire [ACC_BITS-1:0]     slot_conv_out [0:MAX_PARALLEL-1];
    wire                    slot_valid    [0:MAX_PARALLEL-1];
    wire                    slot_end      [0:MAX_PARALLEL-1];

    // Accumulators
    reg signed [ACC_BITS-1:0]      pix_acc;
    reg signed [ACC_BITS-1:0]      r_pix_acc; // latched version of pix_acc
    reg signed [64-1:0]            scaled_pix_acc; // final acc * m0 BEFORE n_shift
    reg signed [N_BITS-1:0]        q_pix;     // final quantised pixel value to write to RES

    // padding logic
    always @(*) begin
        is_padding = (input_row == 0) || (input_row == ACT_SIZE +1) || (input_col == 0) || (input_col == ACT_SIZE +1);
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
            if (input_col == ACT_SIZE + 1) begin
                input_col <= 0;
                if (input_row == ACT_SIZE + 1) begin
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
                pixel_bram_addr = ((input_row-1) * (ACT_SIZE) + (input_col-1));
            end   
            else                pixel_bram_en = 0;
        end 
        end

    genvar gi;
    generate
        for (gi = 0; gi < MAX_PARALLEL; gi = gi + 1) begin : SLOT

            wire slot_active = conv_running && (gi < active_slots);

            // ------------------------------------------------------------------
            // convolver  (sees PAD_SIZE-wide stream)
            // ------------------------------------------------------------------
            // act_zp is the zero-point subtracted and padded activation value fed into the convolver
            reg signed [N_BITS-1:0] act_zp;

            always @(*) begin
                if (!is_padded_act) begin
                    // is_padding is used to determine enable BRAM read
                    // use enable to get correct act_zp
                    // real pixel, subtract zp_in
                    // TODO: think about how to handle future rounds
                    // act_zp = pixel_bram_data[gi * N_BITS +: N_BITS] - r_zp_in;
                    // bake zp_in into the bias calculation later
                    act_zp = pixel_bram_data[gi * N_BITS +: N_BITS];
                end
                else begin 
                    // zp_in subtraction: padding→0, real pixel→(pixel-zp_in)
                    // now pad with zp_in instead of 0 since bias has the baked in values
                    act_zp = r_zp_in;
                end
            end 

            convolver #(
                .n (PAD_SIZE),   // convolver sees the PADDED width
                .k (K),
                .s (STRIDE),
                .N (N_BITS), 
                .ACC_BITS(ACC_BITS)
            ) u_conv (
                .clk        (clk),
                .ce         (slot_active),
                .global_rst (rst || rst_convolvers),
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
                    state          <= S_RUNNING;
                end
            end

            S_RUNNING: begin
                if (input_row == 0 && input_col == 0 && !rst_convolvers) begin
                    // start convolvers only when padded activations are ready
                    conv_running <= 1;
                end
                if (slot_end[0]) begin
                    // use timing from first convolver as reference
                    state <= S_WAIT_ROUND;
                    rst_convolvers <= 1;
                    conv_running <= 0;
                end
            end

            S_WAIT_ROUND: begin
                rst_convolvers <= 0;
                if (round == TOTAL_ROUNDS -1) begin
                    // all ch_in convolutions completed
                    state <= S_IDLE;
                    done <= 1;
                end else begin
                    // start another batch of convolutions
                    round <= round + 1;
                    rst_convolvers <= 1;
                    state <= S_RUNNING;
                end
            end

            endcase


        end
    
    end

    // ------------------------------------------------------------------
    // conv_op accumulates
    // ------------------------------------------------------------------
    
    // intermediate acc values are stored in ACC_BRAM
    // each pixel accumulates over TOTAL_ROUNDS rounds of convolutions
    // updated acc values are written to ACC_BRAM two cycles after valid conv_op


    // adder for all active convolver conv_op values
    integer vi;     // used to iterate through active convolvers

    always @(*) begin
        pix_acc = 0;
        for (vi = 0; vi < MAX_PARALLEL; vi = vi + 1) begin
            if (vi < active_slots)
                pix_acc = pix_acc + $signed(slot_conv_out[vi]);
        end
    end

    // accumulator data to BRAM
    always @(*) begin
        if (state == S_RUNNING) begin
            // TODO: my slot_valid does not align with the q_out value, is one cycle earlier
            // currently removed slot_valid[0] condition, but see if i rlly needed it?
            ACC_write_data_in = (round == 0) ? r_pix_acc + r_bias : ACC_read_data_out + r_pix_acc;
        end
    end

    // only valid accumulates will occur during S_RUNNING state
    // acc values updated 2 cycles after valid conv_op output
    always @(posedge clk)
    begin
        ACC_read_en <= 1; 
        case(state)
        S_IDLE: begin
            if (start) begin
                ACC_write_en <= 0;
                ACC_write_address <= 0;
                ACC_read_address <= 0;

                r_pix_acc <= 0;
            end
        end

        S_RUNNING: begin
            r_pix_acc <= pix_acc; // latch current acc value for use in next cycle's calculation

            if (slot_valid[0]) begin
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
            if (round != TOTAL_ROUNDS -1) begin
                // start another batch of convolutions, reset addresses for new acc. values
                ACC_write_address <= 0;
                ACC_read_address <= 0;
                r_pix_acc <= 0;
            end
        end
        
        endcase
    end

    // ------------------------------------------------------------------
    // re-scaling (comb, done pixel by pixel)
    // ------------------------------------------------------------------
    always @(*) begin
        scaled_pix_acc = ACC_write_data_in * r_m0;
        q_pix = (scaled_pix_acc >>> r_n_shift) + r_zp_out;
    end

    // ------------------------------------------------------------------
    // write outputs
    // ------------------------------------------------------------------
    
    // in LAST ROUND can write to RES_RAM after every conv_valid output

    always @(*) begin
        RES_write_data_in = q_pix;
        RES_write_address = ACC_write_address;
        if (round == TOTAL_ROUNDS -1) begin
            // only write to RES in last round after each pixel's valid convolution outputs
            RES_write_en = ACC_write_en; 
        end
        else RES_write_en = 0;
    end


endmodule