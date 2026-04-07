`timescale 1ns / 1ps

// 1x1 pointwise convolution engine
//
// For each pixel: multiply 16 channels by 16 weights, sum.
// No padding, no line buffers, no pipeline fill time.
// ACC/RES interface matches conv3d.v for drop-in muxing.

module conv1d #(
    parameter MAX_PARALLEL = 16,
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

    // Quantisation scalars
    input  wire signed [N_BITS-1:0]           zp_in,
    input  wire signed [N_BITS-1:0]           zp_out,
    input  wire signed [ACC_BITS-1:0]         bias,
    input  wire signed [M0_BITS-1:0]          m0,
    input  wire        [SHIFT_BITS-1:0]       n_shift,

    // Pixel BRAM interface (same as conv3d)
    output reg  [DEPTH_BITS-1:0]              pixel_bram_addr,
    output reg                                pixel_bram_en,
    input  wire signed [MAX_PARALLEL*N_BITS-1:0] pixel_bram_data,

    // Weights: 16 x int8 (one weight per channel, K=1)
    input  wire signed [MAX_PARALLEL*N_BITS-1:0] weights_all_channels,

    // ACC BRAM interface (same as conv3d)
    output reg                                ACC_write_en,
    output reg        [DEPTH_BITS-1:0]        ACC_write_address,
    output reg signed [ACC_BITS-1:0]          ACC_write_data_in,
    output reg                                ACC_read_en,
    output reg        [DEPTH_BITS-1:0]        ACC_read_address,
    input  signed     [ACC_BITS-1:0]          ACC_read_data_out,

    // Result output (same as conv3d)
    output reg                                RES_write_en,
    output reg [DEPTH_BITS-1:0]               RES_write_address,
    output reg signed [N_BITS-1:0]            RES_write_data_in,

    // Weight loading control (same as conv3d)
    output reg                                req_weights,
    input  wire                               weights_ready,

    output reg                                done
);

    // Runtime-derived constants
    wire [3:0]  total_rounds   = cin[7:4] + (|cin[3:0]);   // ceil(cin / 16)
    wire [4:0]  last_round_cin = (cin[3:0] == 4'd0) ? 5'd16 : {1'b0, cin[3:0]};

    reg [17:0] act_size_sq;

    // Registered scalars
    reg signed [N_BITS-1:0]     r_zp_out;
    reg signed [ACC_BITS-1:0]   r_bias;
    reg signed [M0_BITS-1:0]    r_m0;
    reg        [SHIFT_BITS-1:0] r_n_shift;

    // FSM
    localparam S_IDLE       = 2'd0;
    localparam S_RUNNING    = 2'd1;
    localparam S_WAIT_ROUND = 2'd2;

    reg [1:0] state;
    reg [3:0] round;

    // Pixel counter
    reg [17:0] pixel_count;
    wire       reading_done = (pixel_count >= act_size_sq);

    // Pipeline drain: after last pixel read, MAC pipeline needs 3 cycles
    reg [1:0] drain;

    // Active channel count
    wire [$clog2(MAX_PARALLEL+1)-1:0] active_slots;
    assign active_slots = (round == total_rounds - 1)
                        ? last_round_cin[$clog2(MAX_PARALLEL+1)-1:0]
                        : MAX_PARALLEL[$clog2(MAX_PARALLEL+1)-1:0];

    // ------------------------------------------------------------------
    // MAC array: combinational 16-wide multiply-accumulate
    // ------------------------------------------------------------------
    reg signed [ACC_BITS-1:0] pix_acc;
    reg signed [ACC_BITS-1:0] r_pix_acc;
    integer vi;

    always @(*) begin
        pix_acc = 0;
        for (vi = 0; vi < MAX_PARALLEL; vi = vi + 1) begin
            if (vi < active_slots)
                pix_acc = pix_acc
                    + $signed(pixel_bram_data[vi*N_BITS +: N_BITS])
                    * $signed(weights_all_channels[vi*N_BITS +: N_BITS]);
        end
    end

    // mac_valid: 1 cycle after pixel_bram_en (BRAM read latency)
    reg mac_valid;

    // ------------------------------------------------------------------
    // Pixel read addressing (combinational)
    // ------------------------------------------------------------------
    always @(*) begin
        pixel_bram_en   = 0;
        pixel_bram_addr = 0;
        if (state == S_RUNNING && !reading_done) begin
            pixel_bram_en   = 1;
            pixel_bram_addr = round * act_size_sq + pixel_count;
        end
    end

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state       <= S_IDLE;
            done        <= 0;
            round       <= 0;
            r_zp_out    <= 0;
            r_bias      <= 0;
            r_m0        <= 0;
            r_n_shift   <= 0;
            req_weights <= 0;
            mac_valid   <= 0;
            pixel_count <= 0;
            drain       <= 0;
            act_size_sq <= 0;
        end else begin
            done        <= 0;
            mac_valid   <= 0;

            case (state)
            S_IDLE: begin
                if (start) begin
                    r_zp_out    <= zp_out;
                    r_bias      <= bias;
                    r_m0        <= m0;
                    r_n_shift   <= n_shift;
                    round       <= 0;
                    pixel_count <= 0;
                    drain       <= 0;
                    req_weights <= 0;
                    act_size_sq <= act_size * act_size;
                    state       <= S_RUNNING;
                end
            end

            S_RUNNING: begin
                // Advance pixel counter while reading
                if (!reading_done) begin
                    pixel_count <= pixel_count + 1;
                end

                // MAC valid follows pixel_bram_en by 1 cycle
                mac_valid <= pixel_bram_en;

                // Pipeline drain after all pixels read
                if (reading_done) begin
                    drain <= drain + 1;
                end

                // Transition when pipeline fully drained
                // drain=3: mac_valid done, ACC_write_en done, last write committed
                if (drain == 2'd3) begin
                    state <= S_WAIT_ROUND;
                    if (round != total_rounds - 1)
                        req_weights <= 1;
                end
            end

            S_WAIT_ROUND: begin
                req_weights <= 0;
                if (round == total_rounds - 1) begin
                    done  <= 1;
                    state <= S_IDLE;
                end else if (weights_ready) begin
                    round       <= round + 1;
                    pixel_count <= 0;
                    drain       <= 0;
                    state       <= S_RUNNING;
                end
            end

            endcase
        end
    end

    // ------------------------------------------------------------------
    // ACC accumulation (matches conv3d timing exactly)
    // ------------------------------------------------------------------
    always @(*) begin
        ACC_write_data_in = (round == 0) ? r_pix_acc + r_bias
                                         : ACC_read_data_out + r_pix_acc;
    end

    always @(posedge clk) begin
        ACC_read_en <= 1;

        case (state)
        S_IDLE: begin
            if (start) begin
                ACC_write_en      <= 0;
                ACC_write_address <= 0;
                ACC_read_address  <= 0;
                r_pix_acc         <= 0;
            end
        end

        S_RUNNING: begin
            r_pix_acc <= pix_acc;

            if (mac_valid) begin
                ACC_read_address <= ACC_read_address + 1;
                ACC_write_en     <= 1;
            end else begin
                ACC_write_en     <= 0;
            end

            if (ACC_write_en) begin
                ACC_write_address <= ACC_write_address + 1;
            end
        end

        S_WAIT_ROUND: begin
            if (round != total_rounds - 1) begin
                ACC_write_address <= 0;
                ACC_read_address  <= 0;
                r_pix_acc         <= 0;
            end
        end

        endcase
    end

    // ------------------------------------------------------------------
    // Requantization (combinational, identical to conv3d)
    // ------------------------------------------------------------------
    reg signed [64-1:0]        scaled_pix_acc;
    reg signed [ACC_BITS-1:0]  q_pix_wide;
    reg signed [N_BITS-1:0]    q_pix;

    always @(*) begin
        scaled_pix_acc = ACC_write_data_in * r_m0;
        q_pix_wide = (scaled_pix_acc >>> r_n_shift) + r_zp_out;
        if (q_pix_wide > 127)
            q_pix = 127;
        else if (q_pix_wide < -128)
            q_pix = -128;
        else
            q_pix = q_pix_wide[N_BITS-1:0];
    end

    // ------------------------------------------------------------------
    // RES output (only on last round)
    // ------------------------------------------------------------------
    always @(*) begin
        RES_write_data_in = q_pix;
        RES_write_address = ACC_write_address;
        if (round == total_rounds - 1)
            RES_write_en = ACC_write_en;
        else
            RES_write_en = 0;
    end

endmodule
