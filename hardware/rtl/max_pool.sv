`timescale 1ns / 1ps
`include "layer_config.svh"

module max_pool #(
    parameter N_BITS     = 8,
    parameter DEPTH_BITS = 16
)(
    input  logic                        clk,
    input  logic                        rst_n,

    /* Layer config (active layer) */
    input  logic [1:0]                  layer_type,
    input  logic [8:0]                  act_size,      // pre-pool spatial dimension (h_in = w_in)

    /* Input from activation stage */
    input  logic                        in_valid,
    input  logic [DEPTH_BITS-1:0]       in_addr,
    input  logic signed [N_BITS-1:0]    in_data,

    /* Output (to result bus or feature map) */
    output logic                        out_valid,
    output logic [DEPTH_BITS-1:0]       out_addr,
    output logic signed [N_BITS-1:0]    out_data
);

    /* ================================================================
     *  Bypass Detection
     *
     *  Only CONV3_POOL layers (layer_type == 2'b01) use the pooler.
     *  All other layer types pass through with zero latency.
     * ================================================================ */
    wire pool_en = (layer_type == CONV3_POOL);

    /* ================================================================
     *  Position Tracking
     *
     *  Row/col counters track the current pixel position within
     *  the output channel's feature map (act_size x act_size).
     *  The {row[0], col[0]} bits directly encode which of the 4
     *  phases in the 2x2 pooling window we are in.
     * ================================================================ */
    logic [8:0] col_cnt;
    logic [8:0] row_cnt;

    /* ================================================================
     *  Shift Register Buffer
     *
     *  Stores even-row column-wise maxima so they are available
     *  when the odd row arrives.  Depth = act_size/2 (number of
     *  pool windows per row).  Max depth = 128 for act_size=256.
     *  Implemented as a simple register array — 128 x 8 bits is
     *  small enough for distributed RAM / registers.
     * ================================================================ */
    logic signed [N_BITS-1:0] sr_buf [0:127];

    /* ================================================================
     *  Max Register
     *
     *  Accumulates the pair-wise maximum within a row.
     *  Written on even-col cycles, read on odd-col cycles.
     * ================================================================ */
    logic signed [N_BITS-1:0] max_reg;

    /* ================================================================
     *  Output Registers (pool path)
     * ================================================================ */
    logic                        pool_valid;
    logic [DEPTH_BITS-1:0]       pool_addr;
    logic signed [N_BITS-1:0]    pool_data;
    logic [DEPTH_BITS-1:0]       out_cnt;

    /* ================================================================
     *  Signed Maximum
     * ================================================================ */
    function automatic logic signed [N_BITS-1:0] smax(
        input logic signed [N_BITS-1:0] a,
        input logic signed [N_BITS-1:0] b
    );
        smax = (a > b) ? a : b;
    endfunction

    /* ================================================================
     *  Pool Datapath
     *
     *  2x2 stride-2 max pooling.  The 4 phases are:
     *
     *    {row[0], col[0]}
     *    00 : Even row, even col — store pixel in max_reg
     *    01 : Even row, odd col  — row-wise max → shift register
     *    10 : Odd row, even col  — compare pixel with SR entry
     *    11 : Odd row, odd col   — final 2x2 max → valid output
     *
     *  The shift register bridges the gap between even and odd
     *  rows: depth = act_size/2, so by the time the odd row's
     *  first pixel arrives, the SR has rotated back to present
     *  the matching even-row column max at its output.
     * ================================================================ */
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt    <= '0;
            row_cnt    <= '0;
            max_reg    <= '0;
            pool_valid <= 1'b0;
            pool_addr  <= '0;
            pool_data  <= '0;
            out_cnt    <= '0;
        end else begin
            pool_valid <= 1'b0;

            if (pool_en && in_valid) begin
                case ({row_cnt[0], col_cnt[0]})
                    2'b00: begin
                        max_reg <= in_data;
                    end
                    2'b01: begin
                        sr_buf[col_cnt >> 1] <= smax(in_data, max_reg);
                    end
                    2'b10: begin
                        max_reg <= smax(in_data, sr_buf[col_cnt >> 1]);
                    end
                    2'b11: begin
                        pool_data  <= smax(in_data, max_reg);
                        pool_addr  <= out_cnt;
                        pool_valid <= 1'b1;
                        out_cnt    <= out_cnt + 1;
                    end
                endcase

                /* Position counter update */
                if (col_cnt == act_size - 9'd1) begin
                    col_cnt <= '0;
                    if (row_cnt == act_size - 9'd1) begin
                        row_cnt <= '0;
                        out_cnt <= '0;
                    end else begin
                        row_cnt <= row_cnt + 9'd1;
                    end
                end else begin
                    col_cnt <= col_cnt + 9'd1;
                end
            end
        end
    end

    /* ================================================================
     *  Output Mux
     *
     *  Bypass: combinational pass-through (zero latency).
     *  Pool:   registered output (1-cycle latency).
     * ================================================================ */
    assign out_valid = pool_en ? pool_valid : in_valid;
    assign out_addr  = pool_en ? pool_addr  : in_addr;
    assign out_data  = pool_en ? pool_data  : in_data;

endmodule
