`timescale 1ns / 1ps

module circular_buffer #(
    parameter WIDTH     = 8,
    parameter MAX_DEPTH = 255        // max runtime depth (layer 0: PAD_SIZE-K = 255)
)(
    input  clk,
    input  ce,
    input  rst,
    input  [$clog2(MAX_DEPTH+2)-1:0] depth,   // runtime delay in cycles (0..MAX_DEPTH)
    input  signed [WIDTH-1:0]        data_in,
    output signed [WIDTH-1:0]        data_out
);

    // Internal memory: depth+1 slots for depth cycles of delay
    // (matches flip-flop chain timing exactly)
    (* ram_style = "distributed" *) reg signed [WIDTH-1:0] mem [0:MAX_DEPTH];

    reg [$clog2(MAX_DEPTH+2)-1:0] wr_ptr;

    always @(posedge clk or posedge rst) begin
        if (rst)
            wr_ptr <= 0;
        else if (ce) begin
            if (wr_ptr == depth)
                wr_ptr <= 0;
            else
                wr_ptr <= wr_ptr + 1;
        end
    end

    // Separate always block without reset — allows LUTRAM inference
    always @(posedge clk)
        if (ce)
            mem[wr_ptr] <= data_in;

    // Combinational read: oldest entry = next slot after write pointer
    wire [$clog2(MAX_DEPTH+2)-1:0] rd_ptr = (wr_ptr == depth) ? {$clog2(MAX_DEPTH+2){1'b0}} : wr_ptr + 1;
    assign data_out = mem[rd_ptr];

endmodule
