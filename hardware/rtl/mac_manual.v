//file: mac_manual.v
//note that this file has been modified after the addiion of fixed point arithmetic support
//this post has not been edited for the same keeping in mind first time readers. Do refer 
//to that post in this series.
`timescale 1ns / 1ps

module mac_manual #(
    parameter N = 8,
    parameter ACC = 32
)(
    input clk,
    input sclr,
    input ce,
    input  signed [N-1:0]   a,
    input  signed [N-1:0]   b,
    input  signed [ACC-1:0] c,
    output reg signed [ACC-1:0] p
);

    (* use_dsp = "yes" *)
    wire signed [2*N-1:0] product = a * b;

    always @(posedge clk, posedge sclr) begin
        if (sclr)
            p <= 0;
        else if (ce)
            p <= product + c;
    end
endmodule