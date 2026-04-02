//file: mac_manual.v
//note that this file has been modified after the addiion of fixed point arithmetic support
//this post has not been edited for the same keeping in mind first time readers. Do refer 
//to that post in this series.
`timescale 1ns / 1ps

module mac_manual #(
    parameter N = 8, 
    parameter ACC = 32
    )(
    input clk,sclr,ce,
    input signed [N-1:0] a,
    input signed [N-1:0] b,
    input signed [ACC-1:0] c,
    output reg signed [ACC-1:0] p
    );

always@(posedge clk,posedge sclr)
 begin
    if(sclr)
    begin
        p<=0;
    end
    else if(ce)
    begin
        p <= (a*b+c);                   //performs the multiply accumulate operation
    end
 end
endmodule