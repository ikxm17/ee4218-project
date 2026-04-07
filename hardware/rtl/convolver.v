//file: convolver.v
//credit to: https://thedatabus.in/convolver/
`timescale 1ns / 1ps

module convolver #(
    parameter k = 9'h003,
    parameter s = 1,
    parameter N = 8,
    parameter ACC_BITS = 32
)(
    input clk,
    input ce,
    input global_rst,
    input [8:0] n,
    input signed [N-1:0] activation,
    input signed [(k*k)*N-1:0] weight1,
    output signed [ACC_BITS-1:0] conv_op,
    output valid_conv,
    output end_conv
);

reg [31:0] count,count2,count3,row_count;
reg en1,en2,en3;

// Combinational n-dependent values (must be valid immediately, not
// registered, because end_conv/fill checks fire on the first cycle
// after reset — a registered version initializes to 0 and would
// falsely trigger end_conv and pipeline-full conditions)
wire [17:0] n_sq_plus2 = n * n + 18'd2;
wire [8:0]  n_minus_k  = n - k;
wire [17:0] fill_count = (k - 1) * n + (k - 1);

wire signed [ACC_BITS-1:0] tmp [k*k+1:0];
wire signed [N-1:0] weight [0:k*k-1];

//breaking our weights into separate variables. We are forced to do this because verilog does not allow us to pass multi-dimensional 
//arrays as parameters
//----------------------------------------------------------------------------------------------------------------------------------
generate
    genvar l;
    for(l=0;l<k*k;l=l+1)
    begin
        assign weight [l][N-1:0] = weight1[N*l +: N];       
    end 
endgenerate
//----------------------------------------------------------------------------------------------------------------------------------
assign tmp[0] = 32'h0000000;

//The following generate loop enables us to lay out any number of MAC units specified during the synthesis, without having to commit to a //fixed size 
generate
genvar i;
  for(i = 0;i<k*k;i=i+1)
  begin: MAC
    if((i+1)%k ==0)                       //end of the row
    begin
      if(i==k*k-1)                        //end of convolver
      begin
      mac_manual #(.N(N)) mac(        //implements a*b+c
        .clk(clk),                        // input clk
        .ce(ce),                          // input ce
        .sclr(global_rst),                // input sclr
        .a(activation),                   // activation input [15 : 0] a
        .b(weight[i]),                    // weight input [15 : 0] b
        .c(tmp[i]),                       // previous mac sum input [32 : 0] c
        .p(conv_op)                       // output [32 : 0] p
        );
      end
      else
      begin
      wire signed [ACC_BITS-1:0] tmp2;
      //make a mac unit
      mac_manual #(.N(N),.ACC(ACC_BITS)) mac(
        .clk(clk), 
        .ce(ce), 
        .sclr(global_rst), 
        .a(activation), 
        .b(weight[i]), 
        .c(tmp[i]), 
        .p(tmp2) 
        );

      circular_buffer #(
          .WIDTH     (ACC_BITS),
          .MAX_DEPTH (255)
      ) SR (
          .clk      (clk),
          .ce       (ce),
          .rst      (global_rst),
          .depth    (n_minus_k),
          .data_in  (tmp2),
          .data_out (tmp[i+1])
      );
      end
    end
    else
    begin
    mac_manual #(.N(N)) mac2(
      .clk(clk), 
      .ce(ce),
      .sclr(global_rst),
      .a(activation),
      .b(weight[i]),
      .c(tmp[i]), 
      .p(tmp[i+1])
      );
    end 
  end 
endgenerate

//The following logic generates the 'valid_conv' and 'end_conv' output signals that tell us if the output is valid.
always@(posedge clk) 
begin
  if(global_rst)
  begin
    count <=0;                      //master counter: counts the clock cycles
    count2<=0;                      //counts the valid convolution outputs
    count3<=0;                      // counts the number of invalid onvolutions where the kernel wraps around the next row of inputs.
    row_count <= 0;                 //counts the number of rows of the output.  
    en1<=0;
    en2<=1;
    en3<=0;
  end
  else if(ce)
  begin
    if(count == fill_count)           // time taken for the pipeline to fill up is (k-1)*n+k-1
    begin
      en1 <= 1'b1;
      count <= count+1'b1;
    end
    else
    begin 
      count<= count+1'b1;
    end
  end
  if(en1 && en2) 
  begin
    if(count2 == n_minus_k)
    begin
      count2 <= 0;
      en2 <= 0 ;
      row_count <= row_count + 1'b1;
    end
    else 
    begin
      count2 <= count2 + 1'b1;
    end
  end

  if(~en2) 
  begin
  if(count3 == k-2)
  begin
    count3<=0;
    en2 <= 1'b1;
  end
  else
    count3 <= count3 + 1'b1;
  end
  //one in every 's' convolutions becomes valid, also some exceptional cases handled for high when count2 = 0
  if((((count2 + 1) % s == 0) && (row_count % s == 0))||(count3 == k-2)&&(row_count % s == 0)||(count == fill_count))
  begin                                                                                                                        
    en3 <= 1;                                                                                                                             
  end
  else 
    en3 <= 0;
end
    assign end_conv = (count >= n_sq_plus2) ? 1'b1 : 1'b0;
    assign valid_conv = (en1&&en2&&en3);
endmodule