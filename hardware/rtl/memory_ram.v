`timescale 1ns / 1ps

/* 
----------------------------------------------------------------------------------
--	(c) Rajesh C Panicker, NUS
--  Description : Module implementing a single port fully synchronous RAM to act as local memory for the AXI Stream Coprocessor
--	License terms :
--	You are free to use this code as long as you
--		(i) DO NOT post a modified version of this on any public repository;
--		(ii) use it only for educational purposes;
--		(iii) accept the responsibility to ensure that your implementation does not violate any intellectual property of any entity.
--		(iv) accept that the program is provided "as is" without warranty of any kind or assurance regarding its suitability for any particular purpose;
--		(v) send an email to rajesh.panicker@ieee.org briefly mentioning its use (except when used for the course EE4218 at the National University of Singapore);
--		(vi) retain this notice in this file or any files derived from this.
----------------------------------------------------------------------------------
*/

// width is the number of bits per location; depth_bits is the number of address bits. 2^depth_bits is the number of locations
// Simple Dual-Port Block RAM with Dual Clocks (Verilog) from UG901
module memory_RAM
	#(
		parameter width = 8, 					// width is the number of bits per location
		parameter depth_bits = 2				// depth is the number of locations (2^number of address bits)
	) 
	(
		input clka,
		input clkb,	
		input ena,	
		input enb,	
		input wea,			
		input [depth_bits-1:0] addra,
		input [depth_bits-1:0] addrb,
		input [width-1:0] dia,
		output reg [width-1:0] dob
	);
    
    reg [width-1:0] RAM [0:2**depth_bits-1];  
    
  	// the following is from a template given in Vivado synthesis manual.
  	// Read up more about write first, read first, no change modes.

	always @(posedge clka) begin
		if (ena) begin
			if (wea) RAM[addra] <= dia;
		end
	end

	always @(posedge clkb) begin
		if (enb) begin
			dob <= RAM[addrb];
		end
	end

endmodule