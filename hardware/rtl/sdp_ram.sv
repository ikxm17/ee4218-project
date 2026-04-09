`timescale 1ns / 1ps
/*
 * Simple Dual-Port RAM with One Clock (UG901)
*/
module sdp_ram #(
    parameter DATA_WIDTH = 72,  // Ultra-RAMs are 72-bit wide
    parameter DEPTH = 4096,     // Ultra-RAMs are 288Kb blocks (minimum depth = 288 x 1024 / 72 = 4096)
    parameter RAM_STYLE = "ultra", // "ultra", "block", "distributed"
    parameter MEM_FILE = "", // File path for memory initialisation (contents should be hex numbers)
    localparam ADDR_WIDTH = $clog2(DEPTH)
)(
    /* Inputs */
    input logic clk,
    input logic en_a,
    input logic en_b,
    input logic we_a,
    input logic [ADDR_WIDTH-1:0] addr_a,
    input logic [ADDR_WIDTH-1:0] addr_b,
    input logic [DATA_WIDTH-1:0] din_a,
    /* Outputs */
    output logic [DATA_WIDTH-1:0] dout_b
);  
    (* ram_style = RAM_STYLE *) reg [DATA_WIDTH-1:0] ram [DEPTH-1:0];

    // Zero-init mirrors silicon URAM/BRAM bitstream programming so xsim
    // doesn't start with 'x. Without this the HDL engine diverges from
    // silicon between L1 and L10: the conv kernel reads an address that
    // wasn't fully written by an earlier layer (padding / RMW edge), gets
    // 'x in sim but 0 on silicon, and the 'x propagates through the
    // accumulator into wrong int8 outputs. L0 is unaffected because it
    // overwrites every word it reads from. Synthesis-safe: URAM init
    // defaults to 0 in hardware, so this loop is a no-op for the bitstream.
    initial begin
        for (int i = 0; i < DEPTH; i++) ram[i] = '0;
        if (MEM_FILE != "") $readmemh(MEM_FILE, ram);
    end

    /* Write-only Port */
    always_ff @(posedge clk) begin
        if (en_a) begin
            if (we_a) begin
                ram[addr_a] <= din_a;
            end
        end
    end

    /* Read-only Port */
    always_ff @(posedge clk) begin
        if (en_b) begin
            dout_b <= ram[addr_b];
        end
    end
endmodule