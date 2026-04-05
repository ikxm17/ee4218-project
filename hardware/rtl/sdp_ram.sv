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

    initial if (MEM_FILE != "") $readmemh(MEM_FILE, ram);

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