`timescale 1ns / 1ps

module tb_conv3d;

    // Parameters matching DUT
    parameter ACT_SIZE     = 4;  // Toggle between 256 and 16 for your test cases
    parameter K            = 3;
    parameter STRIDE       = 1;
    parameter C_IN         = 3; // Toggle between 3 and 128
    parameter MAX_PARALLEL = 4;
    parameter N_BITS       = 8;
    parameter ACC_BITS     = 32;
    parameter M0_BITS      = 32;
    parameter SHIFT_BITS   = 6;
    parameter DEPTH_BITS   = 16;

    // Clock and Reset
    logic clk;
    logic rst;
    logic start;
    wire  done;

    // Quantization Scalars
    logic signed [N_BITS-1:0]   zp_in, zp_out;
    logic signed [ACC_BITS-1:0] bias;
    logic signed [M0_BITS-1:0]  m0;
    logic        [SHIFT_BITS-1:0] n_shift;

    // Buffers for file reading
    logic signed [N_BITS-1:0]       zp_in_file [0:0];   // 1-element array for $readmemh
    logic signed [N_BITS-1:0]       zp_out_file [0:0];
    logic signed [ACC_BITS-1:0]     bias_file [0:0];
    logic signed [M0_BITS-1:0]      m0_file [0:0];
    logic        [SHIFT_BITS-1:0]   n_shift_file [0:0];

    // BRAM Interfaces
    wire [DEPTH_BITS-1:0]              pixel_bram_addr;
    wire                               pixel_bram_en;
    logic signed [MAX_PARALLEL*N_BITS-1:0]    pixel_bram_data;

    logic signed [MAX_PARALLEL*K*K*N_BITS-1:0]   weights_all_channels;

    // Acc BRAM
    wire                               ACC_write_en;
    wire        [DEPTH_BITS-1:0]       ACC_write_address;
    wire signed [ACC_BITS-1:0]         ACC_write_data_in;
    wire                               ACC_read_en;
    wire         [DEPTH_BITS-1:0]      ACC_read_address;
    logic signed [ACC_BITS-1:0]        ACC_read_data_out; // Must be logic to drive it

    // Result BRAM
    wire                                 RES_write_en;
    wire            [DEPTH_BITS-1:0]     RES_write_address;
    wire signed     [N_BITS-1:0]         RES_write_data_in;

    // Internal Memory for Simulation
    // logic [N_BITS-1:0] pixel_mem [C_IN-1:0][(ACT_SIZE*ACT_SIZE)-1:0];
    // logic [N_BITS-1:0] weight_mem [C_IN-1:0][(K*K)-1:0];
    logic signed [N_BITS-1:0] res_mem [(ACT_SIZE*ACT_SIZE)-1:0];
    logic signed [ACC_BITS-1:0] ref_acc [(ACT_SIZE*ACT_SIZE)-1:0];

    // Instantiate DUT
    conv3d #(
        .ACT_SIZE(ACT_SIZE), .K(K), .STRIDE(STRIDE), .C_IN(C_IN),
        .MAX_PARALLEL(MAX_PARALLEL), .N_BITS(N_BITS), .ACC_BITS(ACC_BITS),
        .M0_BITS(M0_BITS), .SHIFT_BITS(SHIFT_BITS), .DEPTH_BITS(DEPTH_BITS)
    ) dut (.*);

    // Clock Generation
    initial clk = 0;
    always #5 clk = ~clk;

    // BRAM Data Feeding Logic
    // -------------------------------------------------------------------------
    // Pixel and Weight Memory Definitions
    // -------------------------------------------------------------------------
    // pixel_mem: [Channel][Spatial_Index]
    logic signed [N_BITS-1:0] pixel_mem [0:C_IN-1][0:(ACT_SIZE*ACT_SIZE)-1];
    
    // weight_mem: [Out_Ch][kH][kW][In_Ch]
    // For testing one Out_Ch, we simplify to [kH][kW][In_Ch]
    logic signed [N_BITS-1:0] weight_mem [0:K-1][0:K-1][0:C_IN-1];

    // -------------------------------------------------------------------------
    // Weights Preparation (Combinational)
    // -------------------------------------------------------------------------
    // The module expects weights sliced per input channel: [In_Ch][K*K]
    always_comb begin
        weights_all_channels = '0;
        for (int i = 0; i < MAX_PARALLEL; i++) begin
            automatic int current_in_ch = (dut.round * MAX_PARALLEL) + i;
            if (current_in_ch < C_IN) begin
                // Flatten the KxK kernel for the specific input channel
                for (int kh = 0; kh < K; kh++) begin
                    for (int kw = 0; kw < K; kw++) begin
                        // Indexing: channel i's K*K block, specifically the kh,kw bit range
                        // TODO: FIX THIS INDEXING 
                        // currently its giving 1st weight of every channel to the first 
                        weights_all_channels[(i * K * K * N_BITS) + (kh * K + kw) * N_BITS +: N_BITS] 
                            = weight_mem[kh][kw][current_in_ch];
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // BRAM Data Feeding Logic (Synchronous Read)
    // -------------------------------------------------------------------------
    // Most BRAMs have a 1-cycle latency. 
    // We check pixel_bram_en and return MAX_PARALLEL channels at pixel_bram_addr
    always_ff @(posedge clk) begin
        if (pixel_bram_en) begin
            for (int i = 0; i < MAX_PARALLEL; i++) begin
                automatic int current_in_ch = (dut.round * MAX_PARALLEL) + i;
                if (current_in_ch < C_IN) begin
                    pixel_bram_data[i*N_BITS +: N_BITS] <= pixel_mem[current_in_ch][pixel_bram_addr];
                end else begin
                    pixel_bram_data[i*N_BITS +: N_BITS] <= 0;
                end
            end
        end else begin
            pixel_bram_data <= '0; // Or keep last value depending on BRAM config
        end
    end
    
    // Acc BRAM Read/Write Logic
    logic [ACC_BITS-1:0] acc_bram_mem [0:(1<<DEPTH_BITS)-1];

    always_ff @(posedge clk) begin
        if (ACC_write_en) begin
            acc_bram_mem[ACC_write_address] <= ACC_write_data_in;
        end
        if (ACC_read_en) begin
            ACC_read_data_out <= acc_bram_mem[ACC_read_address];
        end
    end
    
    // Result Capture
    always @(posedge clk) begin
        if (RES_write_en)
            res_mem[RES_write_address] <= RES_write_data_in;
    end

    // Main Test Sequence
    initial begin
        // 1. Load data from .mem files
        $readmemh("zp_in.mem", zp_in_file);   zp_in   = zp_in_file[0];
        $readmemh("zp_out.mem", zp_out_file); zp_out  = zp_out_file[0];
        $readmemh("bias.mem", bias_file);     bias    = bias_file[0];
        $readmemh("m0.mem", m0_file);         m0      = m0_file[0];
        $readmemh("n_shift.mem", n_shift_file); n_shift = n_shift_file[0];

        // Loading Multi-dimensional arrays (requires specific file formatting)
        $readmemh("pixels.mem", pixel_mem);
        $readmemh("weights.mem", weight_mem);

        // 2. Drive Reset and Start
        rst = 1; start = 0;
        #20 rst = 0;
        #10 start = 1;
        #10 start = 0;

        // 3. Wait for Done
        wait(done);
        #100;

        // 4. Verification
        run_reference_model();
        compare_results();
        
        $finish;
    end

    // Reference Model Logic
    task automatic run_reference_model();
        logic signed [ACC_BITS-1:0] product;
        logic signed [N_BITS:0] val; // Extra bit for subtraction sign
        logic signed [N_BITS-1:0] weight_val;

        // 1. Initialize reference accumulator once
        for(int i=0; i<ACT_SIZE*ACT_SIZE; i++) ref_acc[i] = 0;

        // 2. Loop through channels
        for (int c = 0; c < C_IN; c++) begin
            for (int r = 0; r < ACT_SIZE; r++) begin
                for (int col = 0; col < ACT_SIZE; col++) begin
                    // We don't reset 'sum' here because we accumulate into ref_acc
                    // for every channel 'c' that covers this pixel.
                    
                    // 3x3 Convolution window
                    for (int ky = 0; ky < K; ky++) begin
                        for (int kx = 0; kx < K; kx++) begin
                            int py = r + ky - 1; // Padding = 1
                            int px = col + kx - 1;
                            
                            // Bounds check for Zero Padding
                            if (py >= 0 && py < ACT_SIZE && px >= 0 && px < ACT_SIZE) begin
                                // Convert to signed and handle Zero Point
                                val = $signed(pixel_mem[c][py*ACT_SIZE + px]);
                            end else begin
                                // Else: val is zp_in, no need to add to ref_acc
                                val = $signed(zp_in);        
                            end
                            weight_val = $signed(weight_mem[ky][kx][c]);
                            // Accumulate directly into the reference memory
                            ref_acc[r*ACT_SIZE + col] += val * weight_val;
                        end 
                        
                    end
                end
            end
        end
    endtask

    task automatic compare_results();
        logic signed [63:0] scaled;
        logic [N_BITS-1:0] expected;
        int errors = 0;

        for (int i = 0; i < ACT_SIZE*ACT_SIZE; i++) begin
            scaled = (ref_acc[i] + $signed(bias)) * m0; 
            expected = (scaled >>> n_shift) + $signed(zp_out); 
            
            if (res_mem[i] !== expected) begin
                $display("Error at pixel %d: Hardware=%h, Expected=%h", i, res_mem[i], expected);
                errors++;
            end
        end
        
        if (errors == 0) $display("PASS: All outputs match reference model!");
        else $display("FAIL: %d mismatches found.", errors);
    endtask

endmodule