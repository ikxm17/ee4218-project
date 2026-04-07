`timescale 1ns / 1ps

module tb_conv3d;

    localparam MEM_PATH = "../../../../../../testbench/conv3d/";

    // Parameters matching DUT
    parameter ACT_SIZE     = 256;  // Toggle between 256 and 16 for your test cases
    parameter K            = 3;
    parameter STRIDE       = 1;
    parameter C_IN         = 3; // Toggle between 3 and 128
    parameter MAX_PARALLEL = 2;
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

    logic req_weights;
    logic weights_ready;

    // Runtime layer dimensions (conv3d takes these as input ports now,
    // not compile-time parameters).  Lowercase names match the conv3d
    // port names so the .* implicit connection picks them up.
    wire [8:0] act_size = ACT_SIZE[8:0];
    wire [7:0] cin      = C_IN[7:0];

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
    // Note: conv3d's pixel_bram_addr is a flat DEPTH_BITS-wide port now
    // (was previously scaled by C_IN/MAX_PARALLEL when conv3d was per-layer).
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
    // ACT_SIZE and C_IN are no longer parameters of conv3d — they're driven
    // by runtime input ports (act_size, cin) that .* picks up by name.
    conv3d #(
        .K(K), .STRIDE(STRIDE),
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
    // always_comb begin
    //     weights_all_channels = '0;
    //     for (int i = 0; i < MAX_PARALLEL; i++) begin
    //         automatic int current_in_ch = (dut.round * MAX_PARALLEL) + i;
    //         if (current_in_ch < C_IN) begin
    //             // Flatten the KxK kernel for the specific input channel
    //             for (int kh = 0; kh < K; kh++) begin
    //                 for (int kw = 0; kw < K; kw++) begin
    //                     // Indexing: channel i's K*K block, specifically the kh,kw bit range
    //                     weights_all_channels[(i * K * K * N_BITS) + (kh * K + kw) * N_BITS +: N_BITS] 
    //                         = weight_mem[kh][kw][current_in_ch];
    //                 end
    //             end
    //         end
    //     end
    // end

    // Task to manage weight updates for Round 1 and beyond
    task automatic handle_weight_requests();
        forever begin
            @(posedge clk);
            if (req_weights) begin
                // The DUT is asking for the NEXT set of weights. 
                // We pass the current dut.round + 1 to calculate the next round's weights.
                update_weights_buffer(dut.round+1);
                
                // Assert ready for exactly one cycle
                weights_ready <= 1'b1;
                @(posedge clk);
                weights_ready <= 1'b0;
            end
        end
    endtask

    // Logic to format weights based on a specific round index
    task automatic update_weights_buffer(input int round_idx);
        weights_all_channels = '0; 
        
        for (int i = 0; i < MAX_PARALLEL; i++) begin
            automatic int current_in_ch = (round_idx * MAX_PARALLEL) + i;
            
            if (current_in_ch < C_IN) begin
                for (int kh = 0; kh < K; kh++) begin
                    for (int kw = 0; kw < K; kw++) begin
                        weights_all_channels[(i * K * K * N_BITS) + (kh * K + kw) * N_BITS +: N_BITS] 
                            = weight_mem[kh][kw][current_in_ch];
                    end
                end
            end
        end
    endtask
    
    // -------------------------------------------------------------------------
    // BRAM Data Feeding Logic (Synchronous Read)
    // -------------------------------------------------------------------------
    // Most BRAMs have a 1-cycle latency. 
    // We check pixel_bram_en and return MAX_PARALLEL channels at pixel_bram_addr
    always_ff @(posedge clk) begin
        if (pixel_bram_en) begin
            // pre-fill with zeros in for channels that are not active
            pixel_bram_data <= '0;

            for (int i = 0; i < MAX_PARALLEL; i++) begin
                automatic int current_in_ch = (dut.round * MAX_PARALLEL) + i;
                if (current_in_ch < C_IN) begin
                    // using a modulo to handle pixel_bram_addr for subsequent rounds
                    // in actual implementation, pixel_bram_addr should be used directly as the RAM should be N_BITS*MAX_PARALLEL wide with depth = (C_IN/ MAX_PARALLEL) * (ACT_SIZE*ACT_SIZE)
                    pixel_bram_data[i*N_BITS +: N_BITS] <= pixel_mem[current_in_ch][pixel_bram_addr % (ACT_SIZE * ACT_SIZE)];
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
        $readmemh({MEM_PATH, "zp_in.mem"}, zp_in_file);   zp_in   = zp_in_file[0];
        $readmemh({MEM_PATH, "zp_out.mem"}, zp_out_file); zp_out  = zp_out_file[0];
        $readmemh({MEM_PATH, "bias.mem"}, bias_file);     bias    = bias_file[0];
        $readmemh({MEM_PATH, "m0.mem"}, m0_file);         m0      = m0_file[0];
        $readmemh({MEM_PATH, "n_shift.mem"}, n_shift_file); n_shift = n_shift_file[0];

        // Loading Multi-dimensional arrays (requires specific file formatting)
        $readmemh({MEM_PATH, "pixels.mem"}, pixel_mem);
        $readmemh({MEM_PATH, "weights.mem"}, weight_mem);

        // 2. Drive Reset and Start
        rst = 1; start = 0;
        weights_ready = 0; 
        update_weights_buffer(0);         // Pre-load round 0 weights manually before execution begins

        #20 rst = 0;
        #10 start = 1;
        #10 start = 0;

        // if req_weights asserted, load new weights and assert weights_ready for 1 cycle
        // Parallel Handshake Logic for subsequent rounds (Round 1+)
        fork
            handle_weight_requests();
        join_none

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