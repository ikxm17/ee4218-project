`timescale 1ns / 1ps

module tb_inference_hdl;

    localparam MEM_PATH = "";

    // Parameters matching DUT (layer 0)
    parameter ACT_SIZE     = 256;
    parameter POOL_OUT     = ACT_SIZE / 2;   // layer 0 is CONV3_POOL → 128x128
    parameter K            = 3;
    parameter C_IN         = 3;
    parameter MAX_PARALLEL = 16;
    parameter N_BITS       = 8;
    parameter ACC_BITS     = 32;
    parameter DEPTH_BITS   = 16;

    // Number of layers to test
    localparam NUM_TEST_LAYERS = 10;

    // Clock and Reset
    logic clk;
    logic aresetn;
    logic start;
    logic done;

    // Pixel BRAM interface
    logic [DEPTH_BITS-1:0]              pixel_bram_addr;
    logic                               pixel_bram_en;
    logic [MAX_PARALLEL*N_BITS-1:0]     pixel_bram_data;

    // RES output interface
    logic                               res_write_en;
    logic [DEPTH_BITS-1:0]              res_write_addr;
    logic signed [N_BITS-1:0]           res_write_data;

    // AXI-Stream stubs
    logic s_axis_tvalid, s_axis_tlast, s_axis_tready;
    logic m_axis_tready, m_axis_tvalid, m_axis_tlast;
    logic [23:0] s_axis_tdata, m_axis_tdata;

    // =========================================================================
    //  Clock & Cycle Counter
    // =========================================================================
    integer cycle_count;
    integer start_cycle;

    initial clk = 0;
    always #5 clk = ~clk;  // 10ns period, 100 MHz

    always_ff @(posedge clk or negedge aresetn) begin
        if (!aresetn)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;
    end

    // AXI-Stream tie-offs
    assign s_axis_tvalid = 1'b0;
    assign s_axis_tlast  = 1'b0;
    assign s_axis_tdata  = '0;
    assign m_axis_tready = 1'b0;

    // =========================================================================
    //  DUT
    // =========================================================================
    top #(
        .MAX_PARALLEL (MAX_PARALLEL),
        .N_BITS       (N_BITS),
        .DEPTH_BITS   (DEPTH_BITS)
    ) dut (
        .aclk            (clk),
        .aresetn         (aresetn),
        .start           (start),
        .done            (done),
        .s_axis_tvalid   (s_axis_tvalid),
        .s_axis_tlast    (s_axis_tlast),
        .s_axis_tdata    (s_axis_tdata),
        .s_axis_tready   (s_axis_tready),
        .m_axis_tready   (m_axis_tready),
        .m_axis_tvalid   (m_axis_tvalid),
        .m_axis_tlast    (m_axis_tlast),
        .m_axis_tdata    (m_axis_tdata),
        .pixel_bram_addr (pixel_bram_addr),
        .pixel_bram_en   (pixel_bram_en),
        .pixel_bram_data (pixel_bram_data),
        .res_write_en    (res_write_en),
        .res_write_addr  (res_write_addr),
        .res_write_data  (res_write_data)
    );

    // =========================================================================
    //  Per-Layer URAM Word Counts
    //
    //  URAM words = ceil(cout / 16) * h_out * w_out
    //  Pool layers output h_in/2 × w_in/2
    // =========================================================================
    localparam int URAM_WORDS [0:NUM_TEST_LAYERS-1] = '{
        16384,  // L0:  128x128x16  = 1 * 128*128
        16384,  // L1:  128x128x16  = 1 * 128*128
         4096,  // L2:  64x64x16    = 1 * 64*64
         8192,  // L3:  64x64x32    = 2 * 64*64
         2048,  // L4:  32x32x32    = 2 * 32*32
         4096,  // L5:  32x32x64    = 4 * 32*32
         1024,  // L6:  16x16x64    = 4 * 16*16
         1024,  // L7:  16x16x64    = 4 * 16*16
          512,  // L8:  8x8x128     = 8 * 8*8
          512   // L9:  8x8x128     = 8 * 8*8
    };

    // =========================================================================
    //  Memories
    // =========================================================================
    // Input pixels
    logic signed [N_BITS-1:0] pixel_mem [0:C_IN-1][0:ACT_SIZE*ACT_SIZE-1];

    // Golden URAM data — one buffer, reloaded per layer
    logic [127:0] golden_buf [0:16383];

    // Keep single-channel golden for backwards compat
    logic signed [N_BITS-1:0] golden_mem [0:POOL_OUT*POOL_OUT-1];

    // =========================================================================
    //  Pixel BRAM Feeding (1-cycle latency)
    // =========================================================================
    always_ff @(posedge clk) begin
        if (pixel_bram_en) begin
            pixel_bram_data <= '0;
            for (int i = 0; i < MAX_PARALLEL; i++) begin
                if (i < C_IN)
                    pixel_bram_data[i*N_BITS +: N_BITS] <=
                        pixel_mem[i][pixel_bram_addr % (ACT_SIZE * ACT_SIZE)];
            end
        end else begin
            pixel_bram_data <= '0;
        end
    end

    // =========================================================================
    //  Verification Tasks
    // =========================================================================

    task automatic verify_rom_contents();
        logic [71:0]  qp_word0;
        logic [127:0] wt_word0;
        // Expected values (from generate_conv3d_golden.py output)
        logic [71:0]  qp_expected = 72'h294d9185ff000013ab;
        logic [127:0] wt_expected = 128'h00000000000000000000000000cda9b6;
        integer errors = 0;

        qp_word0 = dut.u_qp_mem.ram[0];
        wt_word0 = dut.u_wt_mem.ram[0];

        $display("Checking ROM contents loaded from .mem files.");
        if (qp_word0 === qp_expected)
            $display("  %s QP ROM[0]  = 0x%018h", "[PASS]", qp_word0);
        else begin
            $display("  %s QP ROM[0]  Expected: 0x%018h, Got: 0x%018h", "[FAIL]", qp_expected, qp_word0);
            errors++;
        end

        if (wt_word0 === wt_expected)
            $display("  %s WT ROM[0]  = 0x%032h", "[PASS]", wt_word0);
        else begin
            $display("  %s WT ROM[0]  Expected: 0x%032h, Got: 0x%032h", "[FAIL]", wt_expected, wt_word0);
            errors++;
        end

        if (errors == 0) $display("ROM CHECKS PASSED");
        else $fatal(1, "ROM CHECKS FAILED - .mem files not loaded correctly");
    endtask

    task automatic verify_uram(
        input string buf_name,
        input int    num_words
    );
        integer errors = 0;
        logic [127:0] actual;

        $display("Checking %s (%0d x 128-bit words)...", buf_name, num_words);
        for (int i = 0; i < num_words; i++) begin
            if (buf_name == "fmap_a")
                actual = dut.u_fmap_a.ram[i];
            else
                actual = dut.u_fmap_b.ram[i];

            if (actual !== golden_buf[i]) begin
                if (errors < 10)
                    $display("  [FAIL] %s[%0d] Expected: 0x%032h, Got: 0x%032h",
                             buf_name, i, golden_buf[i], actual);
                errors++;
            end
        end

        if (errors == 0)
            $display("  [PASS] %s: all %0d words match", buf_name, num_words);
        else
            $display("  [FAIL] %s: %0d / %0d mismatches", buf_name, errors, num_words);
    endtask

    task automatic load_golden(input int layer_idx);
        case (layer_idx)
            0: $readmemh({MEM_PATH, "golden_layer0_uram.mem"}, golden_buf);
            1: $readmemh({MEM_PATH, "golden_layer1_uram.mem"}, golden_buf);
            2: $readmemh({MEM_PATH, "golden_layer2_uram.mem"}, golden_buf);
            3: $readmemh({MEM_PATH, "golden_layer3_uram.mem"}, golden_buf);
            4: $readmemh({MEM_PATH, "golden_layer4_uram.mem"}, golden_buf);
            5: $readmemh({MEM_PATH, "golden_layer5_uram.mem"}, golden_buf);
            6: $readmemh({MEM_PATH, "golden_layer6_uram.mem"}, golden_buf);
            7: $readmemh({MEM_PATH, "golden_layer7_uram.mem"}, golden_buf);
            8: $readmemh({MEM_PATH, "golden_layer8_uram.mem"}, golden_buf);
            9: $readmemh({MEM_PATH, "golden_layer9_uram.mem"}, golden_buf);
        endcase
    endtask

    // =========================================================================
    //  Debug Monitors (enable with +define+DEBUG_TRACE in xsim)
    // =========================================================================
`ifdef DEBUG_TRACE
    always_ff @(posedge clk) begin
        if (dut.u_inference.state == 3'd4) begin // S_NEXT_LAYER
            $display("[DBG cycle %0d] S_NEXT_LAYER: layer_idx=%0d -> %0d",
                     cycle_count,
                     dut.u_inference.layer_idx,
                     dut.u_inference.layer_idx + 1);
        end
    end
`endif

    // =========================================================================
    //  Main Test Sequence
    // =========================================================================
    integer total_errors;

    initial begin
        $display("=========================================");
        $display(" tb_inference_hdl - %0d-layer pipeline test", NUM_TEST_LAYERS);
        for (int li = 0; li < NUM_TEST_LAYERS; li++) begin
            $display("  Layer %0d: %0d URAM words", li, URAM_WORDS[li]);
        end
        $display("  Parallelism: %0d convolvers", MAX_PARALLEL);
        $display("=========================================");

        // 1. Load test data
        $display("[INIT] Loading test data...");
        $readmemh({MEM_PATH, "pixels_layer0.mem"}, pixel_mem);
        $readmemh({MEM_PATH, "golden_ch_out0.mem"}, golden_mem);
        $display("[INIT] Pixel mem: %0d channels x %0d pixels", C_IN, ACT_SIZE*ACT_SIZE);
        #1;
        verify_rom_contents();

        // 2. Reset
        aresetn = 0; start = 0;
        #100;
        aresetn = 1;
        #20;

        // 3. Start inference (mid-cycle to avoid posedge race)
        $display("-----------------------------------------");
        #10; start = 1;
        start_cycle = cycle_count;
        $display("[CYCLE %0d] Start pulse asserted", cycle_count);
        #10; start = 0;

        // 4. Run and verify each layer as it completes
        total_errors = 0;

        for (int li = 0; li < NUM_TEST_LAYERS; li++) begin
            if (li < NUM_TEST_LAYERS - 1) begin
                // Wait for FSM to advance to next layer
                wait(dut.u_inference.layer_idx == li + 1);
                @(posedge clk); // one extra cycle for URAM write to settle
            end else begin
                // Last layer: wait for done
                wait(done);
                #20;
            end

            $display("-----------------------------------------");
            $display("[CYCLE %0d] Layer %0d complete", cycle_count, li);

            // Load golden for this layer and verify
            load_golden(li);
            if (li[0] == 0) begin
                $display("Verifying Layer %0d output (fmap_a, %0d words)...", li, URAM_WORDS[li]);
                verify_uram("fmap_a", URAM_WORDS[li]);
            end else begin
                $display("Verifying Layer %0d output (fmap_b, %0d words)...", li, URAM_WORDS[li]);
                verify_uram("fmap_b", URAM_WORDS[li]);
            end
        end

        // 5. Summary
        $display("=========================================");
        $display(" Summary:");
        $display("  Total cycles:      %0d", cycle_count - start_cycle);
        $display("  Layers verified:   %0d", NUM_TEST_LAYERS);
        $display("=========================================");

        #100;
        $finish;
    end

    // Timeout watchdog
    initial begin
        #500_000_000;   // 500ms timeout (~50M cycles)
        $display("TIMEOUT: simulation exceeded 500ms");
        $finish;
    end

endmodule
