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
    //  Memories
    // =========================================================================
    // Golden reference for URAM-packed output (128-bit words)
    localparam L0_OUT_SIZE = 128;   // 128x128 after pool
    localparam L0_URAM_WORDS = L0_OUT_SIZE * L0_OUT_SIZE;  // 16384 (1 group x 128x128)
    localparam L1_OUT_SIZE = 128;   // 128x128 no pool
    localparam L1_URAM_WORDS = L1_OUT_SIZE * L1_OUT_SIZE;  // 16384

    logic signed [N_BITS-1:0] pixel_mem  [0:C_IN-1][0:ACT_SIZE*ACT_SIZE-1];
    logic [127:0] golden_layer0 [0:L0_URAM_WORDS-1];
    logic [127:0] golden_layer1 [0:L1_URAM_WORDS-1];
    // Keep single-channel golden for backwards compat
    logic signed [N_BITS-1:0] golden_mem [0:L0_OUT_SIZE*L0_OUT_SIZE-1];

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
    //  Result Capture (debug — primary verification is via URAM readback)
    // =========================================================================

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

    task automatic verify_qp_regs();
        // Expected values for layer 0, kout 0
        logic signed [31:0] exp_bias   = 32'sd5035;
        logic        [31:0] exp_m0     = 32'h4d9185ff;
        logic         [5:0] exp_nshift = 6'd41;
        logic signed  [7:0] exp_zp_in  = -8'sd128;
        logic signed  [7:0] exp_zp_out = 8'sd3;
        integer errors = 0;

        $display("Checking QP shadow registers (layer 0, output channel 0).");

        if (dut.u_inference.r_bias === exp_bias)
            $display("  %s r_bias   = %0d (0x%08h)", "[PASS]", dut.u_inference.r_bias, dut.u_inference.r_bias);
        else begin
            $display("  %s r_bias   Expected: %0d, Got: %0d", "[FAIL]", exp_bias, dut.u_inference.r_bias);
            errors++;
        end

        if (dut.u_inference.r_m0 === exp_m0)
            $display("  %s r_m0     = %0d (0x%08h)", "[PASS]", dut.u_inference.r_m0, dut.u_inference.r_m0);
        else begin
            $display("  %s r_m0     Expected: 0x%08h, Got: 0x%08h", "[FAIL]", exp_m0, dut.u_inference.r_m0);
            errors++;
        end

        if (dut.u_inference.r_nshift === exp_nshift)
            $display("  %s r_nshift = %0d", "[PASS]", dut.u_inference.r_nshift);
        else begin
            $display("  %s r_nshift Expected: %0d, Got: %0d", "[FAIL]", exp_nshift, dut.u_inference.r_nshift);
            errors++;
        end

        if (dut.u_inference.r_cfg.zp_in === exp_zp_in)
            $display("  %s zp_in    = %0d", "[PASS]", dut.u_inference.r_cfg.zp_in);
        else begin
            $display("  %s zp_in    Expected: %0d, Got: %0d", "[FAIL]", exp_zp_in, dut.u_inference.r_cfg.zp_in);
            errors++;
        end

        if (dut.u_inference.r_cfg.zp_out === exp_zp_out)
            $display("  %s zp_out   = %0d", "[PASS]", dut.u_inference.r_cfg.zp_out);
        else begin
            $display("  %s zp_out   Expected: %0d, Got: %0d", "[FAIL]", exp_zp_out, dut.u_inference.r_cfg.zp_out);
            errors++;
        end

        if (errors == 0) $display("QP SHADOW REGISTER CHECKS PASSED");
        else begin
            $display("QP SHADOW REGISTER CHECKS FAILED: %0d errors", errors);
            $fatal(1, "QP verification failed - stopping simulation");
        end
    endtask

    task automatic verify_uram(
        input string buf_name,
        input int    num_words,
        input logic [127:0] expected [0:16383]
    );
        integer errors = 0;
        logic [127:0] actual;

        $display("Checking %s (%0d x 128-bit words)...", buf_name, num_words);
        for (int i = 0; i < num_words; i++) begin
            if (buf_name == "fmap_a")
                actual = dut.u_fmap_a.ram[i];
            else
                actual = dut.u_fmap_b.ram[i];

            if (actual !== expected[i]) begin
                if (errors < 10)
                    $display("  [FAIL] %s[%0d] Expected: 0x%032h, Got: 0x%032h",
                             buf_name, i, expected[i], actual);
                errors++;
            end
        end

        if (errors == 0)
            $display("  [PASS] %s: all %0d words match", buf_name, num_words);
        else
            $display("  [FAIL] %s: %0d / %0d mismatches", buf_name, errors, num_words);
    endtask

    // =========================================================================
    //  Main Test Sequence
    // =========================================================================
    initial begin
        $display("=========================================");
        $display(" tb_inference_hdl - 2-layer pipeline test");
        $display("  Layer 0: CONV3_POOL 256x256x3 -> 128x128x16");
        $display("  Layer 1: CONV3     128x128x16 -> 128x128x16");
        $display("  Parallelism: %0d convolvers", MAX_PARALLEL);
        $display("=========================================");

        // 1. Load test data
        $display("[INIT] Loading test data...");
        $readmemh({MEM_PATH, "pixels_layer0.mem"}, pixel_mem);
        $readmemh({MEM_PATH, "golden_ch_out0.mem"}, golden_mem);
        $readmemh({MEM_PATH, "golden_layer0_uram.mem"}, golden_layer0);
        $readmemh({MEM_PATH, "golden_layer1_uram.mem"}, golden_layer1);
        $display("[INIT] Pixel mem:     %0d channels x %0d pixels", C_IN, ACT_SIZE*ACT_SIZE);
        $display("[INIT] Golden URAM L0: %0d words", L0_URAM_WORDS);
        $display("[INIT] Golden URAM L1: %0d words", L1_URAM_WORDS);
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

        // 4. Wait for loading, verify QP
        repeat(15) @(posedge clk);
        verify_qp_regs();

        // 5. Wait for full 2-layer inference
        $display("[CYCLE %0d] Running 2-layer inference...", cycle_count);
        wait(done);
        $display("[CYCLE %0d] Inference complete", cycle_count);
        #100;

        // 6. Verify URAM contents
        $display("-----------------------------------------");
        $display("Verifying Layer 0 output (fmap_a, 128x128x16)...");
        verify_uram("fmap_a", L0_URAM_WORDS, golden_layer0);

        $display("Verifying Layer 1 output (fmap_b, 128x128x16)...");
        verify_uram("fmap_b", L1_URAM_WORDS, golden_layer1);

        // 7. Summary
        $display("=========================================");
        $display(" Summary:");
        $display("  Total cycles:      %0d", cycle_count - start_cycle);
        $display("=========================================");

        #100;
        $finish;
    end

    // Timeout watchdog
    initial begin
        #200_000_000;   // 200ms timeout
        $display("TIMEOUT: simulation exceeded 200ms");
        $finish;
    end

endmodule
