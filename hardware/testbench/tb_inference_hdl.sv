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
    //  Layer 1 Pipeline Debug Monitor
    //
    //  Traces data at key pipeline stages during layer 1, ch_out=0 to
    //  pinpoint where the computation diverges from the golden reference.
    // =========================================================================
    integer l1_read_cnt, l1_acc_cnt, l1_silu_cnt;
    logic   l1_started;

    initial begin
        l1_read_cnt = 0;
        l1_acc_cnt  = 0;
        l1_silu_cnt = 0;
        l1_started  = 0;
    end

    always_ff @(posedge clk) begin
        if (dut.u_inference.curr_layer_idx == 5'd1 &&
            dut.u_inference.ch_out == 8'd0) begin

            // Print QP values when conv3d starts for layer 1 ch_out=0
            if (dut.u_inference.conv3d_start && !l1_started) begin
                l1_started <= 1;
                $display("\n=== L1 CH0 CONV START (cycle %0d) ===", cycle_count);
                $display("  bias=%0d (0x%08h)",
                         $signed(dut.u_inference.r_bias),
                         dut.u_inference.r_bias);
                $display("  m0=0x%08h, nshift=%0d",
                         dut.u_inference.r_m0,
                         dut.u_inference.r_nshift);
                $display("  zp_in=%0d, zp_out=%0d",
                         $signed(dut.u_inference.r_cfg.zp_in),
                         $signed(dut.u_inference.r_cfg.zp_out));
                $display("  fmap_a[0]=0x%032h (expected input at addr 0)",
                         dut.u_fmap_a.ram[0]);
                $display("  QP ROM[0]=0x%018h, ROM[16]=0x%018h",
                         dut.u_qp_mem.ram[0], dut.u_qp_mem.ram[16]);
                $display("  conv3d.r_bias=%0d, conv3d.r_m0=0x%08h, conv3d.r_n_shift=%0d",
                         $signed(dut.u_inference.u_conv3d.r_bias),
                         dut.u_inference.u_conv3d.r_m0,
                         dut.u_inference.u_conv3d.r_n_shift);
            end

            // Print first 3 non-padding pixel data seen by conv3d
            if (dut.u_inference.u_conv3d.conv_running &&
                !dut.u_inference.u_conv3d.is_padded_act &&
                l1_read_cnt < 3) begin
                $display("  [L1 PIXRD %0d] pixel_bram_data=0x%032h",
                         l1_read_cnt,
                         dut.u_inference.u_conv3d.pixel_bram_data);
                l1_read_cnt <= l1_read_cnt + 1;
            end

            // Print first 5 accumulator outputs
            if (dut.u_inference.u_conv3d.ACC_write_en &&
                l1_acc_cnt < 5) begin
                $display("  [L1 ACC %0d] addr=%0d, acc_in=%0d (0x%08h), round=%0d, q_pix=%0d",
                         l1_acc_cnt,
                         dut.u_inference.u_conv3d.ACC_write_address,
                         $signed(dut.u_inference.u_conv3d.ACC_write_data_in),
                         dut.u_inference.u_conv3d.ACC_write_data_in,
                         dut.u_inference.u_conv3d.round,
                         $signed(dut.u_inference.u_conv3d.q_pix));
                l1_acc_cnt <= l1_acc_cnt + 1;
            end
        end

        // Post-SiLU output for layer 1 ch_out=0
        if (dut.u_inference.curr_layer_idx == 5'd1 &&
            dut.u_inference.ch_out == 8'd0 &&
            dut.u_activation.out_valid &&
            l1_silu_cnt < 5) begin
            $display("  [L1 SILU %0d] addr=%0d, data=%0d (0x%02h), lut_addr=0x%04h",
                     l1_silu_cnt,
                     dut.u_activation.out_addr,
                     $signed(dut.u_activation.out_data),
                     dut.u_activation.out_data[7:0],
                     dut.u_activation.lut_addr);
            l1_silu_cnt <= l1_silu_cnt + 1;
        end
    end

    // =========================================================================
    //  QP ROM Read Trace — Layer Transition
    //
    //  Monitors the QP ROM port signals during S_NEXT_LAYER → S_LOAD to
    //  verify the address/enable/data timing for the layer 1 QP read.
    // =========================================================================
    logic qp_trace_armed;
    integer qp_trace_cnt;
    initial begin qp_trace_armed = 0; qp_trace_cnt = 0; end

    always_ff @(posedge clk) begin
        // Arm when layer_idx is about to transition (state=S_NEXT_CHOUT, last ch_out)
        if (dut.u_inference.state == 3'd3 &&                // S_NEXT_CHOUT
            dut.u_inference.layer_idx == 5'd0 &&             // still layer 0
            dut.u_inference.ch_out + 8'd1 >= dut.u_inference.r_cfg.cout) begin
            qp_trace_armed <= 1;
            qp_trace_cnt   <= 0;
        end

        // Print 15 cycles of QP ROM signals after arming
        if (qp_trace_armed && qp_trace_cnt < 15) begin
            $display("  [QP TRACE %0d] state=%0d, qp_en=%b, qp_addr=%0d, qp_dout=0x%018h, r_bias=%0d, load_cnt=%0d",
                     qp_trace_cnt,
                     dut.u_inference.state,
                     dut.u_qp_mem.en_b,
                     dut.u_qp_mem.addr_b,
                     dut.u_qp_mem.dout_b,
                     $signed(dut.u_inference.r_bias),
                     dut.u_inference.load_cnt);
            qp_trace_cnt <= qp_trace_cnt + 1;
            if (qp_trace_cnt == 14) qp_trace_armed <= 0;
        end
    end

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

        // 6. Diagnostic dumps
        $display("-----------------------------------------");
        $display("=== DIAGNOSTICS ===");

        // Dump first 4 URAM words from fmap_a (layer 0 output)
        $display("fmap_a[0:3] (layer 0 output, first 4 words):");
        for (int d = 0; d < 4; d++)
            $display("  fmap_a[%0d] = 0x%032h", d, dut.u_fmap_a.ram[d]);

        // Dump first 4 URAM words from fmap_b (layer 1 output)
        $display("fmap_b[0:3] (layer 1 output, first 4 words):");
        for (int d = 0; d < 4; d++)
            $display("  fmap_b[%0d] = 0x%032h", d, dut.u_fmap_b.ram[d]);

        // Check golden layer 0 vs fmap_a for first 4 words
        $display("golden_layer0[0:3] (expected layer 0 output):");
        for (int d = 0; d < 4; d++)
            $display("  golden_l0[%0d] = 0x%032h", d, golden_layer0[d]);

        // Check golden layer 1 vs fmap_b for first 4 words
        $display("golden_layer1[0:3] (expected layer 1 output):");
        for (int d = 0; d < 4; d++)
            $display("  golden_l1[%0d] = 0x%032h", d, golden_layer1[d]);

        // Check layer 1 QP registers (should be layer 1, last ch_out=15)
        $display("Final state: layer_idx=%0d, ch_out=%0d",
                 dut.u_inference.layer_idx, dut.u_inference.ch_out);
        $display("  r_cfg.zp_in=%0d, r_cfg.zp_out=%0d",
                 dut.u_inference.r_cfg.zp_in, dut.u_inference.r_cfg.zp_out);
        $display("  r_cfg.h_in=%0d, r_cfg.cin=%0d, r_cfg.cout=%0d",
                 dut.u_inference.r_cfg.h_in, dut.u_inference.r_cfg.cin,
                 dut.u_inference.r_cfg.cout);

        $display("=== END DIAGNOSTICS ===");
        $display("-----------------------------------------");

        // 7. Verify URAM contents
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
