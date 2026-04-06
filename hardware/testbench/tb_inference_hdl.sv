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
    localparam NUM_TEST_LAYERS = 17;

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

    // AXI stubs (TB_MODE=1, all tied off internally by top.sv)
    logic [12:0] s_axi_lite_awaddr;
    logic        s_axi_lite_awvalid;
    logic        s_axi_lite_awready;
    logic [31:0] s_axi_lite_wdata;
    logic [3:0]  s_axi_lite_wstrb;
    logic        s_axi_lite_wvalid;
    logic        s_axi_lite_wready;
    logic [1:0]  s_axi_lite_bresp;
    logic        s_axi_lite_bvalid;
    logic        s_axi_lite_bready;
    logic [12:0] s_axi_lite_araddr;
    logic        s_axi_lite_arvalid;
    logic        s_axi_lite_arready;
    logic [31:0] s_axi_lite_rdata;
    logic [1:0]  s_axi_lite_rresp;
    logic        s_axi_lite_rvalid;
    logic        s_axi_lite_rready;
    logic [31:0] s_axis_tdata;
    logic        s_axis_tvalid;
    logic        s_axis_tlast;
    logic [0:0]  s_axis_tuser;
    logic        s_axis_tready;
    logic        irq_done;

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

    // AXI tie-offs (TB_MODE=1: all unused)
    assign s_axi_lite_awaddr  = '0;
    assign s_axi_lite_awvalid = 1'b0;
    assign s_axi_lite_wdata   = '0;
    assign s_axi_lite_wstrb   = '0;
    assign s_axi_lite_wvalid  = 1'b0;
    assign s_axi_lite_bready  = 1'b0;
    assign s_axi_lite_araddr  = '0;
    assign s_axi_lite_arvalid = 1'b0;
    assign s_axi_lite_rready  = 1'b0;
    assign s_axis_tdata       = '0;
    assign s_axis_tvalid      = 1'b0;
    assign s_axis_tlast       = 1'b0;
    assign s_axis_tuser       = 1'b0;

    // =========================================================================
    //  DUT (TB_MODE=1: testbench mode, direct start/done/pixel_bram)
    // =========================================================================
    top #(
        .MAX_PARALLEL  (MAX_PARALLEL),
        .N_BITS        (N_BITS),
        .DEPTH_BITS    (DEPTH_BITS),
        .TB_MODE(1)
    ) dut (
        .aclk                (clk),
        .aresetn             (aresetn),
        .start               (start),
        .done                (done),
        .pixel_bram_addr     (pixel_bram_addr),
        .pixel_bram_en       (pixel_bram_en),
        .pixel_bram_data     (pixel_bram_data),
        .res_write_en        (res_write_en),
        .res_write_addr      (res_write_addr),
        .res_write_data      (res_write_data),
        .s_axi_lite_awaddr   (s_axi_lite_awaddr),
        .s_axi_lite_awvalid  (s_axi_lite_awvalid),
        .s_axi_lite_awready  (s_axi_lite_awready),
        .s_axi_lite_wdata    (s_axi_lite_wdata),
        .s_axi_lite_wstrb    (s_axi_lite_wstrb),
        .s_axi_lite_wvalid   (s_axi_lite_wvalid),
        .s_axi_lite_wready   (s_axi_lite_wready),
        .s_axi_lite_bresp    (s_axi_lite_bresp),
        .s_axi_lite_bvalid   (s_axi_lite_bvalid),
        .s_axi_lite_bready   (s_axi_lite_bready),
        .s_axi_lite_araddr   (s_axi_lite_araddr),
        .s_axi_lite_arvalid  (s_axi_lite_arvalid),
        .s_axi_lite_arready  (s_axi_lite_arready),
        .s_axi_lite_rdata    (s_axi_lite_rdata),
        .s_axi_lite_rresp    (s_axi_lite_rresp),
        .s_axi_lite_rvalid   (s_axi_lite_rvalid),
        .s_axi_lite_rready   (s_axi_lite_rready),
        .s_axis_tdata        (s_axis_tdata),
        .s_axis_tvalid       (s_axis_tvalid),
        .s_axis_tlast        (s_axis_tlast),
        .s_axis_tuser        (s_axis_tuser),
        .s_axis_tready       (s_axis_tready),
        .irq_done            (irq_done)
    );

    // =========================================================================
    //  Per-Layer Verification Config
    //
    //  URAM words = ceil(cout / 16) * h_out * w_out
    //  buf_sel: 0 = verify fmap_a, 1 = verify fmap_b
    //  wr_offset: URAM address offset for sub-pingpong branches
    // =========================================================================
    localparam int URAM_WORDS [0:NUM_TEST_LAYERS-1] = '{
        16384,  // L0:  128x128x16
        16384,  // L1:  128x128x16
         4096,  // L2:  64x64x16
         8192,  // L3:  64x64x32
         2048,  // L4:  32x32x32
         4096,  // L5:  32x32x64
         1024,  // L6:  16x16x64
         1024,  // L7:  16x16x64
          512,  // L8:  8x8x128
          512,  // L9:  8x8x128
          128,  // L10: 8x8x24   (CONV1)
          256,  // L11: 8x8x64   (cv2)
          256,  // L12: 8x8x64   (cv2)
          256,  // L13: 8x8x64   (cv2, CONV1_LIN)
          128,  // L14: 8x8x24   (cv3)
          128,  // L15: 8x8x24   (cv3)
           64   // L16: 8x8x3    (cv3, CONV1_LIN)
    };

    // pp_buf_sel per layer (0=fmap_a, 1=fmap_b)
    localparam bit PP_BUF_SEL [0:NUM_TEST_LAYERS-1] = '{
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  // backbone
        0,     // L10
        1, 0, 1,  // cv2: L11-L13
        1, 0, 1   // cv3: L14-L16
    };

    // Write offset per layer
    localparam int WR_OFFSET [0:NUM_TEST_LAYERS-1] = '{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // backbone
        0,         // L10
        256, 256, 256,  // cv2
        512, 512, 512   // cv3
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

    task automatic verify_uram_at_offset(
        input string buf_name,
        input int    num_words,
        input int    offset
    );
        integer errors = 0;
        logic [127:0] actual;

        $display("Checking %s[%0d..%0d] (%0d x 128-bit words)...",
                 buf_name, offset, offset + num_words - 1, num_words);
        for (int i = 0; i < num_words; i++) begin
            if (buf_name == "fmap_a")
                actual = dut.u_fmap_a.ram[offset + i];
            else
                actual = dut.u_fmap_b.ram[offset + i];

            if (actual !== golden_buf[i]) begin
                if (errors < 10)
                    $display("  [FAIL] %s[%0d] Expected: 0x%032h, Got: 0x%032h",
                             buf_name, offset + i, golden_buf[i], actual);
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
             0: $readmemh({MEM_PATH, "golden_layer0_uram.mem"},  golden_buf);
             1: $readmemh({MEM_PATH, "golden_layer1_uram.mem"},  golden_buf);
             2: $readmemh({MEM_PATH, "golden_layer2_uram.mem"},  golden_buf);
             3: $readmemh({MEM_PATH, "golden_layer3_uram.mem"},  golden_buf);
             4: $readmemh({MEM_PATH, "golden_layer4_uram.mem"},  golden_buf);
             5: $readmemh({MEM_PATH, "golden_layer5_uram.mem"},  golden_buf);
             6: $readmemh({MEM_PATH, "golden_layer6_uram.mem"},  golden_buf);
             7: $readmemh({MEM_PATH, "golden_layer7_uram.mem"},  golden_buf);
             8: $readmemh({MEM_PATH, "golden_layer8_uram.mem"},  golden_buf);
             9: $readmemh({MEM_PATH, "golden_layer9_uram.mem"},  golden_buf);
            10: $readmemh({MEM_PATH, "golden_layer10_uram.mem"}, golden_buf);
            11: $readmemh({MEM_PATH, "golden_layer11_uram.mem"}, golden_buf);
            12: $readmemh({MEM_PATH, "golden_layer12_uram.mem"}, golden_buf);
            13: $readmemh({MEM_PATH, "golden_layer13_uram.mem"}, golden_buf);
            14: $readmemh({MEM_PATH, "golden_layer14_uram.mem"}, golden_buf);
            15: $readmemh({MEM_PATH, "golden_layer15_uram.mem"}, golden_buf);
            16: $readmemh({MEM_PATH, "golden_layer16_uram.mem"}, golden_buf);
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
    initial begin
        $display("=========================================");
        $display(" tb_inference_hdl - %0d-layer pipeline test", NUM_TEST_LAYERS);
        $display("  Backbone:  layers 0-10 (CONV3 + CONV1)");
        $display("  cv2 branch: layers 11-13 (offset 256)");
        $display("  cv3 branch: layers 14-16 (offset 512)");
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
        for (int li = 0; li < NUM_TEST_LAYERS; li++) begin
            if (li < NUM_TEST_LAYERS - 1) begin
                // Wait for FSM to advance to next layer
                wait(dut.u_inference.layer_idx == li + 1);
                @(posedge clk);
            end else begin
                // Last layer: wait for done
                wait(done);
                #20;
            end

            $display("-----------------------------------------");
            $display("[CYCLE %0d] Layer %0d complete", cycle_count, li);

            // Load golden and verify at correct buffer + offset
            load_golden(li);
            if (PP_BUF_SEL[li] == 0) begin
                $display("Verifying Layer %0d output (fmap_a +%0d, %0d words)...",
                         li, WR_OFFSET[li], URAM_WORDS[li]);
                verify_uram_at_offset("fmap_a", URAM_WORDS[li], WR_OFFSET[li]);
            end else begin
                $display("Verifying Layer %0d output (fmap_b +%0d, %0d words)...",
                         li, WR_OFFSET[li], URAM_WORDS[li]);
                verify_uram_at_offset("fmap_b", URAM_WORDS[li], WR_OFFSET[li]);
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
