`timescale 1ns / 1ps

/*
 * tb_tinyissimoyolo_accel — AXI integration testbench
 *
 * Tests the USE_FMAP_INPUT=1 path:
 *   1. Preload packed pixels into fmap_b via AXI-Lite FIFO writes
 *   2. Run full 17-layer inference
 *   3. Read detection results via AXI-Lite
 *   4. Compare against golden URAM data
 *
 * Validates that the packed pixel preload + byte extraction path
 * produces identical results to the direct pixel BRAM path.
 */
module tb_tinyissimoyolo_accel;

    localparam MEM_PATH = "";

    parameter ACT_SIZE     = 256;
    parameter C_IN         = 3;
    parameter MAX_PARALLEL = 16;
    parameter N_BITS       = 8;
    parameter DEPTH_BITS   = 16;
    parameter AXI_ADDR_W   = 13;

    localparam NUM_TEST_LAYERS = 17;

    // =========================================================================
    //  Clock & Reset
    // =========================================================================
    logic clk;
    logic aresetn;

    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle_count;
    always_ff @(posedge clk or negedge aresetn) begin
        if (!aresetn) cycle_count <= 0;
        else          cycle_count <= cycle_count + 1;
    end

    // =========================================================================
    //  AXI-Lite signals
    // =========================================================================
    logic [AXI_ADDR_W-1:0] s_axi_lite_awaddr;
    logic                   s_axi_lite_awvalid;
    logic                   s_axi_lite_awready;
    logic [31:0]            s_axi_lite_wdata;
    logic [3:0]             s_axi_lite_wstrb;
    logic                   s_axi_lite_wvalid;
    logic                   s_axi_lite_wready;
    logic [1:0]             s_axi_lite_bresp;
    logic                   s_axi_lite_bvalid;
    logic                   s_axi_lite_bready;
    logic [AXI_ADDR_W-1:0] s_axi_lite_araddr;
    logic                   s_axi_lite_arvalid;
    logic                   s_axi_lite_arready;
    logic [31:0]            s_axi_lite_rdata;
    logic [1:0]             s_axi_lite_rresp;
    logic                   s_axi_lite_rvalid;
    logic                   s_axi_lite_rready;

    // S_AXIS (tied off — testing FIFO mode, not camera)
    logic [31:0] s_axis_tdata;
    logic        s_axis_tvalid;
    logic        s_axis_tlast;
    logic [0:0]  s_axis_tuser;
    logic        s_axis_tready;

    assign s_axis_tdata  = '0;
    assign s_axis_tvalid = 1'b0;
    assign s_axis_tlast  = 1'b0;
    assign s_axis_tuser  = 1'b0;

    // Testbench ports (unused in USE_FMAP_INPUT=1, but must be connected)
    logic                             tb_done;
    logic [DEPTH_BITS-1:0]            tb_pixel_bram_addr;
    logic                             tb_pixel_bram_en;
    logic [MAX_PARALLEL*N_BITS-1:0]   tb_pixel_bram_data;
    logic                             tb_res_write_en;
    logic [DEPTH_BITS-1:0]            tb_res_write_addr;
    logic signed [N_BITS-1:0]         tb_res_write_data;
    logic                             irq_done;

    assign tb_pixel_bram_data = '0;

    // =========================================================================
    //  DUT (USE_FMAP_INPUT=1: AXI IP mode)
    // =========================================================================
    top #(
        .MAX_PARALLEL   (MAX_PARALLEL),
        .N_BITS         (N_BITS),
        .DEPTH_BITS     (DEPTH_BITS),
        .USE_FMAP_INPUT (1),
        .AXI_ADDR_W     (AXI_ADDR_W)
    ) dut (
        .aclk                (clk),
        .aresetn             (aresetn),
        .start               (1'b0),
        .done                (tb_done),
        .pixel_bram_addr     (tb_pixel_bram_addr),
        .pixel_bram_en       (tb_pixel_bram_en),
        .pixel_bram_data     (tb_pixel_bram_data),
        .res_write_en        (tb_res_write_en),
        .res_write_addr      (tb_res_write_addr),
        .res_write_data      (tb_res_write_data),
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
    //  Register addresses (must match axil_regs.sv)
    // =========================================================================
    localparam [AXI_ADDR_W-1:0] ADDR_CTRL        = 13'h000,
                                ADDR_STATUS      = 13'h004,
                                ADDR_MODE        = 13'h008,
                                ADDR_PIXEL_FIFO  = 13'h020,
                                ADDR_PIXEL_CNT   = 13'h024,
                                ADDR_CYCLE_CNT   = 13'h00C,
                                ADDR_RESULT_BASE = 13'h100;

    // =========================================================================
    //  AXI-Lite BFM Tasks
    // =========================================================================

    task automatic axil_write(
        input [AXI_ADDR_W-1:0] addr,
        input [31:0]           data
    );
        @(posedge clk);
        s_axi_lite_awaddr  <= addr;
        s_axi_lite_awvalid <= 1'b1;
        s_axi_lite_wdata   <= data;
        s_axi_lite_wstrb   <= 4'hF;
        s_axi_lite_wvalid  <= 1'b1;
        s_axi_lite_bready  <= 1'b1;

        // Wait for handshake
        @(posedge clk);
        while (!(s_axi_lite_awready && s_axi_lite_wready))
            @(posedge clk);

        s_axi_lite_awvalid <= 1'b0;
        s_axi_lite_wvalid  <= 1'b0;

        // Wait for response
        while (!s_axi_lite_bvalid)
            @(posedge clk);
        @(posedge clk);
        s_axi_lite_bready <= 1'b0;
    endtask

    task automatic axil_read(
        input  [AXI_ADDR_W-1:0] addr,
        output [31:0]            data
    );
        @(posedge clk);
        s_axi_lite_araddr  <= addr;
        s_axi_lite_arvalid <= 1'b1;
        s_axi_lite_rready  <= 1'b1;

        // Wait for address accepted
        @(posedge clk);
        while (!s_axi_lite_arready)
            @(posedge clk);
        s_axi_lite_arvalid <= 1'b0;

        // Wait for data valid
        while (!s_axi_lite_rvalid)
            @(posedge clk);
        data = s_axi_lite_rdata;
        @(posedge clk);
        s_axi_lite_rready <= 1'b0;
    endtask

    // =========================================================================
    //  Pixel data and golden reference
    // =========================================================================
    logic signed [N_BITS-1:0] pixel_mem [0:C_IN-1][0:ACT_SIZE*ACT_SIZE-1];
    logic [127:0] golden_buf [0:16383];

    // Per-layer verification config (same as tb_inference_hdl)
    localparam int URAM_WORDS [0:NUM_TEST_LAYERS-1] = '{
        16384, 16384, 4096, 8192, 2048, 4096, 1024, 1024, 512, 512,
        128, 256, 256, 256, 128, 128, 64
    };
    localparam bit PP_BUF_SEL [0:NUM_TEST_LAYERS-1] = '{
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1
    };
    localparam int WR_OFFSET [0:NUM_TEST_LAYERS-1] = '{
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256, 256, 256, 512, 512, 512
    };

    task automatic load_golden(input int layer_idx);
        string fname;
        $sformat(fname, "%sgolden_layer%0d_uram.mem", MEM_PATH, layer_idx);
        $readmemh(fname, golden_buf);
    endtask

    task automatic verify_uram_at_offset(
        input string buf_name,
        input int    num_words,
        input int    offset
    );
        integer errors = 0;
        logic [127:0] actual;

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
            $display("  [PASS] %s[%0d..%0d]: all %0d words match",
                     buf_name, offset, offset + num_words - 1, num_words);
        else
            $display("  [FAIL] %s: %0d / %0d mismatches", buf_name, errors, num_words);
    endtask

    // =========================================================================
    //  Main Test Sequence
    // =========================================================================
    integer start_cycle;
    logic [31:0] status_val;
    logic [31:0] read_val;

    initial begin
        $display("=========================================");
        $display(" tb_tinyissimoyolo_accel — AXI integration");
        $display("  Mode:  FIFO pixel write (AXI-Lite)");
        $display("  Path:  packed pixels → fmap_b → inference → result readout");
        $display("=========================================");

        // Init AXI signals
        s_axi_lite_awaddr  = '0;
        s_axi_lite_awvalid = 1'b0;
        s_axi_lite_wdata   = '0;
        s_axi_lite_wstrb   = '0;
        s_axi_lite_wvalid  = 1'b0;
        s_axi_lite_bready  = 1'b0;
        s_axi_lite_araddr  = '0;
        s_axi_lite_arvalid = 1'b0;
        s_axi_lite_rready  = 1'b0;

        // Load pixel data
        $display("[INIT] Loading pixel data...");
        $readmemh({MEM_PATH, "pixels_layer0.mem"}, pixel_mem);

        // Reset
        aresetn = 0;
        #100;
        aresetn = 1;
        #20;

        // -----------------------------------------------------------------
        //  Step 1: Set mode = FIFO (0)
        // -----------------------------------------------------------------
        $display("[STEP 1] Set MODE = 0 (FIFO)");
        axil_write(ADDR_MODE, 32'h0);

        // -----------------------------------------------------------------
        //  Step 2: Start preload (CTRL[0]=start, CTRL[1]=fifo_rst)
        // -----------------------------------------------------------------
        $display("[STEP 2] Write CTRL = 0x03 (start + fifo_rst)");
        axil_write(ADDR_CTRL, 32'h03);

        // -----------------------------------------------------------------
        //  Step 3: Write 65536 packed pixels to FIFO
        // -----------------------------------------------------------------
        $display("[STEP 3] Writing 65536 packed pixels to FIFO...");
        start_cycle = cycle_count;
        for (int n = 0; n < ACT_SIZE * ACT_SIZE; n++) begin
            // Pack: {pad=0, ch2=B, ch1=G, ch0=R} as int8
            axil_write(ADDR_PIXEL_FIFO, {
                8'd0,
                pixel_mem[2][n],
                pixel_mem[1][n],
                pixel_mem[0][n]
            });
        end
        $display("[STEP 3] FIFO write complete: %0d cycles, pixel_count=%0d",
                 cycle_count - start_cycle, 0);

        // Verify pixel count
        axil_read(ADDR_PIXEL_CNT, read_val);
        $display("  PIXEL_CNT = %0d (expected 65536)", read_val);

        // -----------------------------------------------------------------
        //  Step 4: Wait for inference to complete
        // -----------------------------------------------------------------
        $display("[STEP 4] Waiting for inference (done)...");
        start_cycle = cycle_count;
        status_val = 0;
        while (!(status_val & 32'h2)) begin  // bit 1 = done
            axil_read(ADDR_STATUS, status_val);
        end
        $display("[STEP 4] Inference done: STATUS=0x%08h", status_val);

        // Read cycle count
        axil_read(ADDR_CYCLE_CNT, read_val);
        $display("  CYCLE_COUNT = %0d (~%.1f ms at 100 MHz)",
                 read_val, real'(read_val) / 100_000.0);

        // -----------------------------------------------------------------
        //  Step 5: Verify intermediate layer outputs (hierarchical access)
        // -----------------------------------------------------------------
        $display("-----------------------------------------");
        $display("[STEP 5] Verifying all %0d layers...", NUM_TEST_LAYERS);
        for (int li = 0; li < NUM_TEST_LAYERS; li++) begin
            load_golden(li);
            if (PP_BUF_SEL[li] == 0)
                verify_uram_at_offset("fmap_a", URAM_WORDS[li], WR_OFFSET[li]);
            else
                verify_uram_at_offset("fmap_b", URAM_WORDS[li], WR_OFFSET[li]);
        end

        // -----------------------------------------------------------------
        //  Step 6: Read results via AXI-Lite and compare
        // -----------------------------------------------------------------
        $display("-----------------------------------------");
        $display("[STEP 6] Reading detection results via AXI-Lite...");
        begin
            integer result_errors = 0;
            logic [127:0] axil_word, golden_word;

            // Load cv2 golden (layer 13)
            load_golden(13);
            $display("  Comparing cv2 (256 words)...");
            for (int w = 0; w < 256; w++) begin
                // Read 4 × 32-bit = one 128-bit URAM word
                for (int lane = 0; lane < 4; lane++) begin
                    axil_read(ADDR_RESULT_BASE + w * 16 + lane * 4, read_val);
                    axil_word[lane * 32 +: 32] = read_val;
                end
                if (axil_word !== golden_buf[w]) begin
                    if (result_errors < 5)
                        $display("    [FAIL] result[%0d] Exp: 0x%032h Got: 0x%032h",
                                 w, golden_buf[w], axil_word);
                    result_errors++;
                end
            end

            // Load cv3 golden (layer 16)
            load_golden(16);
            $display("  Comparing cv3 (64 words)...");
            for (int w = 0; w < 64; w++) begin
                for (int lane = 0; lane < 4; lane++) begin
                    axil_read(ADDR_RESULT_BASE + (256 + w) * 16 + lane * 4, read_val);
                    axil_word[lane * 32 +: 32] = read_val;
                end
                if (axil_word !== golden_buf[w]) begin
                    if (result_errors < 10)
                        $display("    [FAIL] result[%0d] Exp: 0x%032h Got: 0x%032h",
                                 256 + w, golden_buf[w], axil_word);
                    result_errors++;
                end
            end

            if (result_errors == 0)
                $display("  [PASS] AXI-Lite result readout: all 320 words match");
            else
                $display("  [FAIL] AXI-Lite result readout: %0d mismatches", result_errors);
        end

        // -----------------------------------------------------------------
        //  Summary
        // -----------------------------------------------------------------
        $display("=========================================");
        $display(" AXI Integration Test Complete");
        $display("=========================================");

        #100;
        $finish;
    end

    // Timeout watchdog
    initial begin
        #600_000_000;   // 600ms (extra time for AXI overhead)
        $display("TIMEOUT: simulation exceeded 600ms");
        $finish;
    end

endmodule
