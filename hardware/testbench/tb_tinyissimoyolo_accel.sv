`timescale 1ns / 1ps

/*
 * tb_tinyissimoyolo_accel — AXI integration testbench
 *
 * Tests the TB_MODE=0 path (AXI IP mode):
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

    // Testbench ports (unused in TB_MODE=0, but must be connected)
    logic                             tb_done;
    logic [DEPTH_BITS-1:0]            tb_pixel_bram_addr;
    logic                             tb_pixel_bram_en;
    logic [MAX_PARALLEL*N_BITS-1:0]   tb_pixel_bram_data;
    logic                             irq_done;

    assign tb_pixel_bram_data = '0;

    // =========================================================================
    //  DUT (TB_MODE=0: AXI IP mode)
    // =========================================================================
    inference_top #(
        .MAX_PARALLEL   (MAX_PARALLEL),
        .N_BITS         (N_BITS),
        .DEPTH_BITS     (DEPTH_BITS),
        .TB_MODE (0),
        .AXI_ADDR_W     (AXI_ADDR_W)
    ) dut (
        .aclk                (clk),
        .aresetn             (aresetn),
        .start               (1'b0),
        .done                (tb_done),
        .pixel_bram_addr     (tb_pixel_bram_addr),
        .pixel_bram_en       (tb_pixel_bram_en),
        .pixel_bram_data     (tb_pixel_bram_data),
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
    //  Cycle Monitor (sim-only observational instrumentation)
    //
    //  Plain module instance — earlier bind attempts hit xsim's port
    //  expression resolution quirk (VRFC 10-9543: phase/cycle_count saw
    //  the implicit 1-bit testbench wires, not the DUT's signals).  Using
    //  fully-qualified `dut.…` paths through the regular port map sidesteps
    //  that completely.
    // =========================================================================
    cycle_monitor u_cycle_mon (
        .clk             (clk),
        .rstn            (aresetn),
        .phase           (dut.gen_axi_integration.phase),
        .cycle_count     (dut.gen_axi_integration.cycle_count),
        .hdl_state       (dut.u_inference_hdl.state),
        .layer_idx       (dut.u_inference_hdl.layer_idx),
        .ch_out          (dut.u_inference_hdl.ch_out),
        .act_out_valid   (dut.u_inference_hdl.act_out_valid),
        .pool_out_valid  (dut.u_inference_hdl.pool_out_valid),
        .rmw_s0_valid    (dut.u_inference_hdl.rmw_s0_valid),
        .out_buf_wr_en   (dut.u_inference_hdl.out_buf_wr_en)
    );

    // =========================================================================
    //  Register addresses (must match axil_regs.sv)
    // =========================================================================
    localparam [AXI_ADDR_W-1:0] ADDR_CTRL          = 13'h000,
                                ADDR_STATUS        = 13'h004,
                                ADDR_MODE          = 13'h008,
                                ADDR_CYCLE_CNT     = 13'h00C,
                                ADDR_LAYER_IDX     = 13'h010,
                                ADDR_RESULT_BASE_R = 13'h014,
                                ADDR_RESULT_BUF_R  = 13'h018,
                                ADDR_MAX_LAYERS_R  = 13'h01C,
                                ADDR_PIXEL_FIFO    = 13'h020,
                                ADDR_PIXEL_CNT     = 13'h024,
                                ADDR_RESULT_BASE   = 13'h100;

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
        // Use case statement with compile-time string concatenation;
        // xsim $readmemh does not support runtime-constructed filenames.
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

    // -------------------------------------------------------------------------
    // Verify a layer's URAM via AXI-Lite reads through the result region,
    // exactly as the PYNQ driver does via drv.read_window().
    //
    // Slides the result window (ADDR_RESULT_BASE_R / ADDR_RESULT_BUF_R) over
    // the chunk at [offset .. offset + num_words - 1] in 320-word chunks,
    // reading 4 × 32-bit lanes per URAM word via axil_read, and compares
    // against golden_buf[] (which must be loaded by load_golden(layer) first).
    //
    // This closes the coverage gap that the hierarchical verify_uram_at_offset
    // leaves open: all 4 sim flavors verify the URAM storage directly, but
    // neither that nor the existing cv2/cv3 AXI-Lite readout at base 256/512
    // ever exercises the `result_base_addr = N` arithmetic for N ∈ [0, 15].
    // -------------------------------------------------------------------------
    task automatic verify_axil_at_offset(
        input int buf_sel,        // 0 = fmap_a, 1 = fmap_b
        input int num_words,
        input int offset,
        input int max_to_check    // cap on words to check for sim runtime
    );
        integer errors = 0;
        integer checked = 0;
        logic [127:0] actual;
        logic [31:0]  axi_lane;
        int           words_this_check;

        words_this_check = (max_to_check < num_words) ? max_to_check : num_words;

        $display("  [AXIL] Reading %s[%0d..%0d] via result region (buf=%0d)...",
                 (buf_sel == 0) ? "fmap_a" : "fmap_b",
                 offset, offset + words_this_check - 1, buf_sel);

        // Select the buffer (0=fmap_a, 1=fmap_b)
        axil_write(ADDR_RESULT_BUF_R, buf_sel);

        for (int w = 0; w < words_this_check; w++) begin
            // Slide the window: each read window chunk holds 320 URAM words.
            // Update result_base every 320 words so (ADDR_RESULT_BASE +
            // lane*4) within the chunk stays within the 0x100..0x14FF region.
            if (w % 320 == 0) begin
                axil_write(ADDR_RESULT_BASE_R, offset + w);
            end

            // Read 4 × 32-bit lanes to assemble one 128-bit URAM word.
            for (int lane = 0; lane < 4; lane++) begin
                axil_read(ADDR_RESULT_BASE + (w % 320) * 16 + lane * 4, axi_lane);
                actual[lane * 32 +: 32] = axi_lane;
            end

            if (actual !== golden_buf[w]) begin
                if (errors < 5)
                    $display("    [AXIL FAIL] word[%0d] Exp: 0x%032h Got: 0x%032h",
                             w, golden_buf[w], actual);
                errors++;
            end
            checked++;
        end

        if (errors == 0)
            $display("  [AXIL PASS] %0d / %0d words match via AXI-Lite",
                     checked, checked);
        else
            $display("  [AXIL FAIL] %0d / %0d mismatches via AXI-Lite",
                     errors, checked);
    endtask

    // -------------------------------------------------------------------------
    // Cascade bisection helper: snapshot layer N's output at the first moment
    // it is safe to read, i.e. immediately after the FSM advances past layer N
    // but before the ping-pong partner that shares the same (buffer, offset)
    // slot has started writing.
    //
    // Uses `>` (not `==`) so the wait never hangs if we arrive slightly late
    // after some other blocking step consumed sim time — at worst we read the
    // buffer after the safe window has closed and the verify fails loudly.
    //
    // All reads inside `verify_uram_at_offset` are 0 sim-time procedural URAM
    // hierarchical accesses, so the snapshot itself cannot race the FSM.
    // -------------------------------------------------------------------------
    task automatic snapshot_layer(input int layer_to_check);
        wait(dut.u_inference_hdl.layer_idx > layer_to_check);
        @(posedge clk);
        $display("  [BISECT L%0d] layer %0d boundary (cycle=%0d, layer_idx=%0d)",
                 layer_to_check, layer_to_check, cycle_count,
                 dut.u_inference_hdl.layer_idx);
        load_golden(layer_to_check);
        if (PP_BUF_SEL[layer_to_check] == 0)
            verify_uram_at_offset("fmap_a",
                                  URAM_WORDS[layer_to_check],
                                  WR_OFFSET[layer_to_check]);
        else
            verify_uram_at_offset("fmap_b",
                                  URAM_WORDS[layer_to_check],
                                  WR_OFFSET[layer_to_check]);
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
        //  Step 3.5: Verify layer 0 output via hierarchical fmap_a snapshot
        //
        //  Closes the coverage gap that hid the +1 URAM-word silicon shift:
        //  the original Step 5 verifies only "surviving" layers (10/12/13/15/16)
        //  because by the time inference is fully done, layer 0's output in
        //  fmap_a[0..16383] has been overwritten by layers 2/4/6/8.
        //
        //  Insertion point: wait for the FSM to advance into layer 1 (i.e.
        //  layer_idx==1). At that moment layer 0 has committed its full 16384
        //  words to fmap_a, and layer 1 writes to fmap_b (PP_BUF_SEL[1]=1) so
        //  fmap_a remains undisturbed until layer 2 begins. We snapshot here.
        // -----------------------------------------------------------------
        $display("[STEP 3.5] Waiting for FSM to advance past layer 0...");
        wait(dut.u_inference_hdl.layer_idx == 1);
        @(posedge clk);
        $display("[STEP 3.5] Layer 0 complete (cycle=%0d). Verifying fmap_a[0..16383] vs golden_layer0...",
                 cycle_count);
        load_golden(0);
        if (PP_BUF_SEL[0] == 0)
            verify_uram_at_offset("fmap_a", URAM_WORDS[0], WR_OFFSET[0]);
        else
            verify_uram_at_offset("fmap_b", URAM_WORDS[0], WR_OFFSET[0]);

        // -----------------------------------------------------------------
        //  Step 3.6: Verify layer 0 via AXI-Lite reads at result_base=0
        //
        //  The hierarchical check above reads dut.u_fmap_a.ram[] directly,
        //  which is the same URAM storage cells the AXI-Lite read region
        //  targets. But this does NOT exercise the axil_regs read-address
        //  decode path with result_base_addr=0 — the cv2/cv3 Step 6 check
        //  only ever uses base 256 / 512.
        //
        //  If PYNQ's drv.read_window(0, buf=0, 16384) returns a +1-shifted
        //  view of layer 0 while the hierarchical check passes, the bug
        //  is in the axil_regs result-region read path (not in the compute
        //  pipeline). This test reproduces that case in sim.
        //
        //  Limit to the first 32 URAM words to keep xsim runtime reasonable
        //  (each axil_read burns ~5-10 cycles).
        // -----------------------------------------------------------------
        // -----------------------------------------------------------------
        //  Step 3.6 was deliberately REMOVED.
        //
        //  The prior revision of this file ran AXI-Lite reads of
        //  `verify_axil_at_offset(PP_BUF_SEL[0], 16384, 0, 640)` here,
        //  AFTER Step 3.5 and BEFORE Step 3.7. That is unsafe: Step 3.5
        //  returns at the layer-0 → layer-1 boundary but layer 1 is
        //  still actively streaming pixel reads from fmap_a port B. In
        //  inference_top.sv:524-529 the fmap_a read-port mux gives
        //  `result_rd_a` priority over `input_rd_en`, so every AXI-Lite
        //  read pulsed by axil_regs hijacks the URAM read port away
        //  from conv3d for one cycle. The period-5 corruption that
        //  resulted looked *exactly* like an RTL compute bug at layer 1
        //  ch_out=0 byte[0] — but the RTL is fine; the testbench was
        //  poisoning the conv input stream.
        //
        //  Base=0 coverage of the axil_regs result-region read decode
        //  path has been moved to Step 6a below, which runs after
        //  Step 4 (`done`) when no conv reads are in flight. At that
        //  point fmap_a[0..127] holds layer 10's output, so we verify
        //  the base=0 path against `golden_layer10_uram.mem` instead of
        //  layer 0 (whose fmap_a contents have been overwritten).
        // -----------------------------------------------------------------

        // -----------------------------------------------------------------
        //  Step 3.7: Per-layer bisection snapshot
        //
        //  Layer 0 passes bit-exact (Steps 3.5 & 3.6). Layers 10/12/13/15/16
        //  fail at Step 5. Walk the cascade and snapshot every intermediate
        //  layer at its first safe window (after layer N completes, before
        //  its ping-pong partner at N+2 overwrites the slot).
        //
        //  Each snapshot is 0 sim time — the wait() advances the clock to
        //  the boundary, the URAM hierarchical reads are procedural. The
        //  total extra sim cost is the time we'd spend waiting for `done`
        //  anyway.
        //
        //  This is a diagnostic block: on a healthy pipeline every layer
        //  reports PASS. The first FAIL localizes where the HDL diverges
        //  from `generate_conv3d_golden.py`.
        // -----------------------------------------------------------------
        $display("-----------------------------------------");
        $display("[STEP 3.7] Cascade bisection — snapshot every layer at its boundary...");
        // Upper bound is NUM_TEST_LAYERS-1 because snapshot_layer waits for
        // `layer_idx > layer_to_check`, and no layer_idx > 16 ever exists
        // (the cascade ends at layer 16). Layer 16's final state is
        // verified separately via Step 5's survivor scan.
        for (int li = 1; li < NUM_TEST_LAYERS - 1; li++) begin
            snapshot_layer(li);
        end

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
        //  Step 5: Verify surviving layer outputs (hierarchical access)
        //
        //  Ping-pong buffers are reused across layers, so only the last
        //  writer at each (buffer, offset) survives after full inference:
        //    fmap_a[0:127]   = layer 10   fmap_b[256:511] = layer 13
        //    fmap_a[256:511] = layer 12   fmap_b[512:575] = layer 16
        //    fmap_a[512:639] = layer 15
        //  Layers 0-9, 11, 14 are overwritten and cannot be verified.
        // -----------------------------------------------------------------
        $display("-----------------------------------------");
        $display("[STEP 5] Verifying surviving layers (10,12,13,15,16)...");
        for (int li = 0; li < NUM_TEST_LAYERS; li++) begin
            // Skip layers whose URAM contents have been overwritten
            if (li < 10 || li == 11 || li == 14) continue;
            load_golden(li);
            if (PP_BUF_SEL[li] == 0)
                verify_uram_at_offset("fmap_a", URAM_WORDS[li], WR_OFFSET[li]);
            else
                verify_uram_at_offset("fmap_b", URAM_WORDS[li], WR_OFFSET[li]);
        end

        // -----------------------------------------------------------------
        //  Step 5a: AXI-Lite read of layer 10 via result_base = 0
        //
        //  This covers the axil_regs result-region address-decode path
        //  for result_base_addr = 0, which the Step 6 cv2/cv3 checks
        //  never exercise (cv2 uses base 256, cv3 uses base 512). It
        //  runs AFTER inference completes so it cannot race the
        //  conv3d pixel stream on fmap_a's read port. Layer 10's
        //  output survives at fmap_a[0..127].
        //
        //  Restores base=256/buf=1 before Step 6 so that the cv2/cv3
        //  readout continues to use the reset defaults that the
        //  on-board driver also relies on.
        // -----------------------------------------------------------------
        $display("-----------------------------------------");
        $display("[STEP 5a] AXI-Lite read of layer 10 via result_base=0...");
        load_golden(10);
        verify_axil_at_offset(
            .buf_sel     (PP_BUF_SEL[10]),
            .num_words   (URAM_WORDS[10]),
            .offset      (WR_OFFSET[10]),
            .max_to_check(URAM_WORDS[10])
        );
        axil_write(ADDR_RESULT_BASE_R, 32'd256);
        axil_write(ADDR_RESULT_BUF_R,  32'd1);

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
