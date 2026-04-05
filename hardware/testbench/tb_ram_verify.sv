`timescale 1ns / 1ps
`include "layer_config.svh"

module tb_ram_verify;

    localparam MEM_PATH = "../../../../../../weights/hdl/";

    logic clk = 0;
    always #5 clk = ~clk; // 100 MHz

    /* ================================================================
     *  Weight Memory — URAM, 128-bit x 32768
     * ================================================================ */
    localparam WT_DATA_W = C_PAR * 8;          // 128
    localparam WT_DEPTH  = 32768;
    localparam WT_VALID  = WEIGHT_ROM_DEPTH;   // 25654
    localparam WT_ADDR_W = $clog2(WT_DEPTH);

    logic                    wt_en_b;
    logic [WT_ADDR_W-1:0]   wt_addr_b;
    logic [WT_DATA_W-1:0]   wt_dout_b;

    sdp_ram #(
        .DATA_WIDTH (WT_DATA_W),
        .DEPTH      (WT_DEPTH),
        .RAM_STYLE  ("ultra"),
        .MEM_FILE   ({MEM_PATH, "weight_rom.mem"})
    ) u_wt_mem (
        .clk    (clk),
        .en_a   (1'b0),
        .en_b   (wt_en_b),
        .we_a   (1'b0),
        .addr_a ('0),
        .addr_b (wt_addr_b),
        .din_a  ('0),
        .dout_b (wt_dout_b)
    );

    reg [WT_DATA_W-1:0] wt_ref [0:WT_DEPTH-1];
    initial $readmemh({MEM_PATH, "weight_rom.mem"}, wt_ref);

    /* ================================================================
     *  QP Packed Memory — BRAM, 72-bit x 1024
     * ================================================================ */
    localparam QP_DATA_W = 72;
    localparam QP_DEPTH  = 1024;
    localparam QP_VALID  = QP_PACKED_ROM_DEPTH; // 827
    localparam QP_ADDR_W = $clog2(QP_DEPTH);

    logic                    qp_en_b;
    logic [QP_ADDR_W-1:0]   qp_addr_b;
    logic [QP_DATA_W-1:0]   qp_dout_b;

    sdp_ram #(
        .DATA_WIDTH (QP_DATA_W),
        .DEPTH      (QP_DEPTH),
        .RAM_STYLE  ("block"),
        .MEM_FILE   ({MEM_PATH, "qp_packed_rom.mem"})
    ) u_qp_mem (
        .clk    (clk),
        .en_a   (1'b0),
        .en_b   (qp_en_b),
        .we_a   (1'b0),
        .addr_a ('0),
        .addr_b (qp_addr_b),
        .din_a  ('0),
        .dout_b (qp_dout_b)
    );

    reg [QP_DATA_W-1:0] qp_ref [0:QP_DEPTH-1];
    initial $readmemh({MEM_PATH, "qp_packed_rom.mem"}, qp_ref);

    /* ================================================================
     *  Sigmoid Memory — Distributed RAM, 8-bit x 4352
     * ================================================================ */
    localparam SIG_DATA_W = 8;
    localparam SIG_DEPTH  = SIGMOID_LUT_DEPTH; // 4352
    localparam SIG_VALID  = SIGMOID_LUT_DEPTH;
    localparam SIG_ADDR_W = $clog2(SIG_DEPTH);

    logic                    sig_en_b;
    logic [SIG_ADDR_W-1:0]  sig_addr_b;
    logic [SIG_DATA_W-1:0]  sig_dout_b;

    sdp_ram #(
        .DATA_WIDTH (SIG_DATA_W),
        .DEPTH      (SIG_DEPTH),
        .RAM_STYLE  ("distributed"),
        .MEM_FILE   ({MEM_PATH, "sigmoid_lut.mem"})
    ) u_sig_mem (
        .clk    (clk),
        .en_a   (1'b0),
        .en_b   (sig_en_b),
        .we_a   (1'b0),
        .addr_a ('0),
        .addr_b (sig_addr_b),
        .din_a  ('0),
        .dout_b (sig_dout_b)
    );

    reg [SIG_DATA_W-1:0] sig_ref [0:SIG_DEPTH-1];
    initial $readmemh({MEM_PATH, "sigmoid_lut.mem"}, sig_ref);

    /* ================================================================
     *  Verification
     * ================================================================ */
    integer errors, first_err;
    integer total_pass, total_fail;

    initial begin
        $display("========================================");
        $display("  RAM Content Verification");
        $display("========================================");

        total_pass = 0;
        total_fail = 0;
        wt_en_b  = 0;
        qp_en_b  = 0;
        sig_en_b = 0;
        #100; // let $readmemh initialisation settle

        // Sanity check: verify files were actually loaded (not all-X)
        if ($isunknown(wt_ref[0])) begin
            $display("[FATAL] wt_mem : reference array is X — .mem file not loaded");
            $display("        Check MEM_PATH: %s", MEM_PATH);
            $finish;
        end
        if ($isunknown(qp_ref[0])) begin
            $display("[FATAL] qp_mem : reference array is X — .mem file not loaded");
            $finish;
        end
        if ($isunknown(sig_ref[0])) begin
            $display("[FATAL] sig_mem: reference array is X — .mem file not loaded");
            $finish;
        end
        $display("  All .mem files loaded successfully");

        /* ---- Weight Memory ---- */
        errors = 0;
        first_err = -1;
        for (int i = 0; i <= WT_VALID; i++) begin
            @(posedge clk);
            if (i < WT_VALID) begin
                wt_en_b   = 1'b1;
                wt_addr_b = i[WT_ADDR_W-1:0];
            end else begin
                wt_en_b   = 1'b0;
            end
            if (i > 0) begin
                if (wt_dout_b !== wt_ref[i-1]) begin
                    errors++;
                    if (first_err == -1) first_err = i - 1;
                    if (errors <= 5)
                        $display("  MISMATCH wt_mem[%0d]: got %h, exp %h",
                                 i-1, wt_dout_b, wt_ref[i-1]);
                end
            end
        end
        if (errors == 0) begin
            $display("[PASS] wt_mem : %0d/%0d entries verified", WT_VALID, WT_VALID);
            total_pass++;
        end else begin
            $display("[FAIL] wt_mem : %0d mismatches (first @ 0x%0h)", errors, first_err);
            total_fail++;
        end

        /* ---- QP Packed Memory ---- */
        errors = 0;
        first_err = -1;
        for (int i = 0; i <= QP_VALID; i++) begin
            @(posedge clk);
            if (i < QP_VALID) begin
                qp_en_b   = 1'b1;
                qp_addr_b = i[QP_ADDR_W-1:0];
            end else begin
                qp_en_b   = 1'b0;
            end
            if (i > 0) begin
                if (qp_dout_b !== qp_ref[i-1]) begin
                    errors++;
                    if (first_err == -1) first_err = i - 1;
                    if (errors <= 5)
                        $display("  MISMATCH qp_mem[%0d]: got %h, exp %h",
                                 i-1, qp_dout_b, qp_ref[i-1]);
                end
            end
        end
        if (errors == 0) begin
            $display("[PASS] qp_mem : %0d/%0d entries verified", QP_VALID, QP_VALID);
            total_pass++;
        end else begin
            $display("[FAIL] qp_mem : %0d mismatches (first @ 0x%0h)", errors, first_err);
            total_fail++;
        end

        /* ---- Sigmoid Memory ---- */
        errors = 0;
        first_err = -1;
        for (int i = 0; i <= SIG_VALID; i++) begin
            @(posedge clk);
            if (i < SIG_VALID) begin
                sig_en_b   = 1'b1;
                sig_addr_b = i[SIG_ADDR_W-1:0];
            end else begin
                sig_en_b   = 1'b0;
            end
            if (i > 0) begin
                if (sig_dout_b !== sig_ref[i-1]) begin
                    errors++;
                    if (first_err == -1) first_err = i - 1;
                    if (errors <= 5)
                        $display("  MISMATCH sig_mem[%0d]: got %h, exp %h",
                                 i-1, sig_dout_b, sig_ref[i-1]);
                end
            end
        end
        if (errors == 0) begin
            $display("[PASS] sig_mem: %0d/%0d entries verified", SIG_VALID, SIG_VALID);
            total_pass++;
        end else begin
            $display("[FAIL] sig_mem: %0d mismatches (first @ 0x%0h)", errors, first_err);
            total_fail++;
        end

        /* ---- Summary ---- */
        $display("========================================");
        $display("  %0d PASSED, %0d FAILED", total_pass, total_fail);
        if (total_fail == 0)
            $display("  ALL RAM CONTENTS VERIFIED");
        else
            $display("  VERIFICATION FAILED");
        $display("========================================");
        $finish;
    end

endmodule
