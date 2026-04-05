`include "layer_config.svh"

module top #(
    parameter AXI_DATA_WIDTH = 24 // RGB8
)(
    input logic aclk,
    input logic aresetn,
    /* Slave (Receive) */
    input logic s_axis_tvalid,
    input logic s_axis_tlast,
    input logic [AXI_DATA_WIDTH-1:0] s_axis_tdata,
    output logic s_axis_tready,
    /* Master (Transmit) */
    input logic m_axis_tready,
    output logic m_axis_tvalid,
    output logic m_axis_tlast,
    output logic [AXI_DATA_WIDTH-1:0] m_axis_tdata
);

    /* ================================================================
     *  Memory Subsystem Parameters
     * ================================================================ */
    localparam WT_MEM_DATA_W  = C_PAR * 8;             // 128
    localparam WT_MEM_DEPTH   = 32768;                  // next pow2 >= WEIGHT_ROM_DEPTH
    localparam WT_MEM_ADDR_W  = $clog2(WT_MEM_DEPTH);  // 15

    localparam QP_MEM_DATA_W  = 72;                     // packed {nshift, m0, bias}
    localparam QP_MEM_DEPTH   = 1024;                   // next pow2 >= QP_PACKED_ROM_DEPTH
    localparam QP_MEM_ADDR_W  = $clog2(QP_MEM_DEPTH);  // 10

    localparam SIG_MEM_DATA_W = 8;
    localparam SIG_MEM_DEPTH  = SIGMOID_LUT_DEPTH;      // 4352
    localparam SIG_MEM_ADDR_W = $clog2(SIG_MEM_DEPTH);  // 13

    localparam FMAP_DATA_W    = C_PAR * 8;              // 128
    localparam FMAP_DEPTH     = 16384;
    localparam FMAP_ADDR_W    = $clog2(FMAP_DEPTH);     // 14

    /* ================================================================
     *  Memory Signals
     * ================================================================ */

    /* Weight Memory */
    logic                       wt_mem_en_a,  wt_mem_we_a;
    logic [WT_MEM_ADDR_W-1:0]  wt_mem_addr_a;
    logic [WT_MEM_DATA_W-1:0]  wt_mem_din_a;
    logic                       wt_mem_en_b;
    logic [WT_MEM_ADDR_W-1:0]  wt_mem_addr_b;
    logic [WT_MEM_DATA_W-1:0]  wt_mem_dout_b;

    /* QP Packed Memory */
    logic                       qp_mem_en_a,  qp_mem_we_a;
    logic [QP_MEM_ADDR_W-1:0]  qp_mem_addr_a;
    logic [QP_MEM_DATA_W-1:0]  qp_mem_din_a;
    logic                       qp_mem_en_b;
    logic [QP_MEM_ADDR_W-1:0]  qp_mem_addr_b;
    logic [QP_MEM_DATA_W-1:0]  qp_mem_dout_b;

    /* Sigmoid Memory */
    logic                        sig_mem_en_a,  sig_mem_we_a;
    logic [SIG_MEM_ADDR_W-1:0]  sig_mem_addr_a;
    logic [SIG_MEM_DATA_W-1:0]  sig_mem_din_a;
    logic                        sig_mem_en_b;
    logic [SIG_MEM_ADDR_W-1:0]  sig_mem_addr_b;
    logic [SIG_MEM_DATA_W-1:0]  sig_mem_dout_b;

    /* Feature Map Buffer A */
    logic                      fmap_a_en_a,  fmap_a_we_a;
    logic [FMAP_ADDR_W-1:0]   fmap_a_addr_a;
    logic [FMAP_DATA_W-1:0]   fmap_a_din_a;
    logic                      fmap_a_en_b;
    logic [FMAP_ADDR_W-1:0]   fmap_a_addr_b;
    logic [FMAP_DATA_W-1:0]   fmap_a_dout_b;

    /* Feature Map Buffer B */
    logic                      fmap_b_en_a,  fmap_b_we_a;
    logic [FMAP_ADDR_W-1:0]   fmap_b_addr_a;
    logic [FMAP_DATA_W-1:0]   fmap_b_din_a;
    logic                      fmap_b_en_b;
    logic [FMAP_ADDR_W-1:0]   fmap_b_addr_b;
    logic [FMAP_DATA_W-1:0]   fmap_b_dout_b;

    /* QP Unpacked Fields */
    logic signed [31:0] qp_bias;
    logic        [31:0] qp_m0;
    logic         [5:0] qp_nshift;

    assign qp_bias   = qp_mem_dout_b[31:0];
    assign qp_m0     = qp_mem_dout_b[63:32];
    assign qp_nshift = qp_mem_dout_b[69:64];

    /* ================================================================
     *  Memory Instances
     * ================================================================ */

    /* Weight Memory — URAM, 128-bit x 32768 */
    sdp_ram #(
        .DATA_WIDTH (WT_MEM_DATA_W),
        .DEPTH      (WT_MEM_DEPTH),
        .RAM_STYLE  ("ultra"),
        .MEM_FILE   ("../../../../../../weights/hdl/weight_rom.mem")
    ) u_wt_mem (
        .clk    (aclk),
        .en_a   (wt_mem_en_a),
        .en_b   (wt_mem_en_b),
        .we_a   (wt_mem_we_a),
        .addr_a (wt_mem_addr_a),
        .addr_b (wt_mem_addr_b),
        .din_a  (wt_mem_din_a),
        .dout_b (wt_mem_dout_b)
    );

    /* QP Packed Memory — BRAM, 72-bit x 1024 */
    sdp_ram #(
        .DATA_WIDTH (QP_MEM_DATA_W),
        .DEPTH      (QP_MEM_DEPTH),
        .RAM_STYLE  ("block"),
        .MEM_FILE   ("../../../../../../weights/hdl/qp_packed_rom.mem")
    ) u_qp_mem (
        .clk    (aclk),
        .en_a   (qp_mem_en_a),
        .en_b   (qp_mem_en_b),
        .we_a   (qp_mem_we_a),
        .addr_a (qp_mem_addr_a),
        .addr_b (qp_mem_addr_b),
        .din_a  (qp_mem_din_a),
        .dout_b (qp_mem_dout_b)
    );

    /* Sigmoid LUT Memory — Distributed RAM, 8-bit x 4352 */
    sdp_ram #(
        .DATA_WIDTH (SIG_MEM_DATA_W),
        .DEPTH      (SIG_MEM_DEPTH),
        .RAM_STYLE  ("distributed"),
        .MEM_FILE   ("../../../../../../weights/hdl/sigmoid_lut.mem")
    ) u_sig_mem (
        .clk    (aclk),
        .en_a   (sig_mem_en_a),
        .en_b   (sig_mem_en_b),
        .we_a   (sig_mem_we_a),
        .addr_a (sig_mem_addr_a),
        .addr_b (sig_mem_addr_b),
        .din_a  (sig_mem_din_a),
        .dout_b (sig_mem_dout_b)
    );

    /* Feature Map Buffer A — URAM, 128-bit x 16384 */
    sdp_ram #(
        .DATA_WIDTH (FMAP_DATA_W),
        .DEPTH      (FMAP_DEPTH),
        .RAM_STYLE  ("ultra")
    ) u_fmap_a (
        .clk    (aclk),
        .en_a   (fmap_a_en_a),
        .en_b   (fmap_a_en_b),
        .we_a   (fmap_a_we_a),
        .addr_a (fmap_a_addr_a),
        .addr_b (fmap_a_addr_b),
        .din_a  (fmap_a_din_a),
        .dout_b (fmap_a_dout_b)
    );

    /* Feature Map Buffer B — URAM, 128-bit x 16384 */
    sdp_ram #(
        .DATA_WIDTH (FMAP_DATA_W),
        .DEPTH      (FMAP_DEPTH),
        .RAM_STYLE  ("ultra")
    ) u_fmap_b (
        .clk    (aclk),
        .en_a   (fmap_b_en_a),
        .en_b   (fmap_b_en_b),
        .we_a   (fmap_b_we_a),
        .addr_a (fmap_b_addr_a),
        .addr_b (fmap_b_addr_b),
        .din_a  (fmap_b_din_a),
        .dout_b (fmap_b_dout_b)
    );

    /* ================================================================
     *  Write Port Tie-offs (pre-loaded memories)
     *  Will be replaced with PS write interface for runtime loading
     * ================================================================ */
    assign wt_mem_en_a   = 1'b0;
    assign wt_mem_we_a   = 1'b0;
    assign wt_mem_addr_a = '0;
    assign wt_mem_din_a  = '0;

    assign qp_mem_en_a   = 1'b0;
    assign qp_mem_we_a   = 1'b0;
    assign qp_mem_addr_a = '0;
    assign qp_mem_din_a  = '0;

    assign sig_mem_en_a   = 1'b0;
    assign sig_mem_we_a   = 1'b0;
    assign sig_mem_addr_a = '0;
    assign sig_mem_din_a  = '0;

    /* ================================================================
     *  FSM & Datapath (TODO)
     * ================================================================ */

    /* Next State Logic */
    always_comb begin : next_state_logic
        // TODO
    end

    /* State Memory */
    always_ff @(posedge aclk) begin : state_memory
        // TODO
    end

    /* Counters */
    always_ff @(posedge aclk) begin : counters
        // TODO
    end

    /* Datapath — default assignments prevent latches */
    always_comb begin : datapath
        // TODO: replace with FSM/compute pipeline logic
        wt_mem_en_b    = 1'b0;
        wt_mem_addr_b  = '0;
        qp_mem_en_b    = 1'b0;
        qp_mem_addr_b  = '0;
        sig_mem_en_b   = 1'b0;
        sig_mem_addr_b = '0;
        fmap_a_en_a    = 1'b0;
        fmap_a_we_a    = 1'b0;
        fmap_a_addr_a  = '0;
        fmap_a_din_a   = '0;
        fmap_a_en_b    = 1'b0;
        fmap_a_addr_b  = '0;
        fmap_b_en_a    = 1'b0;
        fmap_b_we_a    = 1'b0;
        fmap_b_addr_a  = '0;
        fmap_b_din_a   = '0;
        fmap_b_en_b    = 1'b0;
        fmap_b_addr_b  = '0;
    end

    /* Output Registers */
    always_ff @(posedge aclk) begin
        // TODO
    end

    /* Output Logic */
    always_comb begin : output_logic
        // TODO
    end

endmodule
