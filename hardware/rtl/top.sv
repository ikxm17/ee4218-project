`timescale 1ns / 1ps
`include "layer_config.svh"

module top #(
    parameter AXI_DATA_WIDTH = 24, // RGB8
    parameter MAX_PARALLEL   = C_PAR,
    parameter N_BITS         = 8,
    parameter DEPTH_BITS     = 16
)(
    input  logic aclk,
    input  logic aresetn,

    /* Inference control */
    input  logic start,
    output logic done,

    /* Slave AXI-Stream (Receive) */
    input  logic s_axis_tvalid,
    input  logic s_axis_tlast,
    input  logic [AXI_DATA_WIDTH-1:0] s_axis_tdata,
    output logic s_axis_tready,

    /* Master AXI-Stream (Transmit) */
    input  logic m_axis_tready,
    output logic m_axis_tvalid,
    output logic m_axis_tlast,
    output logic [AXI_DATA_WIDTH-1:0] m_axis_tdata,

    /* Pixel BRAM interface (testbench drives) */
    output logic [DEPTH_BITS-1:0]            pixel_bram_addr,
    output logic                             pixel_bram_en,
    input  logic [MAX_PARALLEL*N_BITS-1:0]   pixel_bram_data,

    /* RES output interface (testbench captures) */
    output logic                             res_write_en,
    output logic [DEPTH_BITS-1:0]            res_write_addr,
    output logic signed [N_BITS-1:0]         res_write_data
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

    localparam ACT_MEM_DATA_W = 8;
    localparam ACT_MEM_DEPTH  = ACT_LUT_DEPTH;           // 4352
    localparam ACT_MEM_ADDR_W = $clog2(ACT_MEM_DEPTH);   // 13

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

    /* Activation LUT Memory */
    logic                        act_mem_en_a,  act_mem_we_a;
    logic [ACT_MEM_ADDR_W-1:0]  act_mem_addr_a;
    logic [ACT_MEM_DATA_W-1:0]  act_mem_din_a;
    logic                        act_mem_en_b;
    logic [ACT_MEM_ADDR_W-1:0]  act_mem_addr_b;
    logic [ACT_MEM_DATA_W-1:0]  act_mem_dout_b;

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

    /* QP unpacking is done inside inference_hdl */

    /* ================================================================
     *  Memory Instances
     * ================================================================ */

    /* Weight Memory — URAM, 128-bit x 32768 */
    sdp_ram #(
        .DATA_WIDTH (WT_MEM_DATA_W),
        .DEPTH      (WT_MEM_DEPTH),
        .RAM_STYLE  ("ultra"),
        .MEM_FILE   ("weight_rom.mem")
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
        .MEM_FILE   ("qp_packed_rom.mem")
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

    /* Activation LUT Memory — Distributed RAM, 8-bit x 4352 */
    sdp_ram #(
        .DATA_WIDTH (ACT_MEM_DATA_W),
        .DEPTH      (ACT_MEM_DEPTH),
        .RAM_STYLE  ("distributed"),
        .MEM_FILE   ("silu_lut.mem")
    ) u_silu_mem (
        .clk    (aclk),
        .en_a   (act_mem_en_a),
        .en_b   (act_mem_en_b),
        .we_a   (act_mem_we_a),
        .addr_a (act_mem_addr_a),
        .addr_b (act_mem_addr_b),
        .din_a  (act_mem_din_a),
        .dout_b (act_mem_dout_b)
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

    assign act_mem_en_a   = 1'b0;
    assign act_mem_we_a   = 1'b0;
    assign act_mem_addr_a = '0;
    assign act_mem_din_a  = '0;

    /* ================================================================
     *  URAM Port Routing
     *
     *  Each buffer has port A (write) and port B (read).
     *
     *  Write buffer (output): port A = RMW write, port B = RMW read
     *  Read buffer (input):   port B = conv3d pixel read, port A = unused
     *
     *  buf_sel=0 (even layer): write=A, read=B
     *  buf_sel=1 (odd layer):  write=B, read=A
     * ================================================================ */

    // --- Conv3d input read (read buffer port B) ---
    logic input_rd_en;
    logic [FMAP_ADDR_W-1:0] input_rd_addr;
    assign input_rd_en   = (curr_layer_idx != 0) && pixel_en_int;
    assign input_rd_addr = pixel_addr_int[FMAP_ADDR_W-1:0];

    // --- RMW output write signals (computed in RMW pipeline below) ---
    logic                    rmw_wr_en;
    logic [FMAP_ADDR_W-1:0]  rmw_wr_addr;
    logic [FMAP_DATA_W-1:0]  rmw_wr_data;
    logic                    rmw_rd_en;
    logic [FMAP_ADDR_W-1:0]  rmw_rd_addr;
    logic [FMAP_DATA_W-1:0]  rmw_rd_data;
    assign rmw_rd_data = buf_sel ? fmap_b_dout_b : fmap_a_dout_b;

    // --- fmap_a port allocation ---
    // buf_sel=0: A is write buffer -> port A=RMW write, port B=RMW read
    // buf_sel=1: A is read buffer  -> port A=unused,    port B=conv3d read
    assign fmap_a_en_a   = !buf_sel ? rmw_wr_en     : 1'b0;
    assign fmap_a_we_a   = !buf_sel ? rmw_wr_en     : 1'b0;
    assign fmap_a_addr_a = !buf_sel ? rmw_wr_addr    : '0;
    assign fmap_a_din_a  = !buf_sel ? rmw_wr_data    : '0;
    assign fmap_a_en_b   = !buf_sel ? rmw_rd_en      : input_rd_en;
    assign fmap_a_addr_b = !buf_sel ? rmw_rd_addr    : input_rd_addr;

    // --- fmap_b port allocation ---
    // buf_sel=0: B is read buffer  -> port A=unused,    port B=conv3d read
    // buf_sel=1: B is write buffer -> port A=RMW write, port B=RMW read
    assign fmap_b_en_a   = buf_sel ? rmw_wr_en      : 1'b0;
    assign fmap_b_we_a   = buf_sel ? rmw_wr_en      : 1'b0;
    assign fmap_b_addr_a = buf_sel ? rmw_wr_addr     : '0;
    assign fmap_b_din_a  = buf_sel ? rmw_wr_data     : '0;
    assign fmap_b_en_b   = buf_sel ? rmw_rd_en       : input_rd_en;
    assign fmap_b_addr_b = buf_sel ? rmw_rd_addr     : input_rd_addr;

    /* ================================================================
     *  Inference Controller
     * ================================================================ */
    logic                        conv_res_en;
    logic [DEPTH_BITS-1:0]       conv_res_addr;
    logic signed [N_BITS-1:0]    conv_res_data;
    logic [1:0]                  curr_layer_type;
    logic [4:0]                  curr_layer_idx;
    logic [8:0]                  curr_act_size;
    logic [7:0]                  curr_ch_out;

    /* Internal pixel interface (between inference_hdl and mux) */
    logic [DEPTH_BITS-1:0]            pixel_addr_int;
    logic                             pixel_en_int;
    logic [MAX_PARALLEL*N_BITS-1:0]   pixel_data_int;

    /* Activation → Max Pool intermediate signals */
    logic                        act_out_valid;
    logic [DEPTH_BITS-1:0]       act_out_addr;
    logic signed [N_BITS-1:0]    act_out_data;

    inference_hdl #(
        .MAX_PARALLEL (MAX_PARALLEL),
        .K            (3),
        .N_BITS       (N_BITS),
        .ACC_BITS     (32),
        .DEPTH_BITS   (DEPTH_BITS)
    ) u_inference (
        .aclk             (aclk),
        .aresetn          (aresetn),
        .start            (start),
        .done             (done),
        .wt_mem_en_b      (wt_mem_en_b),
        .wt_mem_addr_b    (wt_mem_addr_b),
        .wt_mem_dout_b    (wt_mem_dout_b),
        .qp_mem_en_b      (qp_mem_en_b),
        .qp_mem_addr_b    (qp_mem_addr_b),
        .qp_mem_dout_b    (qp_mem_dout_b),
        .pixel_bram_addr  (pixel_addr_int),
        .pixel_bram_en    (pixel_en_int),
        .pixel_bram_data  (pixel_data_int),
        .res_write_en     (conv_res_en),
        .res_write_addr   (conv_res_addr),
        .res_write_data   (conv_res_data),
        .curr_layer_type  (curr_layer_type),
        .curr_layer_idx   (curr_layer_idx),
        .curr_act_size    (curr_act_size),
        .curr_ch_out      (curr_ch_out)
    );

    /* ================================================================
     *  Pixel Input Mux & URAM Read Routing
     *
     *  Layer 0: reads from external pixel BRAM (testbench)
     *  Layer 1+: reads from URAM input buffer (ping-pong)
     *
     *  Ping-pong: buf_sel = layer_idx[0]
     *    Even layers -> write fmap_a, read fmap_b
     *    Odd layers  -> write fmap_b, read fmap_a
     * ================================================================ */
    logic buf_sel;
    assign buf_sel = curr_layer_idx[0];

    // Forward address/enable to top-level for layer 0 (testbench)
    assign pixel_bram_addr = pixel_addr_int;
    assign pixel_bram_en   = pixel_en_int;

    // Pixel data mux: layer 0 from external, layer 1+ from URAM
    logic [FMAP_DATA_W-1:0] uram_input_rd_data;
    assign uram_input_rd_data = buf_sel ? fmap_a_dout_b : fmap_b_dout_b;
    assign pixel_data_int = (curr_layer_idx == 0) ? pixel_bram_data : uram_input_rd_data;

    /* ================================================================
     *  Activation Stage (SiLU LUT with CONV1_LIN bypass)
     * ================================================================ */
    activation #(
        .N_BITS     (N_BITS),
        .DEPTH_BITS (DEPTH_BITS),
        .LUT_ADDR_W (ACT_MEM_ADDR_W)
    ) u_activation (
        .clk        (aclk),
        .rst_n      (aresetn),
        .layer_type (curr_layer_type),
        .layer_idx  (curr_layer_idx),
        .in_valid   (conv_res_en),
        .in_addr    (conv_res_addr),
        .in_data    (conv_res_data),
        .lut_en     (act_mem_en_b),
        .lut_addr   (act_mem_addr_b),
        .lut_rdata  (act_mem_dout_b),
        .out_valid  (act_out_valid),
        .out_addr   (act_out_addr),
        .out_data   (act_out_data)
    );

    /* ================================================================
     *  Max Pooling Stage (2x2 stride-2, CONV3_POOL layers only)
     * ================================================================ */
    max_pool #(
        .N_BITS     (N_BITS),
        .DEPTH_BITS (DEPTH_BITS)
    ) u_max_pool (
        .clk        (aclk),
        .rst_n      (aresetn),
        .layer_type (curr_layer_type),
        .act_size   (curr_act_size),
        .in_valid   (act_out_valid),
        .in_addr    (act_out_addr),
        .in_data    (act_out_data),
        .out_valid  (res_write_en),
        .out_addr   (res_write_addr),
        .out_data   (res_write_data)
    );

    /* ================================================================
     *  RMW Output Writer
     *
     *  Pipeline: max_pool output -> read old URAM word (1 cycle) ->
     *            splice byte -> write modified word
     *
     *  URAM address: (ch_out / 16) * H_out * W_out + pixel_addr
     *  Byte position: ch_out[3:0]
     * ================================================================ */

    // Pipeline stage 0: issue URAM read
    logic                     rmw_s0_valid;
    logic [FMAP_ADDR_W-1:0]  rmw_s0_addr;
    logic signed [N_BITS-1:0] rmw_s0_data;
    logic [3:0]               rmw_s0_byte_pos;

    // Compute output spatial size (after pool: act_size/2, else: act_size)
    wire [8:0] h_out = (curr_layer_type == CONV3_POOL) ? (curr_act_size >> 1) : curr_act_size;
    wire [FMAP_ADDR_W-1:0] rmw_base_addr = (curr_ch_out >> 4) * h_out * h_out;

    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            rmw_s0_valid    <= 1'b0;
        end else begin
            rmw_s0_valid    <= res_write_en;
            rmw_s0_addr     <= rmw_base_addr + res_write_addr[FMAP_ADDR_W-1:0];
            rmw_s0_data     <= res_write_data;
            rmw_s0_byte_pos <= curr_ch_out[3:0];
        end
    end

    // Issue read on write buffer port B
    assign rmw_rd_en   = res_write_en;
    assign rmw_rd_addr = rmw_base_addr + res_write_addr[FMAP_ADDR_W-1:0];

    // Pipeline stage 1: splice byte and write
    logic [FMAP_DATA_W-1:0] spliced_word;
    always_comb begin
        spliced_word = rmw_rd_data;
        spliced_word[rmw_s0_byte_pos * 8 +: 8] = rmw_s0_data;
    end

    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            rmw_wr_en <= 1'b0;
        end else begin
            rmw_wr_en   <= rmw_s0_valid;
            rmw_wr_addr <= rmw_s0_addr;
            rmw_wr_data <= spliced_word;
        end
    end

    /* AXI-Stream — not yet connected */
    assign s_axis_tready = 1'b0;
    assign m_axis_tvalid = 1'b0;
    assign m_axis_tlast  = 1'b0;
    assign m_axis_tdata  = '0;

endmodule
