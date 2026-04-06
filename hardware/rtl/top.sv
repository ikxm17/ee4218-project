`timescale 1ns / 1ps
`include "layer_config.svh"

module top #(
    parameter AXI_DATA_WIDTH = 24,
    parameter MAX_PARALLEL   = C_PAR,
    parameter N_BITS         = 8,
    parameter DEPTH_BITS     = 16,
    parameter TB_MODE = 0,  // 0 = AXI IP mode (production), 1 = ext pixel BRAM (testbench)
    parameter AXI_ADDR_W     = 13
)(
    input  logic aclk,
    input  logic aresetn,

    /* --- Direct control (testbench, active when TB_MODE=1) --- */
    input  logic                             start,
    output logic                             done,
    output logic [DEPTH_BITS-1:0]            pixel_bram_addr,
    output logic                             pixel_bram_en,
    input  logic [MAX_PARALLEL*N_BITS-1:0]   pixel_bram_data,
    output logic                             res_write_en,
    output logic [DEPTH_BITS-1:0]            res_write_addr,
    output logic signed [N_BITS-1:0]         res_write_data,

    /* --- AXI4-Lite slave (active when TB_MODE=0) --- */
    input  logic [AXI_ADDR_W-1:0]           s_axi_lite_awaddr,
    input  logic                             s_axi_lite_awvalid,
    output logic                             s_axi_lite_awready,
    input  logic [31:0]                      s_axi_lite_wdata,
    input  logic [3:0]                       s_axi_lite_wstrb,
    input  logic                             s_axi_lite_wvalid,
    output logic                             s_axi_lite_wready,
    output logic [1:0]                       s_axi_lite_bresp,
    output logic                             s_axi_lite_bvalid,
    input  logic                             s_axi_lite_bready,
    input  logic [AXI_ADDR_W-1:0]           s_axi_lite_araddr,
    input  logic                             s_axi_lite_arvalid,
    output logic                             s_axi_lite_arready,
    output logic [31:0]                      s_axi_lite_rdata,
    output logic [1:0]                       s_axi_lite_rresp,
    output logic                             s_axi_lite_rvalid,
    input  logic                             s_axi_lite_rready,

    /* --- AXI4-Stream slave — camera pixels (active when TB_MODE=0) --- */
    input  logic [31:0]                      s_axis_tdata,
    input  logic                             s_axis_tvalid,
    input  logic                             s_axis_tlast,
    input  logic [0:0]                       s_axis_tuser,
    output logic                             s_axis_tready,

    /* --- Interrupt (active when TB_MODE=0) --- */
    output logic                             irq_done
);

    /* ================================================================
     *  Memory Subsystem Parameters
     * ================================================================ */
    localparam WT_MEM_DATA_W  = C_PAR * 8;
    localparam WT_MEM_DEPTH   = 32768;
    localparam WT_MEM_ADDR_W  = $clog2(WT_MEM_DEPTH);

    localparam QP_MEM_DATA_W  = 72;
    localparam QP_MEM_DEPTH   = 1024;
    localparam QP_MEM_ADDR_W  = $clog2(QP_MEM_DEPTH);

    localparam ACT_MEM_DATA_W = 8;
    localparam ACT_MEM_DEPTH  = ACT_LUT_DEPTH;
    localparam ACT_MEM_ADDR_W = $clog2(ACT_MEM_DEPTH);

    localparam FMAP_DATA_W    = C_PAR * 8;
    localparam FMAP_DEPTH     = 16384;
    localparam FMAP_ADDR_W    = $clog2(FMAP_DEPTH);

    /* ================================================================
     *  Memory Signals
     * ================================================================ */
    logic                       wt_mem_en_a,  wt_mem_we_a;
    logic [WT_MEM_ADDR_W-1:0]  wt_mem_addr_a;
    logic [WT_MEM_DATA_W-1:0]  wt_mem_din_a;
    logic                       wt_mem_en_b;
    logic [WT_MEM_ADDR_W-1:0]  wt_mem_addr_b;
    logic [WT_MEM_DATA_W-1:0]  wt_mem_dout_b;

    logic                       qp_mem_en_a,  qp_mem_we_a;
    logic [QP_MEM_ADDR_W-1:0]  qp_mem_addr_a;
    logic [QP_MEM_DATA_W-1:0]  qp_mem_din_a;
    logic                       qp_mem_en_b;
    logic [QP_MEM_ADDR_W-1:0]  qp_mem_addr_b;
    logic [QP_MEM_DATA_W-1:0]  qp_mem_dout_b;

    logic                        act_mem_en_a,  act_mem_we_a;
    logic [ACT_MEM_ADDR_W-1:0]  act_mem_addr_a;
    logic [ACT_MEM_DATA_W-1:0]  act_mem_din_a;
    logic                        act_mem_en_b;
    logic [ACT_MEM_ADDR_W-1:0]  act_mem_addr_b;
    logic [ACT_MEM_DATA_W-1:0]  act_mem_dout_b;

    logic                      fmap_a_en_a,  fmap_a_we_a;
    logic [FMAP_ADDR_W-1:0]   fmap_a_addr_a;
    logic [FMAP_DATA_W-1:0]   fmap_a_din_a;
    logic                      fmap_a_en_b;
    logic [FMAP_ADDR_W-1:0]   fmap_a_addr_b;
    logic [FMAP_DATA_W-1:0]   fmap_a_dout_b;

    logic                      fmap_b_en_a,  fmap_b_we_a;
    logic [FMAP_ADDR_W-1:0]   fmap_b_addr_a;
    logic [FMAP_DATA_W-1:0]   fmap_b_din_a;
    logic                      fmap_b_en_b;
    logic [FMAP_ADDR_W-1:0]   fmap_b_addr_b;
    logic [FMAP_DATA_W-1:0]   fmap_b_dout_b;

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

    /* Write port tie-offs (memories pre-loaded via .mem files) */
    assign {wt_mem_en_a, wt_mem_we_a, wt_mem_addr_a, wt_mem_din_a} = '0;
    assign {qp_mem_en_a, qp_mem_we_a, qp_mem_addr_a, qp_mem_din_a} = '0;
    assign {act_mem_en_a, act_mem_we_a, act_mem_addr_a, act_mem_din_a} = '0;

    /* ================================================================
     *  AXI Integration (TB_MODE=0 only)
     *
     *  Phase FSM:  S_IDLE → S_PRELOAD → S_RUN → S_DONE → S_IDLE
     *  AXI-Lite:   control registers, pixel FIFO, result readout
     *  S_AXIS:     camera pixel accumulator (4×32 → 128-bit)
     * ================================================================ */
    logic                    preload_wr_en;
    logic [FMAP_ADDR_W-1:0] preload_wr_addr;
    logic [FMAP_DATA_W-1:0] preload_wr_data;
    logic                    preload_active;
    logic                    result_rd_en;
    logic [FMAP_ADDR_W-1:0] result_rd_addr;
    logic [FMAP_DATA_W-1:0] result_rd_data;

    /* Inference start/done — muxed between testbench and phase FSM */
    logic inference_start;
    logic inference_done;

    generate if (!TB_MODE) begin : gen_axi_integration
        /* ---- Phase FSM ---- */
        typedef enum logic [2:0] {
            PH_IDLE    = 3'd0,
            PH_PRELOAD = 3'd1,
            PH_RUN     = 3'd2,
            PH_DONE    = 3'd3
        } phase_t;
        phase_t phase, phase_next;

        logic        axil_start;
        logic [1:0]  axil_mode;
        logic        axil_preload_done;
        logic        axis_frame_done;
        logic [31:0] cycle_count;

        /* FIFO preload signals (from axil_regs) */
        logic                    fifo_wr_en;
        logic [FMAP_ADDR_W-1:0] fifo_wr_addr;
        logic [FMAP_DATA_W-1:0] fifo_wr_data;

        /* S_AXIS preload signals */
        logic                    axis_wr_en;
        logic [FMAP_ADDR_W-1:0] axis_wr_addr;
        logic [FMAP_DATA_W-1:0] axis_wr_data;

        // next_state_logic
        always_comb begin
            phase_next = phase;
            case (phase)
                PH_IDLE: begin
                    if (axil_mode[0] == 1'b0 && axil_start)
                        phase_next = PH_PRELOAD;
                    else if (axil_mode[0] == 1'b1)
                        phase_next = PH_PRELOAD;
                end
                PH_PRELOAD: begin
                    if (axil_mode[0] == 1'b0 && axil_preload_done)
                        phase_next = PH_RUN;
                    else if (axil_mode[0] == 1'b1 && axis_frame_done)
                        phase_next = PH_RUN;
                end
                PH_RUN: begin
                    if (inference_done)
                        phase_next = PH_DONE;
                end
                PH_DONE: begin
                    if (axil_start)
                        phase_next = PH_PRELOAD;
                    else if (axil_mode[0] == 1'b1)
                        phase_next = PH_PRELOAD;
                end
                default: phase_next = PH_IDLE;
            endcase
        end

        // state_memory
        always_ff @(posedge aclk or negedge aresetn) begin
            if (!aresetn) phase <= PH_IDLE;
            else          phase <= phase_next;
        end

        assign preload_active = (phase == PH_PRELOAD);
        assign irq_done       = (phase == PH_DONE);

        // start_pulse: edge detect PRELOAD → RUN transition
        logic was_preload;
        always_ff @(posedge aclk or negedge aresetn) begin
            if (!aresetn) was_preload <= 1'b0;
            else          was_preload <= (phase == PH_PRELOAD);
        end
        assign inference_start = (phase == PH_RUN) && was_preload;

        /* Preload write mux: FIFO (mode 0) vs S_AXIS (mode 1) */
        assign preload_wr_en   = axil_mode[0] ? axis_wr_en   : fifo_wr_en;
        assign preload_wr_addr = axil_mode[0] ? axis_wr_addr : fifo_wr_addr;
        assign preload_wr_data = axil_mode[0] ? axis_wr_data : fifo_wr_data;

        /* ---- S_AXIS pixel accumulator ---- */
        logic [FMAP_DATA_W-1:0]  axis_accum;
        logic [1:0]              axis_lane;
        logic [FMAP_ADDR_W-1:0]  axis_fmap_addr;

        assign s_axis_tready = preload_active && axil_mode[0];

        always_ff @(posedge aclk or negedge aresetn) begin
            if (!aresetn) begin
                axis_accum     <= '0;
                axis_lane      <= 2'd0;
                axis_fmap_addr <= '0;
                axis_wr_en     <= 1'b0;
                axis_wr_addr   <= '0;
                axis_wr_data   <= '0;
                axis_frame_done <= 1'b0;
            end else begin
                axis_wr_en      <= 1'b0;
                axis_frame_done <= 1'b0;

                if (phase == PH_IDLE || (phase == PH_DONE && phase_next == PH_PRELOAD)) begin
                    axis_accum     <= '0;
                    axis_lane      <= 2'd0;
                    axis_fmap_addr <= '0;
                end else if (s_axis_tvalid && s_axis_tready) begin
                    axis_accum[axis_lane * 32 +: 32] <= s_axis_tdata;
                    axis_lane <= axis_lane + 1;

                    if (axis_lane == 2'd3) begin
                        axis_wr_en   <= 1'b1;
                        axis_wr_addr <= axis_fmap_addr;
                        axis_wr_data <= {s_axis_tdata, axis_accum[95:64], axis_accum[63:32], axis_accum[31:0]};
                        axis_fmap_addr <= axis_fmap_addr + 1;

                        if (axis_fmap_addr == FMAP_ADDR_W'(16383))
                            axis_frame_done <= 1'b1;
                    end
                end
            end
        end

        /* ---- Cycle counter ---- */
        always_ff @(posedge aclk or negedge aresetn) begin
            if (!aresetn)          cycle_count <= '0;
            else if (inference_start) cycle_count <= '0;
            else if (phase == PH_RUN) cycle_count <= cycle_count + 1;
        end

        /* ---- AXI-Lite register file ---- */
        axil_regs #(
            .DATA_W     (32),
            .ADDR_W     (AXI_ADDR_W),
            .FMAP_DATA_W(FMAP_DATA_W),
            .FMAP_ADDR_W(FMAP_ADDR_W)
        ) u_axil_regs (
            .clk               (aclk),
            .rst_n             (aresetn),
            .s_axi_awaddr      (s_axi_lite_awaddr),
            .s_axi_awvalid     (s_axi_lite_awvalid),
            .s_axi_awready     (s_axi_lite_awready),
            .s_axi_wdata       (s_axi_lite_wdata),
            .s_axi_wstrb       (s_axi_lite_wstrb),
            .s_axi_wvalid      (s_axi_lite_wvalid),
            .s_axi_wready      (s_axi_lite_wready),
            .s_axi_bresp       (s_axi_lite_bresp),
            .s_axi_bvalid      (s_axi_lite_bvalid),
            .s_axi_bready      (s_axi_lite_bready),
            .s_axi_araddr      (s_axi_lite_araddr),
            .s_axi_arvalid     (s_axi_lite_arvalid),
            .s_axi_arready     (s_axi_lite_arready),
            .s_axi_rdata       (s_axi_lite_rdata),
            .s_axi_rresp       (s_axi_lite_rresp),
            .s_axi_rvalid      (s_axi_lite_rvalid),
            .s_axi_rready      (s_axi_lite_rready),
            .o_start           (axil_start),
            .o_mode            (axil_mode),
            .i_busy            (phase == PH_RUN),
            .i_done            (phase == PH_DONE),
            .i_cycle_count     (cycle_count),
            .i_layer_idx       (curr_layer_idx),
            .o_preload_wr_en   (fifo_wr_en),
            .o_preload_wr_addr (fifo_wr_addr),
            .o_preload_wr_data (fifo_wr_data),
            .o_preload_done    (axil_preload_done),
            .o_result_rd_en    (result_rd_en),
            .o_result_rd_addr  (result_rd_addr),
            .i_result_rd_data  (result_rd_data)
        );

    end else begin : gen_tb_mode
        /* Testbench mode: no AXI, direct start/done */
        assign inference_start = start;
        assign preload_wr_en   = 1'b0;
        assign preload_wr_addr = '0;
        assign preload_wr_data = '0;
        assign preload_active  = 1'b0;
        assign result_rd_en    = 1'b0;
        assign result_rd_addr  = '0;
        assign irq_done        = 1'b0;
        assign s_axis_tready   = 1'b0;

        /* Tie off AXI-Lite outputs */
        assign s_axi_lite_awready = 1'b0;
        assign s_axi_lite_wready  = 1'b0;
        assign s_axi_lite_bresp   = 2'b00;
        assign s_axi_lite_bvalid  = 1'b0;
        assign s_axi_lite_arready = 1'b0;
        assign s_axi_lite_rdata   = '0;
        assign s_axi_lite_rresp   = 2'b00;
        assign s_axi_lite_rvalid  = 1'b0;
    end endgenerate

    /* ================================================================
     *  URAM Port Routing
     * ================================================================ */

    // --- Conv3d input read (read buffer port B) ---
    logic input_rd_en;
    logic [FMAP_ADDR_W-1:0] input_rd_addr;

    generate if (!TB_MODE) begin : gen_fmap_input_rd
        assign input_rd_en = pixel_en_int;
        assign input_rd_addr = (curr_layer_idx == 0)
            ? pixel_addr_int[FMAP_ADDR_W+1:2]
            : (pixel_addr_int[FMAP_ADDR_W-1:0] + curr_pp_rd_offset);
    end else begin : gen_bram_input_rd
        assign input_rd_en   = (curr_layer_idx != 0) && pixel_en_int;
        assign input_rd_addr = pixel_addr_int[FMAP_ADDR_W-1:0] + curr_pp_rd_offset;
    end endgenerate

    // --- RMW output write signals ---
    logic                    rmw_wr_en;
    logic [FMAP_ADDR_W-1:0]  rmw_wr_addr;
    logic [FMAP_DATA_W-1:0]  rmw_wr_data;
    logic                    rmw_rd_en;
    logic [FMAP_ADDR_W-1:0]  rmw_rd_addr;
    logic [FMAP_DATA_W-1:0]  rmw_rd_data;
    assign rmw_rd_data = buf_sel ? fmap_b_dout_b : fmap_a_dout_b;

    // --- fmap_a port allocation (unchanged) ---
    assign fmap_a_en_a   = !buf_sel ? rmw_wr_en     : 1'b0;
    assign fmap_a_we_a   = !buf_sel ? rmw_wr_en     : 1'b0;
    assign fmap_a_addr_a = !buf_sel ? rmw_wr_addr    : '0;
    assign fmap_a_din_a  = !buf_sel ? rmw_wr_data    : '0;
    assign fmap_a_en_b   = !buf_sel ? rmw_rd_en      : input_rd_en;
    assign fmap_a_addr_b = !buf_sel ? rmw_rd_addr    : input_rd_addr;

    // --- fmap_b port allocation (preload + result muxes) ---
    assign fmap_b_en_a   = preload_active ? preload_wr_en  : (buf_sel ? rmw_wr_en   : 1'b0);
    assign fmap_b_we_a   = preload_active ? preload_wr_en  : (buf_sel ? rmw_wr_en   : 1'b0);
    assign fmap_b_addr_a = preload_active ? preload_wr_addr : (buf_sel ? rmw_wr_addr : '0);
    assign fmap_b_din_a  = preload_active ? preload_wr_data : (buf_sel ? rmw_wr_data : '0);

    assign fmap_b_en_b   = result_rd_en ? 1'b1                                 : (buf_sel ? rmw_rd_en   : input_rd_en);
    assign fmap_b_addr_b = result_rd_en ? (FMAP_ADDR_W'(256) + result_rd_addr)  : (buf_sel ? rmw_rd_addr : input_rd_addr);
    assign result_rd_data = fmap_b_dout_b;

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
    logic                        curr_pp_buf_sel;
    logic [13:0]                 curr_pp_rd_offset;
    logic [13:0]                 curr_pp_wr_offset;

    logic [DEPTH_BITS-1:0]            pixel_addr_int;
    logic                             pixel_en_int;
    logic [MAX_PARALLEL*N_BITS-1:0]   pixel_data_int;

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
        .start            (inference_start),
        .done             (inference_done),
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
        .curr_layer_type    (curr_layer_type),
        .curr_layer_idx     (curr_layer_idx),
        .curr_act_size      (curr_act_size),
        .curr_ch_out        (curr_ch_out),
        .curr_pp_buf_sel    (curr_pp_buf_sel),
        .curr_pp_rd_offset  (curr_pp_rd_offset),
        .curr_pp_wr_offset  (curr_pp_wr_offset)
    );

    /* Forward done to output port (testbench mode) */
    assign done = inference_done;

    /* ================================================================
     *  Pixel Input Mux & URAM Read Routing
     * ================================================================ */
    logic buf_sel;
    assign buf_sel = curr_pp_buf_sel;

    assign pixel_bram_addr = pixel_addr_int;
    assign pixel_bram_en   = pixel_en_int;

    logic [FMAP_DATA_W-1:0] uram_input_rd_data;
    assign uram_input_rd_data = buf_sel ? fmap_a_dout_b : fmap_b_dout_b;

    generate if (!TB_MODE) begin : gen_fmap_pixel_mux
        logic [1:0] pixel_sel_r;
        always_ff @(posedge aclk) pixel_sel_r <= pixel_addr_int[1:0];

        wire [31:0]            packed_pixel_32 = fmap_b_dout_b[pixel_sel_r * 32 +: 32];
        wire [FMAP_DATA_W-1:0] expanded_pixel  = {{(FMAP_DATA_W-24){1'b0}}, packed_pixel_32[23:0]};

        assign pixel_data_int = (curr_layer_idx == 0) ? expanded_pixel : uram_input_rd_data;
    end else begin : gen_bram_pixel_mux
        assign pixel_data_int = (curr_layer_idx == 0) ? pixel_bram_data : uram_input_rd_data;
    end endgenerate

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
     * ================================================================ */
    logic                     rmw_s0_valid;
    logic [FMAP_ADDR_W-1:0]  rmw_s0_addr;
    logic signed [N_BITS-1:0] rmw_s0_data;
    logic [3:0]               rmw_s0_byte_pos;

    wire [8:0] h_out = (curr_layer_type == CONV3_POOL) ? (curr_act_size >> 1) : curr_act_size;
    wire [FMAP_ADDR_W-1:0] rmw_base_addr = curr_pp_wr_offset + (curr_ch_out >> 4) * h_out * h_out;

    always_ff @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            rmw_s0_valid <= 1'b0;
        end else begin
            rmw_s0_valid    <= res_write_en;
            rmw_s0_addr     <= rmw_base_addr + res_write_addr[FMAP_ADDR_W-1:0];
            rmw_s0_data     <= res_write_data;
            rmw_s0_byte_pos <= curr_ch_out[3:0];
        end
    end

    assign rmw_rd_en   = res_write_en;
    assign rmw_rd_addr = rmw_base_addr + res_write_addr[FMAP_ADDR_W-1:0];

    logic [FMAP_DATA_W-1:0] spliced_word;
    always_comb begin
        spliced_word = (rmw_s0_byte_pos == 4'd0) ? {FMAP_DATA_W{1'b0}} : rmw_rd_data;
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

endmodule
