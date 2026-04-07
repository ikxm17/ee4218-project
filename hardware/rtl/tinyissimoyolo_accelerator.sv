`timescale 1ns / 1ps

module tinyissimoyolo_accelerator #(
    parameter AXI_ADDR_W = 13
)(
    input  logic aclk,
    input  logic aresetn,

    /* --- AXI4-Lite slave --- */
    input  logic [AXI_ADDR_W-1:0] s_axi_lite_awaddr,
    input  logic                   s_axi_lite_awvalid,
    output logic                   s_axi_lite_awready,
    input  logic [31:0]            s_axi_lite_wdata,
    input  logic [3:0]             s_axi_lite_wstrb,
    input  logic                   s_axi_lite_wvalid,
    output logic                   s_axi_lite_wready,
    output logic [1:0]             s_axi_lite_bresp,
    output logic                   s_axi_lite_bvalid,
    input  logic                   s_axi_lite_bready,
    input  logic [AXI_ADDR_W-1:0] s_axi_lite_araddr,
    input  logic                   s_axi_lite_arvalid,
    output logic                   s_axi_lite_arready,
    output logic [31:0]            s_axi_lite_rdata,
    output logic [1:0]             s_axi_lite_rresp,
    output logic                   s_axi_lite_rvalid,
    input  logic                   s_axi_lite_rready,

    /* --- AXI4-Stream slave — camera pixels --- */
    input  logic [31:0]            s_axis_tdata,
    input  logic                   s_axis_tvalid,
    input  logic                   s_axis_tlast,
    input  logic [0:0]             s_axis_tuser,
    output logic                   s_axis_tready,

    /* --- Interrupt --- */
    output logic                   irq_done
);

    inference_top #(
        .TB_MODE    (0),
        .AXI_ADDR_W (AXI_ADDR_W)
    ) u_core (
        .aclk                (aclk),
        .aresetn             (aresetn),

        /* TB ports — tied off */
        .start               (1'b0),
        .done                (),
        .pixel_bram_addr     (),
        .pixel_bram_en       (),
        .pixel_bram_data     ('0),
        .res_write_en        (),
        .res_write_addr      (),
        .res_write_data      (),

        /* AXI-Lite */
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

        /* AXI-Stream */
        .s_axis_tdata        (s_axis_tdata),
        .s_axis_tvalid       (s_axis_tvalid),
        .s_axis_tlast        (s_axis_tlast),
        .s_axis_tuser        (s_axis_tuser),
        .s_axis_tready       (s_axis_tready),

        /* Interrupt */
        .irq_done            (irq_done)
    );

endmodule
