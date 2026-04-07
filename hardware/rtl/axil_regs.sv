`timescale 1ns / 1ps
/*
 * AXI4-Lite Slave — TinyissimoYOLO Accelerator Registers
 *
 * Register map:
 *   0x000  CTRL       R/W  [0]=start, [1]=pixel_fifo_rst, [7]=soft_reset
 *   0x004  STATUS     R    [0]=busy, [1]=done, [2]=idle, [3]=preload_done
 *   0x008  MODE       R/W  [0]=input_src (0=FIFO, 1=S_AXIS), [4]=engine (KIV)
 *   0x00C  CYCLE_CNT  R    inference cycle counter
 *   0x010  LAYER_IDX  R    [4:0]=current layer during inference
 *   0x020  PIXEL_FIFO W    sequential pixel write (65536 × 32-bit)
 *   0x024  PIXEL_CNT  R    number of 32-bit words written to FIFO
 *   0x100  RESULT     R    detection results (320 × 128-bit = 1280 × 32-bit)
 *         ...0x14FF
 */
module axil_regs #(
    parameter DATA_W     = 32,
    parameter ADDR_W     = 13,
    parameter FMAP_DATA_W = 128,
    parameter FMAP_ADDR_W = 14
)(
    input  logic                      clk,
    input  logic                      rst_n,

    /* AXI-Lite slave */
    input  logic [ADDR_W-1:0]        s_axi_awaddr,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,
    input  logic [DATA_W-1:0]        s_axi_wdata,
    input  logic [DATA_W/8-1:0]      s_axi_wstrb,
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,
    input  logic [ADDR_W-1:0]        s_axi_araddr,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,
    output logic [DATA_W-1:0]        s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,

    /* Control outputs */
    output logic                      o_start,
    output logic [1:0]                o_mode,

    /* Status inputs */
    input  logic                      i_busy,
    input  logic                      i_done,
    input  logic [31:0]               i_cycle_count,
    input  logic [4:0]                i_layer_idx,

    /* Pixel FIFO → preload write */
    output logic                      o_preload_wr_en,
    output logic [FMAP_ADDR_W-1:0]    o_preload_wr_addr,
    output logic [FMAP_DATA_W-1:0]    o_preload_wr_data,
    output logic                      o_preload_done,

    /* Result read */
    output logic                      o_result_rd_en,
    output logic [FMAP_ADDR_W-1:0]    o_result_rd_addr,
    input  logic [FMAP_DATA_W-1:0]    i_result_rd_data
);

    /* ================================================================
     *  Address decode
     * ================================================================ */
    localparam [ADDR_W-1:0] ADDR_CTRL        = 13'h000,
                            ADDR_STATUS      = 13'h004,
                            ADDR_MODE        = 13'h008,
                            ADDR_CYCLE_CNT   = 13'h00C,
                            ADDR_LAYER_IDX   = 13'h010,
                            ADDR_PIXEL_FIFO  = 13'h020,
                            ADDR_PIXEL_CNT   = 13'h024,
                            ADDR_RESULT_BASE = 13'h100,
                            ADDR_RESULT_END  = 13'h14FF;

    /* ================================================================
     *  Internal registers
     * ================================================================ */
    logic [31:0] reg_ctrl;
    logic [31:0] reg_mode;

    logic [FMAP_DATA_W-1:0]  pixel_accum;
    logic [1:0]              pixel_lane;
    logic [FMAP_ADDR_W-1:0]  pixel_fmap_addr;
    logic [16:0]             pixel_count;
    logic                    preload_done_r;

    /* ================================================================
     *  Write channel handshake
     * ================================================================ */
    logic [ADDR_W-1:0] wr_addr;
    logic               wr_pending;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b1;
            s_axi_wready  <= 1'b1;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            wr_addr       <= '0;
            wr_pending    <= 1'b0;
        end else begin
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid  <= 1'b0;
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
            end

            if (s_axi_awvalid && s_axi_awready) begin
                wr_addr <= s_axi_awaddr;
                if (s_axi_wvalid && s_axi_wready) begin
                    s_axi_bvalid  <= 1'b1;
                    s_axi_awready <= 1'b0;
                    s_axi_wready  <= 1'b0;
                end else begin
                    wr_pending    <= 1'b1;
                    s_axi_awready <= 1'b0;
                end
            end else if (wr_pending && s_axi_wvalid && s_axi_wready) begin
                s_axi_bvalid <= 1'b1;
                s_axi_wready <= 1'b0;
                wr_pending   <= 1'b0;
            end
        end
    end

    /* ================================================================
     *  Write decode
     * ================================================================ */
    wire wr_fire = s_axi_wvalid && s_axi_wready;
    wire [ADDR_W-1:0] wr_addr_mux = (s_axi_awvalid && s_axi_awready)
                                     ? s_axi_awaddr : wr_addr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ctrl <= '0;
            reg_mode <= '0;
        end else begin
            if (reg_ctrl[0]) reg_ctrl[0] <= 1'b0;
            if (reg_ctrl[1]) reg_ctrl[1] <= 1'b0;

            if (wr_fire) begin
                case (wr_addr_mux)
                    ADDR_CTRL: reg_ctrl <= s_axi_wdata;
                    ADDR_MODE: reg_mode <= s_axi_wdata;
                    default: ;
                endcase
            end
        end
    end

    /* ================================================================
     *  Pixel FIFO — accumulate 4 × 32-bit → 128-bit fmap_b word
     * ================================================================ */
    wire pixel_fifo_write = wr_fire && (wr_addr_mux == ADDR_PIXEL_FIFO);
    wire pixel_fifo_rst   = reg_ctrl[1] || reg_ctrl[7];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_accum       <= '0;
            pixel_lane        <= 2'd0;
            pixel_fmap_addr   <= '0;
            pixel_count       <= '0;
            preload_done_r    <= 1'b0;
            o_preload_wr_en   <= 1'b0;
            o_preload_wr_addr <= '0;
            o_preload_wr_data <= '0;
        end else begin
            o_preload_wr_en <= 1'b0;

            if (pixel_fifo_rst) begin
                pixel_accum     <= '0;
                pixel_lane      <= 2'd0;
                pixel_fmap_addr <= '0;
                pixel_count     <= '0;
                preload_done_r  <= 1'b0;
            end else if (pixel_fifo_write) begin
                pixel_accum[pixel_lane * 32 +: 32] <= s_axi_wdata;
                pixel_count <= pixel_count + 1;
                pixel_lane  <= pixel_lane + 1;

                if (pixel_lane == 2'd3) begin
                    o_preload_wr_en   <= 1'b1;
                    o_preload_wr_addr <= pixel_fmap_addr;
                    o_preload_wr_data <= {s_axi_wdata, pixel_accum[95:64], pixel_accum[63:32], pixel_accum[31:0]};
                    pixel_fmap_addr   <= pixel_fmap_addr + 1;
                end
            end

            if (pixel_count == 17'd65536)
                preload_done_r <= 1'b1;
        end
    end

    assign o_preload_done = preload_done_r;

    /* ================================================================
     *  Read channel handshake — 3-cycle (addr → wait → data)
     * ================================================================ */
    typedef enum logic [1:0] {
        RD_ADDR = 2'd0,
        RD_WAIT = 2'd1,
        RD_DATA = 2'd2
    } rd_state_t;
    rd_state_t rd_state;

    logic [ADDR_W-1:0] rd_addr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b1;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= '0;
            rd_state      <= RD_ADDR;
            rd_addr       <= '0;
        end else begin
            case (rd_state)
                RD_ADDR: begin
                    if (s_axi_arvalid && s_axi_arready) begin
                        rd_addr       <= s_axi_araddr;
                        s_axi_arready <= 1'b0;
                        rd_state      <= RD_WAIT;
                    end
                end
                RD_WAIT: begin
                    s_axi_rdata  <= rd_data_mux;
                    s_axi_rvalid <= 1'b1;
                    rd_state     <= RD_DATA;
                end
                RD_DATA: begin
                    if (s_axi_rready) begin
                        s_axi_rvalid  <= 1'b0;
                        s_axi_arready <= 1'b1;
                        rd_state      <= RD_ADDR;
                    end
                end
                default: rd_state <= RD_ADDR;
            endcase
        end
    end

    /* ================================================================
     *  Read decode
     * ================================================================ */
    wire is_result_read    = (rd_addr >= ADDR_RESULT_BASE) && (rd_addr <= ADDR_RESULT_END);
    wire is_result_read_ar = (s_axi_araddr >= ADDR_RESULT_BASE) && (s_axi_araddr <= ADDR_RESULT_END);

    assign o_result_rd_en   = (rd_state == RD_ADDR) && s_axi_arvalid && s_axi_arready && is_result_read_ar;
    assign o_result_rd_addr = (s_axi_araddr - ADDR_RESULT_BASE) >> 4;

    wire [DATA_W-1:0] result_data_slice = i_result_rd_data[rd_addr[3:2] * 32 +: 32];

    logic [DATA_W-1:0] rd_data_mux;
    always_comb begin
        if (is_result_read)
            rd_data_mux = result_data_slice;
        else begin
            case (rd_addr)
                ADDR_CTRL:      rd_data_mux = reg_ctrl;
                ADDR_STATUS:    rd_data_mux = {28'd0, preload_done_r, ~i_busy, i_done, i_busy};
                ADDR_MODE:      rd_data_mux = reg_mode;
                ADDR_CYCLE_CNT: rd_data_mux = i_cycle_count;
                ADDR_LAYER_IDX: rd_data_mux = {27'd0, i_layer_idx};
                ADDR_PIXEL_CNT: rd_data_mux = {15'd0, pixel_count};
                default:        rd_data_mux = 32'hDEAD_BEEF;
            endcase
        end
    end

    /* ================================================================
     *  Control outputs
     * ================================================================ */
    assign o_start = reg_ctrl[0];
    assign o_mode  = reg_mode[1:0];

endmodule
