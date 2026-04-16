`timescale 1ns / 1ps
`include "layer_config.svh"

/*
 *  cycle_monitor — observational per-(layer, stage) cycle accounting.
 *
 *  Bound into inference_top (see tb_tinyissimoyolo_accel.sv) so that all
 *  observed signals are passed in through the bind port map. No RTL is
 *  modified; this module only reads.
 *
 *  Primary buckets (mutually exclusive — partition the inference window):
 *      WEIGHT_LOAD    state == S_LOAD
 *      COMPUTE        state == S_COMPUTE
 *      NEXT_CHOUT     state == S_NEXT_CHOUT
 *      NEXT_LAYER     state == S_NEXT_LAYER
 *
 *  Parallel activity counters (overlap allowed):
 *      ACT_ACTIVE     act_out_valid high
 *      POOL_ACTIVE    pool_out_valid high
 *      RMW_S0_ACTIVE  rmw_s0_valid high
 *      RMW_WR_ACTIVE  out_buf_wr_en high
 *
 *  Counts are gated by phase == PH_RUN so they exactly partition the same
 *  window the existing `cycle_count` register measures. On phase falling
 *  out of PH_RUN, the monitor dumps cycle_breakdown.csv and prints a
 *  PASS/FAIL line comparing Σ per_layer_total against cycle_count.
 */
module cycle_monitor (
    input  logic        clk,
    input  logic        rstn,

    // From inference_top scope
    input  logic [2:0]  phase,
    input  logic [31:0] cycle_count,

    // From inference_top.u_inference_hdl scope
    input  logic [2:0]  hdl_state,
    input  logic [4:0]  layer_idx,
    input  logic [7:0]  ch_out,
    input  logic        act_out_valid,
    input  logic        pool_out_valid,
    input  logic        rmw_s0_valid,
    input  logic        out_buf_wr_en
);

    // Must match inference_hdl.sv:107-112 and inference_top.sv:255-258
    localparam logic [2:0] S_IDLE       = 3'd0;
    localparam logic [2:0] S_LOAD       = 3'd1;
    localparam logic [2:0] S_COMPUTE    = 3'd2;
    localparam logic [2:0] S_NEXT_CHOUT = 3'd3;
    localparam logic [2:0] S_NEXT_LAYER = 3'd4;
    localparam logic [2:0] PH_RUN       = 3'd2;

    localparam int N_LAYERS = NUM_LAYERS;  // 17, from layer_config.svh
    localparam int N_BUCKETS = 4;
    localparam int BKT_LOAD       = 0;
    localparam int BKT_COMPUTE    = 1;
    localparam int BKT_NEXT_CHOUT = 2;
    localparam int BKT_NEXT_LAYER = 3;

    // Storage. `int unsigned` is 32-bit — plenty for ~3M-cycle inference.
    int unsigned per_layer       [N_LAYERS][N_BUCKETS];
    int unsigned per_layer_total [N_LAYERS];
    int unsigned act_active      [N_LAYERS];
    int unsigned pool_active     [N_LAYERS];
    int unsigned rmw_s0_active   [N_LAYERS];
    int unsigned rmw_wr_active   [N_LAYERS];

    initial begin
        for (int i = 0; i < N_LAYERS; i++) begin
            per_layer_total[i] = 0;
            act_active[i]      = 0;
            pool_active[i]     = 0;
            rmw_s0_active[i]   = 0;
            rmw_wr_active[i]   = 0;
            for (int b = 0; b < N_BUCKETS; b++) per_layer[i][b] = 0;
        end
    end

    wire inference_running = (phase == PH_RUN);

    /* Per-cycle accumulation. Gated by inference_running so the sum
     * across all primary buckets matches the existing cycle_count
     * register exactly (both are gated by the same condition). */
    always @(posedge clk) begin
        if (rstn && inference_running && layer_idx < N_LAYERS) begin
            unique case (hdl_state)
                S_LOAD:       per_layer[layer_idx][BKT_LOAD]++;
                S_COMPUTE:    per_layer[layer_idx][BKT_COMPUTE]++;
                S_NEXT_CHOUT: per_layer[layer_idx][BKT_NEXT_CHOUT]++;
                S_NEXT_LAYER: per_layer[layer_idx][BKT_NEXT_LAYER]++;
                default: ;  // S_IDLE not expected during PH_RUN; ignore
            endcase
            per_layer_total[layer_idx]++;

            if (act_out_valid)  act_active[layer_idx]++;
            if (pool_out_valid) pool_active[layer_idx]++;
            if (rmw_s0_valid)   rmw_s0_active[layer_idx]++;
            if (out_buf_wr_en)  rmw_wr_active[layer_idx]++;
        end
    end

    /* Edge detect on inference_running falling — dump CSV and check parity. */
    logic inference_running_d;
    always @(posedge clk) begin
        if (!rstn) inference_running_d <= 1'b0;
        else      inference_running_d <= inference_running;
    end

    always @(posedge clk) begin
        if (rstn && inference_running_d && !inference_running) begin
            dump_csv();
        end
    end

    task automatic dump_csv();
        int fd;
        int unsigned sum_total;
        layer_cfg_t cfg;  // local, to dodge XSim quirk on LAYER_CFG[var].field

        fd = $fopen("cycle_breakdown.csv", "w");
        if (fd == 0) begin
            $display("[cycle_monitor] ERROR: failed to open cycle_breakdown.csv");
            return;
        end

        $fwrite(fd, "layer_idx,layer_type,h_in,cin,cout,total,weight_load,compute,next_chout,next_layer,act_active,pool_active,rmw_s0_active,rmw_wr_active\n");

        sum_total = 0;
        for (int i = 0; i < N_LAYERS; i++) begin
            cfg = LAYER_CFG[i];
            $fwrite(fd, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                i,
                cfg.layer_type,
                cfg.h_in,
                cfg.cin,
                cfg.cout,
                per_layer_total[i],
                per_layer[i][BKT_LOAD],
                per_layer[i][BKT_COMPUTE],
                per_layer[i][BKT_NEXT_CHOUT],
                per_layer[i][BKT_NEXT_LAYER],
                act_active[i],
                pool_active[i],
                rmw_s0_active[i],
                rmw_wr_active[i]);
            sum_total += per_layer_total[i];
        end
        $fclose(fd);

        $display("[cycle_monitor] CSV written to cycle_breakdown.csv");
        if (sum_total == cycle_count) begin
            $display("[cycle_monitor] PASS: sum(per_layer_total) = %0d = cycle_count",
                     sum_total);
        end else begin
            $display("[cycle_monitor] FAIL: sum(per_layer_total) = %0d, cycle_count = %0d, delta = %0d",
                     sum_total, cycle_count, sum_total - cycle_count);
        end
    endtask

endmodule
