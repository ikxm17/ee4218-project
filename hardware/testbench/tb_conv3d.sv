`timescale 1ns / 1ps

// =============================================================================
//  tb_conv3d
//
//  Single-layer (layer 0) verification of the conv3d primitive, implemented
//  as a 1-layer instance of tb_inference_hdl. The Python golden generator at
//  hardware/scripts/generate_conv3d_golden.py is the canonical oracle and
//  produces golden_layer0_uram.mem already; this wrapper just runs the
//  existing inference_hdl pipeline for layer 0 only and lets tb_inference_hdl
//  do the verification.
//
//  History: this used to maintain a hand-written reference model that drifted
//  out of sync with the hardware quantization (saturation truncation +
//  arithmetic offset). It was retired in favour of the Python oracle.
//
//  What this exercises:
//    - conv3d.v on layer 0 (256x256x3 -> 128x128x16, K=3, CONV3_POOL)
//    - all 16 output channels (the old testbench only checked one)
//    - the activation LUT, max-pool, and URAM RMW writer for layer 0
// =============================================================================
module tb_conv3d;

    tb_inference_hdl #(
        .NUM_TEST_LAYERS(1)
    ) u_tb ();

endmodule
