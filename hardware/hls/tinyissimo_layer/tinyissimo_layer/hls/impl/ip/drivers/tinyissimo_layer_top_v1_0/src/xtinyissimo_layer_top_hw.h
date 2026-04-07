// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of in_h
//        bit 31~0 - in_h[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of in_w
//        bit 31~0 - in_w[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of in_c
//        bit 31~0 - in_c[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of out_c
//        bit 31~0 - out_c[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of kh
//        bit 31~0 - kh[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of kw
//        bit 31~0 - kw[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of pad_h
//        bit 31~0 - pad_h[31:0] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of pad_w
//        bit 31~0 - pad_w[31:0] (Read/Write)
// 0x4c : reserved
// 0x50 : Data signal of use_maxpool
//        bit 0  - use_maxpool[0] (Read/Write)
//        others - reserved
// 0x54 : reserved
// 0x58 : Data signal of use_silu
//        bit 0  - use_silu[0] (Read/Write)
//        others - reserved
// 0x5c : reserved
// 0x60 : Data signal of layer_idx
//        bit 31~0 - layer_idx[31:0] (Read/Write)
// 0x64 : reserved
// 0x68 : Data signal of zp_in
//        bit 7~0 - zp_in[7:0] (Read/Write)
//        others  - reserved
// 0x6c : reserved
// 0x70 : Data signal of zp_out
//        bit 7~0 - zp_out[7:0] (Read/Write)
//        others  - reserved
// 0x74 : reserved
// 0x78 : Data signal of wt_base
//        bit 31~0 - wt_base[31:0] (Read/Write)
// 0x7c : reserved
// 0x80 : Data signal of qp_base
//        bit 31~0 - qp_base[31:0] (Read/Write)
// 0x84 : reserved
// 0x88 : Data signal of fmap_rd_offset
//        bit 31~0 - fmap_rd_offset[31:0] (Read/Write)
// 0x8c : reserved
// 0x90 : Data signal of fmap_wr_offset
//        bit 31~0 - fmap_wr_offset[31:0] (Read/Write)
// 0x94 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL             0x00
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_GIE                 0x04
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IER                 0x08
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ISR                 0x0c
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_H_DATA           0x10
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_IN_H_DATA           32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_W_DATA           0x18
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_IN_W_DATA           32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_C_DATA           0x20
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_IN_C_DATA           32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_OUT_C_DATA          0x28
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_OUT_C_DATA          32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_KH_DATA             0x30
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_KH_DATA             32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_KW_DATA             0x38
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_KW_DATA             32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_PAD_H_DATA          0x40
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_PAD_H_DATA          32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_PAD_W_DATA          0x48
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_PAD_W_DATA          32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_USE_MAXPOOL_DATA    0x50
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_USE_MAXPOOL_DATA    1
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_USE_SILU_DATA       0x58
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_USE_SILU_DATA       1
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_LAYER_IDX_DATA      0x60
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_LAYER_IDX_DATA      32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ZP_IN_DATA          0x68
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_ZP_IN_DATA          8
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ZP_OUT_DATA         0x70
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_ZP_OUT_DATA         8
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_WT_BASE_DATA        0x78
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_WT_BASE_DATA        32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_QP_BASE_DATA        0x80
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_QP_BASE_DATA        32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_FMAP_RD_OFFSET_DATA 0x88
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_FMAP_RD_OFFSET_DATA 32
#define XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_FMAP_WR_OFFSET_DATA 0x90
#define XTINYISSIMO_LAYER_TOP_CONTROL_BITS_FMAP_WR_OFFSET_DATA 32

