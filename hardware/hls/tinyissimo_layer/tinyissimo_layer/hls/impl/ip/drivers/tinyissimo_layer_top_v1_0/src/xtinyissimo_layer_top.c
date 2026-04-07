// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xtinyissimo_layer_top.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XTinyissimo_layer_top_CfgInitialize(XTinyissimo_layer_top *InstancePtr, XTinyissimo_layer_top_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XTinyissimo_layer_top_Start(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL) & 0x80;
    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XTinyissimo_layer_top_IsDone(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XTinyissimo_layer_top_IsIdle(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XTinyissimo_layer_top_IsReady(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XTinyissimo_layer_top_EnableAutoRestart(XTinyissimo_layer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XTinyissimo_layer_top_DisableAutoRestart(XTinyissimo_layer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_AP_CTRL, 0);
}

void XTinyissimo_layer_top_Set_in_h(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_H_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_in_h(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_H_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_in_w(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_W_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_in_w(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_W_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_in_c(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_C_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_in_c(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IN_C_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_out_c(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_OUT_C_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_out_c(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_OUT_C_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_kh(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_KH_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_kh(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_KH_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_kw(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_KW_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_kw(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_KW_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_pad_h(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_PAD_H_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_pad_h(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_PAD_H_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_pad_w(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_PAD_W_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_pad_w(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_PAD_W_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_use_maxpool(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_USE_MAXPOOL_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_use_maxpool(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_USE_MAXPOOL_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_use_silu(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_USE_SILU_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_use_silu(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_USE_SILU_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_layer_idx(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_LAYER_IDX_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_layer_idx(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_LAYER_IDX_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_zp_in(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ZP_IN_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_zp_in(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ZP_IN_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_zp_out(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ZP_OUT_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_zp_out(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ZP_OUT_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_wt_base(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_WT_BASE_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_wt_base(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_WT_BASE_DATA);
    return Data;
}

void XTinyissimo_layer_top_Set_qp_base(XTinyissimo_layer_top *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_QP_BASE_DATA, Data);
}

u32 XTinyissimo_layer_top_Get_qp_base(XTinyissimo_layer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_QP_BASE_DATA);
    return Data;
}

void XTinyissimo_layer_top_InterruptGlobalEnable(XTinyissimo_layer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_GIE, 1);
}

void XTinyissimo_layer_top_InterruptGlobalDisable(XTinyissimo_layer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_GIE, 0);
}

void XTinyissimo_layer_top_InterruptEnable(XTinyissimo_layer_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IER);
    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IER, Register | Mask);
}

void XTinyissimo_layer_top_InterruptDisable(XTinyissimo_layer_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IER);
    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IER, Register & (~Mask));
}

void XTinyissimo_layer_top_InterruptClear(XTinyissimo_layer_top *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XTinyissimo_layer_top_WriteReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ISR, Mask);
}

u32 XTinyissimo_layer_top_InterruptGetEnabled(XTinyissimo_layer_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_IER);
}

u32 XTinyissimo_layer_top_InterruptGetStatus(XTinyissimo_layer_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XTinyissimo_layer_top_ReadReg(InstancePtr->Control_BaseAddress, XTINYISSIMO_LAYER_TOP_CONTROL_ADDR_ISR);
}

