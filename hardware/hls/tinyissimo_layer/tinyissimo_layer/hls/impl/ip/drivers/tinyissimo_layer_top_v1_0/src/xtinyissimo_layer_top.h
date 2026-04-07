// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XTINYISSIMO_LAYER_TOP_H
#define XTINYISSIMO_LAYER_TOP_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xtinyissimo_layer_top_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XTinyissimo_layer_top_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XTinyissimo_layer_top;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XTinyissimo_layer_top_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XTinyissimo_layer_top_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XTinyissimo_layer_top_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XTinyissimo_layer_top_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XTinyissimo_layer_top_Initialize(XTinyissimo_layer_top *InstancePtr, UINTPTR BaseAddress);
XTinyissimo_layer_top_Config* XTinyissimo_layer_top_LookupConfig(UINTPTR BaseAddress);
#else
int XTinyissimo_layer_top_Initialize(XTinyissimo_layer_top *InstancePtr, u16 DeviceId);
XTinyissimo_layer_top_Config* XTinyissimo_layer_top_LookupConfig(u16 DeviceId);
#endif
int XTinyissimo_layer_top_CfgInitialize(XTinyissimo_layer_top *InstancePtr, XTinyissimo_layer_top_Config *ConfigPtr);
#else
int XTinyissimo_layer_top_Initialize(XTinyissimo_layer_top *InstancePtr, const char* InstanceName);
int XTinyissimo_layer_top_Release(XTinyissimo_layer_top *InstancePtr);
#endif

void XTinyissimo_layer_top_Start(XTinyissimo_layer_top *InstancePtr);
u32 XTinyissimo_layer_top_IsDone(XTinyissimo_layer_top *InstancePtr);
u32 XTinyissimo_layer_top_IsIdle(XTinyissimo_layer_top *InstancePtr);
u32 XTinyissimo_layer_top_IsReady(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_EnableAutoRestart(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_DisableAutoRestart(XTinyissimo_layer_top *InstancePtr);

void XTinyissimo_layer_top_Set_in_h(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_in_h(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_in_w(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_in_w(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_in_c(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_in_c(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_out_c(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_out_c(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_kh(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_kh(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_kw(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_kw(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_pad_h(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_pad_h(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_pad_w(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_pad_w(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_use_maxpool(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_use_maxpool(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_use_silu(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_use_silu(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_layer_idx(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_layer_idx(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_zp_in(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_zp_in(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_zp_out(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_zp_out(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_wt_base(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_wt_base(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_Set_qp_base(XTinyissimo_layer_top *InstancePtr, u32 Data);
u32 XTinyissimo_layer_top_Get_qp_base(XTinyissimo_layer_top *InstancePtr);

void XTinyissimo_layer_top_InterruptGlobalEnable(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_InterruptGlobalDisable(XTinyissimo_layer_top *InstancePtr);
void XTinyissimo_layer_top_InterruptEnable(XTinyissimo_layer_top *InstancePtr, u32 Mask);
void XTinyissimo_layer_top_InterruptDisable(XTinyissimo_layer_top *InstancePtr, u32 Mask);
void XTinyissimo_layer_top_InterruptClear(XTinyissimo_layer_top *InstancePtr, u32 Mask);
u32 XTinyissimo_layer_top_InterruptGetEnabled(XTinyissimo_layer_top *InstancePtr);
u32 XTinyissimo_layer_top_InterruptGetStatus(XTinyissimo_layer_top *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
