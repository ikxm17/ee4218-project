// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xtinyissimo_layer_top.h"

extern XTinyissimo_layer_top_Config XTinyissimo_layer_top_ConfigTable[];

#ifdef SDT
XTinyissimo_layer_top_Config *XTinyissimo_layer_top_LookupConfig(UINTPTR BaseAddress) {
	XTinyissimo_layer_top_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XTinyissimo_layer_top_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XTinyissimo_layer_top_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XTinyissimo_layer_top_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XTinyissimo_layer_top_Initialize(XTinyissimo_layer_top *InstancePtr, UINTPTR BaseAddress) {
	XTinyissimo_layer_top_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XTinyissimo_layer_top_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XTinyissimo_layer_top_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XTinyissimo_layer_top_Config *XTinyissimo_layer_top_LookupConfig(u16 DeviceId) {
	XTinyissimo_layer_top_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XTINYISSIMO_LAYER_TOP_NUM_INSTANCES; Index++) {
		if (XTinyissimo_layer_top_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XTinyissimo_layer_top_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XTinyissimo_layer_top_Initialize(XTinyissimo_layer_top *InstancePtr, u16 DeviceId) {
	XTinyissimo_layer_top_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XTinyissimo_layer_top_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XTinyissimo_layer_top_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

