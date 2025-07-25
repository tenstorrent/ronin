// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define HW_ARCH_GRAYSKULL 1
#define HW_ARCH_WROMHOLE 2

//
//    Currently, the TT-Metal host code must be recompiled for each architecture
//    Set HW_ARCH_CONFIG to specify architecture for compilation
//

#ifndef HW_ARCH_CONFIG
//#define HW_ARCH_CONFIG HW_ARCH_GRAYSKULL
#define HW_ARCH_CONFIG HW_ARCH_WORMHOLE
#endif

