// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "hw/inc/grayskull/eth_interface.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "hw/inc/wormhole/eth_interface.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

