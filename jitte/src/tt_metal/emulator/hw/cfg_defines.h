// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "hw/inc/grayskull/cfg_defines.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "hw/inc/wormhole/wormhole_b0_defines/cfg_defines.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

