// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "hw/inc/grayskull/core_config.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "hw/inc/wormhole/core_config.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

