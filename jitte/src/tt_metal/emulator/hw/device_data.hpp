// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "third_party/umd/device/grayskull/device_data.hpp"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "third_party/umd/device/wormhole/device_data.hpp"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

