#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "hw/inc/grayskull/tensix_types.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "hw/inc/wormhole/wormhole_b0_defines/tensix_types.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

