#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "hw/inc/grayskull/noc/noc_overlay_parameters.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "hw/inc/wormhole/noc/noc_overlay_parameters.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

