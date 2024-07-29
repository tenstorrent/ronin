#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "hw/inc/grayskull/dev_mem_map_versim.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "hw/inc/wormhole/dev_mem_map_versim.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

