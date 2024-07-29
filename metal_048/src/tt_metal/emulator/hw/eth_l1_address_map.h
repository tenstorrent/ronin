#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "hw/inc/grayskull/eth_l1_address_map.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "hw/inc/wormhole/eth_l1_address_map.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

