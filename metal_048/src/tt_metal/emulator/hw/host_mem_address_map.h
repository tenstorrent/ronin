#pragma once

#include "config/hw_arch.h"

#if HW_ARCH_CONFIG == HW_ARCH_GRAYSKULL
#include "third_party/umd/src/firmware/riscv/grayskull/host_mem_address_map.h"
#elif HW_ARCH_CONFIG == HW_ARCH_WORMHOLE
#include "third_party/umd/src/firmware/riscv/wormhole/host_mem_address_map.h"
#else
#error "Invalid HW_ARCH_CONFIG value"
#endif

