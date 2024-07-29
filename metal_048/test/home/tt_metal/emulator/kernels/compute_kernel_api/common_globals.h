#pragma once

#include <cstdint>

#include "kernel_structs.h"

#define SYNC SyncHalf

#define FORCE_INLINE inline
#define ALWI inline

#define API extern "C"

// temporarily borrowed from "hostdevcommon/common_runtime_address_map.h"

constexpr static uint32_t TRISC_L1_ARG_BASE = 105 * 1024;

// Some compute kernels use unqualified API functions from 'ckernel'

namespace ckernel { }

using namespace ckernel;

