// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#ifdef METAL_057

#include "tt-metalium/host_api.hpp"

#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/math.hpp"

#include "hostdevcommon/kernel_structs.h"

#else

#include "tt_metal/host_api.hpp"

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"

#include "tt_metal/hostdevcommon/kernel_structs.h"

#endif // METAL_057

namespace ronin {
namespace tanto {
namespace host {

namespace metal = tt::tt_metal;

using metal::DataMovementProcessor;
using metal::NOC;

} // namespace host
} // namespace tanto
} // namespace ronin

