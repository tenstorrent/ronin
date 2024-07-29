// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/** \file base.hpp
 * The basic enums and data structures used by the rest of code base.
 */
#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <array>
#include <vector>
#include <map>

#include <boost/functional/hash.hpp>

#include "tt_metal/common/tt_backend_api_types.hpp" // These are the types exported to frontend team...
#include "tt_metal/common/assert.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "eth_l1_address_map.h"
#include "common/constants.hpp"
#include "common/base_types.hpp"

using std::array;
using std::ostream;
using std::uint8_t;
using std::uint32_t;
using std::uint64_t;
using std::vector;
using std::string;
using std::size_t;
using std::map;

inline constexpr uint32_t align(uint32_t addr, uint32_t alignment) { return ((addr - 1) | (alignment - 1)) + 1; }


namespace tt
{

/**
 * @brief Specifies the target devices on which the graph can be run.
*/
enum class TargetDevice : uint8_t
{
    Model = 0,
    Versim = 1,
    Silicon = 2,
    Golden = 3,
    Invalid = 0xFF,
};

constexpr uint32_t MAX_AVAILABLE_CHIPS = 16;

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1,T2> &p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

} // end namespace tt
