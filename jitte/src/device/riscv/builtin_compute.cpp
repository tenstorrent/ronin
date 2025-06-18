// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>
#include <utility>

#include "riscv/builtin_compute.hpp"

namespace {

#define DECL_BUILTIN(name, count) \
    {ComputeBuiltinId::name, {#name, count}},

std::unordered_map<ComputeBuiltinId, std::pair<std::string, int>> compute_builtin_map = {
COMPUTE_BUILTINS
};

#undef DECL_BUILTIN

} // namespace

std::unordered_map<ComputeBuiltinId, std::pair<std::string, int>> &get_compute_builtin_map() {
    return compute_builtin_map;
}

