// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>
#include <utility>

#include "riscv/builtin_compute_tanto.hpp"

namespace {

#define DECL_BUILTIN(name, count) \
    {ComputeTantoBuiltinId::name, {#name, count}},

std::unordered_map<ComputeTantoBuiltinId, std::pair<std::string, int>> 
    compute_tanto_builtin_map = {
COMPUTE_TANTO_BUILTINS
};

#undef DECL_BUILTIN

} // namespace

std::unordered_map<ComputeTantoBuiltinId, std::pair<std::string, int>> &
        get_compute_tanto_builtin_map() {
    return compute_tanto_builtin_map;
}

