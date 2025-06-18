// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

namespace abi {

inline char *__cxa_demangle(
        const char *mangled_name,
        char *output_buffer,
        size_t *length,
        int *status) {
    // dummy
    *status = 0;
    return nullptr;
}

} // namespace

