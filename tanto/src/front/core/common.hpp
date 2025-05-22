// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace tanto {
namespace front {

enum class FrontendMode {
    UNDEF,
    COMPUTE,
    READ,
    WRITE
};

enum class DataType {
    INT32,
    UINT32,
    FLOAT,
    BFLOAT16,
    GLOBAL,
    LOCAL,
    SEMAPHORE,
    PIPE
};

} // namespace front
} // namespace tanto
} // namespace ronin

