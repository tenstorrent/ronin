// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ronin {
namespace tanto {
namespace front {

bool read_file(const std::string &path, std::string &data);

size_t hash_combine(size_t h1, size_t h2);

} // namespace front
} // namespace tanto
} // namespace ronin

