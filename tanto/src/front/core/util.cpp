// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <fstream>

#include "core/util.hpp"

namespace ronin {
namespace tanto {
namespace front {

bool read_file(const std::string &path, std::string &data) {
    try {
        std::ifstream stream(path, std::ios::binary);
        stream.seekg(0, std::ios::end);
        size_t size = stream.tellg();
        data.resize(size);
        stream.seekg(0);
        stream.read(data.data(), size);
    } catch (...) {
        return false;
    }
    return true;
}

size_t hash_combine(size_t h1, size_t h2) {
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
}

} // namespace front
} // namespace tanto
} // namespace ronin

