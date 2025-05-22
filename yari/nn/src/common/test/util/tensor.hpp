// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

void read_tensor(const std::string &path, std::vector<float> &data);
void write_tensor(const std::string &path, const std::vector<float> &data);

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

