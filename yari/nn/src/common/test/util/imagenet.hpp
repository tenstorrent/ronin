// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

void print_imagenet_classes(std::vector<float> &prob, int batch_size, int topk);

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

