// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <cassert>
#include <vector>
#include <string>

#include "arhat/runtime/arhat.hpp"

#include "test/util/tensor.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

void read_tensor(const std::string &path, std::vector<float> &data) {
    arhat::Tensor tensor;
    tensor.Read(path);
    assert(tensor.Type() == arhat::Tensor::Dtype::Float);
    int volume = tensor.Volume();
    data.resize(volume);
    memcpy(data.data(), tensor.Data(), volume * sizeof(float));
}

void write_tensor(const std::string &path, const std::vector<float> &data) {
    arhat::Tensor tensor;
    int volume = int(data.size());
    int shape[1] = {volume};
    tensor.Reset(arhat::Tensor::Dtype::Float, 1, shape);
    memcpy(tensor.Data(), data.data(), volume * sizeof(float));
    tensor.Write(path);
}

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

