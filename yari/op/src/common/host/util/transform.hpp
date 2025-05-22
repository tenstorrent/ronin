// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace ronin {
namespace op {
namespace common {
namespace util {

std::vector<uint16_t> float_to_u16b(const std::vector<float> &x);
std::vector<float> u16b_to_float(const std::vector<uint16_t> &x);
std::vector<float> tilize(const std::vector<float> &x, int H, int W);
std::vector<float> untilize(const std::vector<float> &x, int H, int W);
std::vector<float> make_faces(const std::vector<float> &x);
std::vector<float> make_tiles(const std::vector<float> &x);
std::vector<float> pad_hw(
    const std::vector<float> &x, 
    int N, 
    int H, 
    int W,
    int C);
std::vector<float> unpad_hw(
    const std::vector<float> &x, 
    int N, 
    int H, 
    int W,
    int C);
std::vector<float> pad(const std::vector<float> &x, int nx0, int ny0);
std::vector<float> pad(
    const std::vector<float> &x,
    int nx0,
    int nx1,
    int ny0,
    int ny1);
std::vector<float> pad(
    const std::vector<float> &x,
    int nx0,
    int nx1,
    int nx2,
    int ny0,
    int ny1,
    int ny2);
std::vector<float> pad(
    const std::vector<float> &x,
    int nx0,
    int nx1,
    int nx2,
    int nx3,
    int ny0,
    int ny1,
    int ny2, 
    int ny3);
std::vector<float> unpad(const std::vector<float> &x, int nx0, int ny0);
std::vector<float> unpad(
    const std::vector<float> &x,
    int nx0,
    int nx1,
    int ny0,
    int ny1);
std::vector<float> unpad(
    const std::vector<float> &x,
    int nx0,
    int nx1,
    int nx2,
    int ny0,
    int ny1,
    int ny2);
std::vector<float> unpad(
    const std::vector<float> &x,
    int nx0,
    int nx1,
    int nx2,
    int nx3,
    int ny0,
    int ny1,
    int ny2, 
    int ny3);

} // namespace util
} // namespace common
} // namespace op
} // namespace ronin

