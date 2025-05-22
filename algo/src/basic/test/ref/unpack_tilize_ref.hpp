// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

class UnpackTilizeRef {
public:
    UnpackTilizeRef();
    ~UnpackTilizeRef();
public:
    void init(int H, int W);
    void run(const float *x, float *y);
private:
    int m_H = 0;
    int m_W = 0;
};

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

