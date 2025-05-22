// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "test/util/tiles.hpp"

#include "test/ref/unpack_untilize_ref.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

//
//    UnpackUntilizeRef
//

UnpackUntilizeRef::UnpackUntilizeRef() { }

UnpackUntilizeRef::~UnpackUntilizeRef() { }

void UnpackUntilizeRef::init(int H, int W) {
    assert(H % 32 == 0);
    assert(W % 32 == 0);
    m_H = H;
    m_W = W;
}

void UnpackUntilizeRef::run(const float *x, float *y) {
    util::untilize(x, y, m_H, m_W);
}

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

