// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <algorithm>

#include "host/base/post_op.hpp"

namespace ronin {
namespace op {
namespace common {
namespace base {

namespace {

std::string str_param(float alpha, float beta) {
    return "(" + std::to_string(alpha) + ", " + std::to_string(beta) + ")";
}

} // namespace

//
//    PostOpSpec
//

PostOpSpec::PostOpSpec() { }

PostOpSpec::PostOpSpec(
        PostOp op,
        float alpha/* = 0.0f*/,
        float beta/* = 0.0f*/):
            m_op(op),
            m_alpha(alpha),
            m_beta(beta) { }

PostOpSpec::~PostOpSpec() { }

std::string PostOpSpec::str() const {
    switch (m_op) {
    case PostOp::NONE:
        return "none";
    case PostOp::CLIP:
        return "clip" + str_param(m_alpha, m_beta);
    case PostOp::RELU:
        return "relu";
    default:
        assert(false);
        return "<invalid>";
    }
}

float PostOpSpec::eval(float x) const {
    switch (m_op) {
    case PostOp::NONE:
        return x;
    case PostOp::CLIP:
        return std::min(std::max(x, m_alpha), m_beta);
    case PostOp::RELU:
        return std::max(x, 0.0f);
    default:
        assert(false);
        return x;
    }
}

} // namespace base
} // namespace common
} // namespace op
} // namespace ronin

