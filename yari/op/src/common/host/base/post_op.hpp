// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ronin {
namespace op {
namespace common {
namespace base {

enum class PostOp {
    NONE,
    CLIP,
    RELU
};

class PostOpSpec {
public:
    PostOpSpec();
    PostOpSpec(
        PostOp op,
        float alpha = 0.0f,
        float beta = 0.0f);
    PostOpSpec(const PostOpSpec &other) = default;
    ~PostOpSpec();
public:
    PostOpSpec &operator=(const PostOpSpec &other) = default;
    PostOp op() const {
        return m_op;
    }
    float alpha() const {
        return m_alpha;
    }
    float beta() const {
        return m_beta;
    }
    std::string str() const;
    float eval(float x) const;
private:
    PostOp m_op = PostOp::NONE;
    float m_alpha = 0.0f;
    float m_beta = 0.0f;
};

} // namespace base
} // namespace common
} // namespace op
} // namespace ronin

