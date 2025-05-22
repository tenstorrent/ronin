// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "clang/Tooling/Transformer/RewriteRule.h"

#include "core/error.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace transformer;

class DeadCodePass {
public:
    DeadCodePass();
    ~DeadCodePass();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool run(const std::string &input_code, std::string &output_code);
private:
    bool rewrite(
        RewriteRule rule,
        const std::string &input_code, 
        std::string &output_code);
    void create_rules();
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    RewriteRule m_pass1_rule;
    RewriteRule m_pass2_rule;
};

} // namespace front
} // namespace tanto
} // namespace ronin

