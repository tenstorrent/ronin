// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "clang/Tooling/Transformer/RewriteRule.h"

#include "core/error.hpp"
#include "core/rules.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace transformer;

class Transform {
public:
    Transform();
    ~Transform();
public:
    void set_error_handler(ErrorHandler *error_handler);
    void reset();
    bool add_param(uint32_t index, uint32_t value);
    bool pass1(const std::string &input_code, std::string &output_code);
    bool pass2_compute(const std::string &input_code, std::string &output_code);
    bool pass2_dataflow(
        const std::string &input_code, 
        std::string &output_code,
        bool write_mode);
private:
    bool rewrite(
        RewriteRule rule,
        const std::string &input_code, 
        std::string &output_code);
    RewriteRule make_pass1_rule();
    RewriteRule make_pass2_compute_rule();
    RewriteRule make_pass2_dataflow_rule(bool write_mode);
    uint32_t get_param_value();
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    std::unordered_map<uint32_t, uint32_t> m_param_map;
    uint32_t m_next_param_index;
    RuleFactory m_rule_factory;
    RewriteRule m_pass1_rule;
    RewriteRule m_pass2_compute_rule;
    RewriteRule m_pass2_read_rule;
    RewriteRule m_pass2_write_rule;
    bool m_rewrite_ok;
};

} // namespace front
} // namespace tanto
} // namespace ronin

