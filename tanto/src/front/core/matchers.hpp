// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace ast_matchers;

//
//    Matcher factories
//

StatementMatcher make_func_call_0_matcher(const std::string &func_name);
StatementMatcher make_func_call_1_matcher(const std::string &func_name);
StatementMatcher make_func_call_2_matcher(const std::string &func_name);
StatementMatcher make_func_call_3_matcher(const std::string &func_name);

StatementMatcher make_member_call_0_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_1_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_2_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_3_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_4_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_5_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_6_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_7_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_8_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_9_matcher(
    const std::string &self_type,
    const std::string &method_name);

StatementMatcher make_member_call_1_with_t_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_2_with_t_matcher(
    const std::string &self_type,
    const std::string &method_name);
StatementMatcher make_member_call_4_with_t_matcher(
    const std::string &self_type,
    const std::string &method_name);

StatementMatcher make_member_call_3_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type);
StatementMatcher make_member_call_4_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type);
StatementMatcher make_member_call_6_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type);
StatementMatcher make_member_call_9_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type);

StatementMatcher make_member_call_4_with_t_dist_dram_matcher(
        const std::string &self_type,
        const std::string &method_name);
StatementMatcher make_member_call_5_with_t_dist_dram_matcher(
        const std::string &self_type,
        const std::string &method_name);

} // namespace front
} // namespace tanto
} // namespace ronin

