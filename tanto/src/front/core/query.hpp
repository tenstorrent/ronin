// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <utility>

#include "core/common.hpp"
#include "core/error.hpp"
#include "core/tooling.hpp"

namespace ronin {
namespace tanto {
namespace front {

class Query {
public:
    Query();
    ~Query();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool run(const std::string &input_code);
    int kernel_param_count() {
        return int(m_kernel_params.size());
    }
    std::string kernel_param_name(int index) {
        return m_kernel_params[index].first;
    }
    DataType kernel_param_type(int index) {
        return m_kernel_params[index].second;
    }
private:
    bool query_kernel_params();
    bool traverse_stmts();
    bool test_custom();
    bool run_tests();
    template<typename M>
    bool eval_match(M m, RangeSelector selector) {
        auto results = m_matcher_tool.match(m);
        return print_match_results(results, selector);
    }
    bool print_match_results( 
        const std::vector<MatchResult> &results,
        RangeSelector selector);
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    MatcherTool m_matcher_tool;
    std::vector<std::pair<std::string, DataType>> m_kernel_params;
};

} // namespace front
} // namespace tanto
} // namespace ronin

