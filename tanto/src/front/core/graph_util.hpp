// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <unordered_set>

#include "core/error.hpp"
#include "core/graph.hpp"

namespace ronin {
namespace tanto {
namespace front {

class StmtGraphFuncSort {
public:
    StmtGraphFuncSort();
    ~StmtGraphFuncSort();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool run(StmtGraph *graph, std::vector<FuncNode *> &result);
public:
    void reset();
    bool collect_calls(StmtNode *stmt);
    bool enter_func(FuncNode *func);
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    std::unordered_set<FuncNode *> m_func_set;
    std::vector<FuncNode *> m_result;
};

} // namespace front
} // namespace tanto
} // namespace ronin

