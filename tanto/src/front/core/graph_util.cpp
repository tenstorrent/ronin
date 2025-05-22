// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>
#include <utility>

#include "clang/AST/Stmt.h"

#include "core/error.hpp"
#include "core/graph.hpp"
#include "core/graph_util.hpp"

namespace ronin {
namespace tanto {
namespace front {

//
//    StmtGraphFuncSort
//

StmtGraphFuncSort::StmtGraphFuncSort():
        m_error_handler(nullptr) { }

StmtGraphFuncSort::~StmtGraphFuncSort() { }

void StmtGraphFuncSort::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
}

bool StmtGraphFuncSort::run(StmtGraph *graph, std::vector<FuncNode *> &result) {
    reset();
    FuncNode *main = graph->main_func();
    if (main == nullptr) {
        error("Missing main function");
        return false;
    }
    if (!enter_func(main)) {
        return false;
    }
    for (int i = 0; i < int(m_result.size()); i++) {
        FuncNode *func = m_result[i];
        StmtNode *top = func->top_stmt();
        assert(top != nullptr);
        if (!collect_calls(top)) {
            return false;
        }
    }
    result = std::move(m_result);
    return true;
}

void StmtGraphFuncSort::reset() {
    m_func_set.clear();
    m_result.clear();
}

bool StmtGraphFuncSort::collect_calls(StmtNode *stmt) {
    if (stmt->stmt_class() == Stmt::StmtClass::CallExprClass) {
        StmtNode *callee = stmt->first_child();
        assert(callee != nullptr);
        FuncNode *func = callee->func_ref();
        if (func == nullptr) {
            return true;
        }
        if (func->top_stmt() == nullptr) {
            return true;
        }
        if (!enter_func(func)) {
            return false;
        }
    }
    for (StmtNode *child = stmt->first_child(); child != nullptr; child = child->next()) {
        if (!collect_calls(child)) {
            return false;
        }
    }
    return true;
}

bool StmtGraphFuncSort::enter_func(FuncNode *func) {
    if (m_func_set.count(func)) {
        // Cannot detect recursion with this approach
        // If detecting recursion is critical, collect caller/callee pairs
        // and perform regular topological sort
#if 0 // Patch 05.02.2025
        error("Recursive function calls are not allowed");
        return false;
#endif
        return true;
    }
    m_func_set.insert(func);
    m_result.push_back(func);
    return true;
}

void StmtGraphFuncSort::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

