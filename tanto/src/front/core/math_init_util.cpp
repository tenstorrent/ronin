// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "clang/AST/Stmt.h"

#include "core/graph.hpp"
#include "core/math_init_util.hpp"

namespace ronin {
namespace tanto {
namespace front {

//
//    MathInitFuncUse
//

MathInitFuncUse::MathInitFuncUse(int arg_count) { 
    m_args.resize(arg_count);
    for (int i = 0; i < arg_count; i++) {
        m_args[i].value = ARG_UNDEF;
        m_args[i].param = nullptr;
    }
}

MathInitFuncUse::~MathInitFuncUse() { }

//
//    MathInitFuncUseMap
//

MathInitFuncUseMap::MathInitFuncUseMap() { }

MathInitFuncUseMap::~MathInitFuncUseMap() { }

void MathInitFuncUseMap::reset() {
    m_use_map.clear();
    m_uses.clear();
}

MathInitFuncUse *MathInitFuncUseMap::enter(FuncNode *func) {
    MathInitFuncUse *use = new MathInitFuncUse(func->param_count());
    m_uses.emplace_back(use);
    m_use_map.emplace(func, use);
    return use;
}

MathInitFuncUse *MathInitFuncUseMap::find(FuncNode *func) {
    auto it = m_use_map.find(func);
    return (it != m_use_map.end()) ? it->second : nullptr;
}

//
//    MathInitFuncUseMapBuilder
//

MathInitFuncUseMapBuilder::MathInitFuncUseMapBuilder():
        m_error_handler(nullptr),
        m_use_map(nullptr),
        m_visitor_ok(false) { }

MathInitFuncUseMapBuilder::~MathInitFuncUseMapBuilder() { }

void MathInitFuncUseMapBuilder::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
}

bool MathInitFuncUseMapBuilder::run(StmtGraph *graph, MathInitFuncUseMap &use_map) {
    m_use_map = &use_map;
    m_use_map->reset();
    m_visitor_ok = true;
    CallbackStmtGraphVisitor visitor;
    visitor.set_on_start_stmt(
        [this](StmtNode *stmt) {
            if (stmt->stmt_class() == Stmt::CallExprClass) {
                if (!add_call(stmt)) {
                    m_visitor_ok = false;
                }
            }
        });
    visitor.visit(graph);
    m_use_map = nullptr;
    return m_visitor_ok;
}

bool MathInitFuncUseMapBuilder::add_call(StmtNode *call) {
    StmtNode *child = call->first_child();
    FuncNode *func = child->func_ref();
    if (func == nullptr || func->top_stmt() == nullptr) {
        return true;
    }
    child = child->next();
    MathInitFuncUse *use = m_use_map->find(func);
    if (use == nullptr) {
        return enter_use(func, child);
    } else {
        return update_use(use, child);
    }
}

bool MathInitFuncUseMapBuilder::enter_use(FuncNode *func, StmtNode *first_arg) {
    MathInitFuncUse *use = m_use_map->enter(func);
    int param_count = use->arg_count();
    int index = 0;
    for (StmtNode *arg = first_arg; arg != nullptr; arg = arg->next()) {
        if (index >= param_count) {
            error("Too many arguments in function call");
            return false;
        }
        int value = get_arg_value(arg);
        VarNode *param = get_arg_param(arg);
        use->set_arg_value(index, value);
        use->set_arg_param(index, param);
        index++;
    }
    if (index != param_count) {
        error("Too few arguments in function call");
    }
    return true;
}

bool MathInitFuncUseMapBuilder::update_use(MathInitFuncUse *use, StmtNode *first_arg) {
    int param_count = use->arg_count();
    int index = 0;
    for (StmtNode *arg = first_arg; arg != nullptr; arg = arg->next()) {
        if (index >= param_count) {
            error("Too many arguments in function call");
            return false;
        }
        int value = get_arg_value(arg);
        VarNode *param = get_arg_param(arg);
        if (use->arg_value(index) != value) {
            use->set_arg_value(index, MathInitFuncUse::ARG_UNDEF);
        }
        if (use->arg_param(index) != param) {
            use->set_arg_param(index, nullptr);
        }
        index++;
    }
    if (index != param_count) {
        error("Too few arguments in function call");
    }
    return true;
}

int MathInitFuncUseMapBuilder::get_arg_value(StmtNode *stmt) {
    if (stmt->is_bool_literal()) {
        return stmt->bool_literal() ? 1 : 0;
    }
    if (stmt->is_int_literal()) {
        return stmt->int_literal();
    }
    return MathInitFuncUse::ARG_UNDEF;
}

VarNode *MathInitFuncUseMapBuilder::get_arg_param(StmtNode *stmt) {
    VarNode *param = stmt->decl_ref();
    if (param == nullptr || param->param_index() < 0) {
        return nullptr;
    }
    return param;
}

void MathInitFuncUseMapBuilder::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

