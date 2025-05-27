// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <string>
#include <vector>
#include <cstdint>

#include "clang/AST/Stmt.h"

#include "core/error.hpp"
#include "core/graph.hpp"
#include "core/math_init_builtin.hpp"
#include "core/math_init_args.hpp"

namespace ronin {
namespace tanto {
namespace front {

//
//    MathInitArgsBuilder
//

MathInitArgsBuilder::MathInitArgsBuilder():
        m_error_handler(nullptr),
        m_arg_count(0) {
    init();
    for (int i = 0; i < MAX_ARGS; i++) {
        m_args[i].param = ARG_UNDEF;
        m_args[i].value = ARG_UNDEF;
    }
}

MathInitArgsBuilder::~MathInitArgsBuilder() { }

void MathInitArgsBuilder::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
}

bool MathInitArgsBuilder::build(StmtNode *stmt, MathBuiltinId id, int group) {
    reset();
    int index = int(id);
    assert(index >= 0 && index < MathBuiltinIdCount);
    assert(group >= 0 && group < MathInitFuncGroup::COUNT);
    int *desc = m_arg_desc[index][group];
    if (desc[0] == -1) {
        return true;
    }
    std::vector<StmtNode *> stmt_args;
    StmtNode *child = stmt->first_child();
    if (child == nullptr) {
        error("Missing callee of member call");
        return false;
    }
    for (child = child->next(); child != nullptr; child = child->next()) {
        stmt_args.push_back(child);
    }
    int stmt_arg_count = int(stmt_args.size());
    for (int i = 0; i < MAX_ARGS; i++) {
        int pos = desc[i];
        if (pos < 0) {
            break;
        }
        if (pos >= stmt_arg_count) {
            error("Arguments do not match argument descriptor");
            return false;
        }
        StmtNode *arg = stmt_args[pos];
        m_args[i].param = get_arg_param(arg);
        m_args[i].value = get_arg_value(arg);
        m_args[i].code = arg->code();
        m_arg_count++;
    }
    return true;
}

void MathInitArgsBuilder::init() {
    for (int i = 0; i < MathBuiltinIdCount; i++) {
        for (int j = 0; j < MathInitFuncGroup::COUNT; j++) {
            for (int k = 0; k < MAX_ARGS; k++) {
                m_arg_desc[i][j][k] = -1;
            }
        }
    }
    // unpack
    set_arg_desc_unpack(MathBuiltinId::COPY, 0);
    set_arg_desc_unpack(MathBuiltinId::ADD, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::SUB, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::MUL, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::ADD_BCAST_ROWS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::SUB_BCAST_ROWS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::MUL_BCAST_ROWS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::ADD_BCAST_COLS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::SUB_BCAST_COLS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::MUL_BCAST_COLS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::ADD_BCAST_SCALAR, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::SUB_BCAST_SCALAR, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::MUL_BCAST_SCALAR, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::MATMUL, 0, 1, 5);
    set_arg_desc_unpack(MathBuiltinId::REDUCE_MAX_ROWS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::REDUCE_MAX_COLS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::REDUCE_MAX_SCALAR, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::REDUCE_SUM_ROWS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::REDUCE_SUM_COLS, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::REDUCE_SUM_SCALAR, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::TRANSPOSE, 0);
    set_arg_desc_unpack(MathBuiltinId::TILIZE_BLOCK, 0, 1);
    set_arg_desc_unpack(MathBuiltinId::UNTILIZE_BLOCK, 0);
    // math
    set_arg_desc_math(MathBuiltinId::MATMUL, 5);
    // pack
    set_arg_desc_pack(MathBuiltinId::PACK, 1);
    set_arg_desc_pack(MathBuiltinId::PACK_ROW, 1);
    set_arg_desc_pack(MathBuiltinId::PACK_COL, 1);
    set_arg_desc_pack(MathBuiltinId::PACK_SCALAR, 1);
    set_arg_desc_pack(MathBuiltinId::TILIZE_BLOCK, 2);
    set_arg_desc_pack(MathBuiltinId::UNTILIZE_BLOCK, 2);
}

void MathInitArgsBuilder::set_arg_desc(
        MathBuiltinId id, 
        int group,
        int p0,
        int p1,
        int p2) {
    int index = int(id);
    assert(index >= 0 && index < MathInitFuncCount);
    assert(group >= 0 && group < MathInitFuncGroup::COUNT);
    m_arg_desc[index][group][0] = p0;
    m_arg_desc[index][group][1] = p1;
    m_arg_desc[index][group][2] = p2;
}

void MathInitArgsBuilder::reset() {
    m_arg_count = 0;
    for (int i = 0; i < MAX_ARGS; i++) {
        m_args[i].param = ARG_UNDEF;
        m_args[i].value = ARG_UNDEF;
        m_args[i].code.clear();
    }
}

int MathInitArgsBuilder::get_arg_param(StmtNode *stmt) {
    VarNode *decl_ref = stmt->decl_ref();
    if (decl_ref != nullptr) {
        int param = decl_ref->param_index();
        // only parameters of parent function are recognized
        if (param >= 0) {
            return param;
        }
    }
    return ARG_UNDEF;
}

int MathInitArgsBuilder::get_arg_value(StmtNode *stmt) {
    if (stmt->is_bool_literal()) {
        return stmt->bool_literal() ? 1 : 0;
    }
    if (stmt->is_int_literal()) {
        int value = stmt->int_literal();
        // integer arguments of init functions are always non-negative
        if (value >= 0) {
            return value;
        }
    }
    return ARG_UNDEF;
}

void MathInitArgsBuilder::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

