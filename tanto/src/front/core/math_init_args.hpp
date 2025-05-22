// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "core/error.hpp"
#include "core/graph.hpp"
#include "core/math_init_builtin.hpp"

namespace ronin {
namespace tanto {
namespace front {

struct MathInitArgConst {
    static constexpr int MAX_ARGS = 3;
    static constexpr int ARG_UNDEF = -1;
};

class MathInitArgsBuilder {
public:
    MathInitArgsBuilder();
    ~MathInitArgsBuilder();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool build(StmtNode *stmt, MathBuiltinId id, int group);
    int arg_count() {
        return m_arg_count;
    }
    int arg_param(int index) {
        assert(index >= 0 && index < m_arg_count);
        return m_args[index].param;
    }
    int arg_value(int index) {
        assert(index >= 0 && index < m_arg_count);
        return m_args[index].value;
    }
    std::string arg_code(int index) {
        assert(index >= 0 && index < m_arg_count);
        return m_args[index].code;
    }
private:
    void init();
    void set_arg_desc_pack(
            MathBuiltinId id, 
            int p0,
            int p1 = -1,
            int p2 = -1) {
        set_arg_desc(id, MathInitFuncGroup::PACK, p0, p1, p2);
    }
    void set_arg_desc_math(
            MathBuiltinId id, 
            int p0,
            int p1 = -1,
            int p2 = -1) {
        set_arg_desc(id, MathInitFuncGroup::MATH, p0, p1, p2);
    }
    void set_arg_desc_unpack(
            MathBuiltinId id, 
            int p0,
            int p1 = -1,
            int p2 = -1) {
        set_arg_desc(id, MathInitFuncGroup::UNPACK, p0, p1, p2);
    }
    void set_arg_desc(
        MathBuiltinId id, 
        int group,
        int p0,
        int p1,
        int p2);
    void reset();
    int get_arg_param(StmtNode *stmt);
    int get_arg_value(StmtNode *stmt);
    void error(const std::string &text);
private:
    static constexpr int MAX_ARGS = MathInitArgConst::MAX_ARGS;
    static constexpr int ARG_UNDEF = MathInitArgConst::ARG_UNDEF;
private:
    struct Arg {
        int param;
        int value;
        std::string code;
    };
private:
    int m_arg_desc[MathBuiltinIdCount][MathInitFuncGroup::COUNT][MAX_ARGS];
    ErrorHandler *m_error_handler;
    int m_arg_count;
    Arg m_args[MAX_ARGS];
};

} // namespace front
} // namespace tanto
} // namespace ronin

