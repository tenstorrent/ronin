// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "core/error.hpp"
#include "core/graph.hpp"

namespace ronin {
namespace tanto {
namespace front {

class MathInitFuncUse {
public:
    MathInitFuncUse(int arg_count);
    ~MathInitFuncUse();
public:
    int arg_count() {
        return int(m_args.size());
    }
    void set_arg_value(int index, int value) {
        m_args[index].value = value;
    }
    int arg_value(int index) {
        return m_args[index].value;
    }
    void set_arg_param(int index, VarNode *param) {
        m_args[index].param = param;
    }
    VarNode *arg_param(int index) {
        return m_args[index].param;
    }
    void set_arg_undef(int index) {
        Arg &arg = m_args[index];
        arg.value = ARG_UNDEF;
        arg.param = nullptr;
    }
    bool is_arg_undef(int index) {
        Arg &arg = m_args[index];
        return (arg.value == ARG_UNDEF && arg.param == nullptr);
    }
public:
    static constexpr int ARG_UNDEF = -1;
private:
    struct Arg {
        int value;
        VarNode *param;
    };
private:
    std::vector<Arg> m_args;
};

class MathInitFuncUseMap {
public:
    MathInitFuncUseMap();
    ~MathInitFuncUseMap();
public:
    void reset();
    MathInitFuncUse *enter(FuncNode *func);
    MathInitFuncUse *find(FuncNode *func);
private:
    std::unordered_map<FuncNode *, MathInitFuncUse *> m_use_map;
    std::vector<std::unique_ptr<MathInitFuncUse>> m_uses;
};

class MathInitFuncUseMapBuilder {
public:
    MathInitFuncUseMapBuilder();
    ~MathInitFuncUseMapBuilder();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool run(StmtGraph *graph, MathInitFuncUseMap &use_map);
private:
    bool add_call(StmtNode *call);
    bool enter_use(FuncNode *func, StmtNode *first_arg);
    bool update_use(MathInitFuncUse *use, StmtNode *first_arg);
    static int get_arg_value(StmtNode *stmt);
    static VarNode *get_arg_param(StmtNode *stmt);
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    MathInitFuncUseMap *m_use_map;
    bool m_visitor_ok;
};

} // namespace front
} // namespace tanto
} // namespace ronin

