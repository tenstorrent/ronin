// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "core/error.hpp"
#include "core/graph.hpp"
#include "core/graph_builder.hpp"
#include "core/graph_util.hpp"
#include "core/math_init_builtin.hpp"
#include "core/math_init_args.hpp"
#include "core/math_init_util.hpp"

namespace ronin {
namespace tanto {
namespace front {

class MathInitCall {
public:
    MathInitCall(MathInitFunc func, int arg_count);
    ~MathInitCall();
public:
    MathInitFunc func() {
        return m_func;
    }
    bool is_none() {
        return (m_func == MathInitFunc::NONE);
    }
    bool is_undef() {
        return (m_func == MathInitFunc::UNDEF);
    }
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
    void set_arg(
        int index, 
        int param,
        int value, 
        const std::string &code);
    bool any_undef_arg();
    bool equal(MathInitCall *other);
    std::string str();
    static MathInitCall *none() {
        return &m_none;
    }
    static MathInitCall *undef() {
        return &m_undef;
    }
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
    static MathInitCall m_none;
    static MathInitCall m_undef;
private:
    MathInitFunc m_func;
    int m_arg_count;
    Arg m_args[MAX_ARGS];
};

class MathInitState {
public:
    MathInitState();
    MathInitState(const MathInitState &other);
    ~MathInitState();
public:
    MathInitState &operator=(const MathInitState &other);
    void set(int index, MathInitCall *call) {
        assert(index >= 0 && index < COUNT);
        m_calls[index] = call;
    }
    MathInitCall *at(int index) const {
        assert(index >= 0 && index < COUNT);
        return m_calls[index];
    }
    bool is_empty() const;
public:
    static constexpr int COUNT = MathInitFuncGroup::COUNT;
private:
    MathInitCall *m_calls[COUNT];
};

class MathInitMask {
public:
    MathInitMask();
    MathInitMask(const MathInitMask &other);
    ~MathInitMask();
public:
    MathInitMask &operator=(const MathInitMask &other);
    void set(int index, bool flag) {
        assert(index >= 0 && index < COUNT);
        m_flags[index] = flag;
    }
    bool at(int index) const {
        assert(index >= 0 && index < COUNT);
        return m_flags[index];
    }
    bool is_empty() const;
public:
    static constexpr int COUNT = MathInitFuncGroup::COUNT;
private:
    bool m_flags[COUNT];
};

class MathInitPass {
public:
    MathInitPass();
    ~MathInitPass();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool run(const std::string &input_code, std::string &output_code);
private:
    void reset();
    bool pass1();
    bool create_orig_inits(StmtNode *stmt);
    MathInitCall *create_math_init_call_with_args(MathInitFunc func);
    MathInitCall *create_math_init_call(MathInitFunc func, int arg_count);
    bool pass2();
    MathInitState eval_func_init_state(FuncNode *func);
    MathInitState filter_func_init_state(FuncNode *func, const MathInitState &stmt_state);
    bool filter_func_init_call(
        FuncNode *func,
        MathInitCall *call,
        MathInitFuncUse *use);
    MathInitState eval_stmt_init_state(StmtNode *stmt);
    MathInitState eval_call_expr_init_state(StmtNode *stmt);
    MathInitCall *map_call_expr_args(MathInitCall *call, StmtNode *first_arg);
    void check_undef_init_funcs(MathInitState &state);
    void combine_init_states(MathInitState &state, const MathInitState &other);
    MathInitCall *combine_init_calls(MathInitCall *call1, MathInitCall *call2);
    bool pass3();
    MathInitMask make_top_init_mask(FuncNode *func);
    void eval_final_init_states(StmtNode *stmt, const MathInitMask &mask);
    void eval_final_init_states_children(StmtNode *stmt, const MathInitMask &mask);
    static bool may_have_final_init_state(StmtNode *stmt);
    void build_key_state_map();
    bool transform(const std::string &input_code, std::string &output_code);
    std::string make_init_call_insert(const MatchResult &result); 
    StmtKey make_init_call_insert_key(const MatchResult &result);
    static void translate_loc(
        const SourceManager &source_manager,
        const SourceLocation &loc,
        int &line,
        int &col);
    static std::string format_math_init_call(MathInitCall *call);
    const MathInitState &get_orig_state(StmtNode *stmt);
    const MathInitState &get_stmt_state(StmtNode *stmt);
    const MathInitState &get_func_state(FuncNode *func);
    const MathInitState &get_final_state(StmtNode *stmt);
    void enter_orig_state(StmtNode *stmt, const MathInitState &state);
    void enter_stmt_state(StmtNode *stmt, const MathInitState &state);
    void enter_func_state(FuncNode *func, const MathInitState &state);
    void enter_final_state(StmtNode *stmt, const MathInitState &state);
    void diag_dump_graph();
    void diag_dump_orig_funcs();
    std::string diag_annot_orig_stmt(StmtNode *stmt);
    void diag_dump_flow_funcs();
    std::string diag_annot_flow_func(FuncNode *func);
    std::string diag_annot_flow_stmt(StmtNode *stmt);
    void diag_dump_final_funcs();
    std::string diag_annot_final_stmt(StmtNode *stmt);
    std::string format_init_state(const MathInitState &state);
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    StmtGraphBuilder m_graph_builder;
    StmtGraphFuncSort m_graph_func_sort;
    MathInitBuiltinHandler m_builtin_handler;
    MathInitFuncHandler m_func_handler;
    MathInitArgsBuilder m_args_builder;
    MathInitFuncUseMapBuilder m_func_use_map_builder;
    MathInitCall *m_math_init_none;
    MathInitCall *m_math_init_undef;
    MathInitState m_empty_state;
    std::unique_ptr<StmtGraph> m_graph;
    MathInitFuncUseMap m_func_use_map;
    std::vector<std::unique_ptr<MathInitCall>> m_math_init_calls;
    std::unordered_map<StmtNode *, MathInitState> m_orig_state_map;
    std::unordered_map<StmtNode *, MathInitState> m_stmt_state_map;
    std::unordered_map<FuncNode *, MathInitState> m_func_state_map;
    std::unordered_map<StmtNode *, MathInitState> m_final_state_map;
    StmtKeyMap<MathInitState> m_key_state_map;
    bool m_visitor_ok;
};

} // namespace front
} // namespace tanto
} // namespace ronin

