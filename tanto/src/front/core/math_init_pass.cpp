// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <string>
#include <functional>

#include "clang/AST/Stmt.h"

#include "core/error.hpp"
#include "core/graph.hpp"
#include "core/tooling.hpp"
#include "core/math_init_pass.hpp"

namespace ronin {
namespace tanto {
namespace front {

constexpr bool DIAG_DUMP_ENABLE = false;

//
//    MathInitCall
//

MathInitCall MathInitCall::m_none(MathInitFunc::NONE, 0);
MathInitCall MathInitCall::m_undef(MathInitFunc::UNDEF, 0);

MathInitCall::MathInitCall(MathInitFunc func, int arg_count):
        m_func(func), m_arg_count(arg_count) { 
    assert(arg_count <= MAX_ARGS);
    for (int i = 0; i < MAX_ARGS; i++) {
        m_args[i].param = ARG_UNDEF;
        m_args[i].value = ARG_UNDEF;
    }
}

MathInitCall::~MathInitCall() { }

void MathInitCall::set_arg(
        int index, 
        int param,
        int value, 
        const std::string &code) {
    assert(index >= 0 && index < m_arg_count);
    m_args[index].param = param;
    m_args[index].value = value;
    m_args[index].code = code;
}

bool MathInitCall::any_undef_arg() {
    for (int i = 0; i < m_arg_count; i++) {
        if (m_args[i].param == ARG_UNDEF && m_args[i].value == ARG_UNDEF) {
            return true;
        }
    }
    return false;
}

bool MathInitCall::equal(MathInitCall *other) {
    if (m_func != other->m_func) {
        return false;
    }
    assert(m_arg_count == other->m_arg_count);
    for (int i = 0; i < m_arg_count; i++) {
        int param1 = m_args[i].param;
        int param2 = other->m_args[i].param;
        int value1 = m_args[i].value;
        int value2 = other->m_args[i].value;
        if (param1 != ARG_UNDEF || param2 != ARG_UNDEF) {
#if 0 // TODO: Revise this (fixed 08.11.2024)
            if (value1 != value2) {
#endif
            if (param1 != param2) {
                return false;
            }
        } else if (value1 != ARG_UNDEF || value2 != ARG_UNDEF) {
            if (value1 != value2) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

std::string MathInitCall::str() {
    // string representation for diagnostic purposes
    std::string result = get_math_init_func_name(m_func);
    if (m_func == MathInitFunc::NONE || m_func == MathInitFunc::UNDEF) {
        return result;
    }
    result += "(";
    for (int i = 0; i < m_arg_count; i++) {
        if (i != 0) {
            result += ", ";
        }
        result += m_args[i].code;
    }
    result += ")";
    return result;
}

//
//    MathInitState
//

MathInitState::MathInitState() {
    for (int i = 0; i < COUNT; i++) {
        m_calls[i] = MathInitCall::none();
    }
}

MathInitState::MathInitState(const MathInitState &other) {
    for (int i = 0; i < COUNT; i++) {
        m_calls[i] = other.m_calls[i];
    }
}

MathInitState::~MathInitState() { }

MathInitState &MathInitState::operator=(const MathInitState &other) {
    for (int i = 0; i < COUNT; i++) {
        m_calls[i] = other.m_calls[i];
    }
    return *this;
}

bool MathInitState::is_empty() const {
    for (int i = 0; i < COUNT; i++) {
        if (!m_calls[i]->is_none()) {
            return false;
        }
    }
    return true;
}

//
//    MathInitMask
//

MathInitMask::MathInitMask() {
    for (int i = 0; i < COUNT; i++) {
        m_flags[i] = true;
    }
}

MathInitMask::MathInitMask(const MathInitMask &other) {
    for (int i = 0; i < COUNT; i++) {
        m_flags[i] = other.m_flags[i];
    }
}

MathInitMask::~MathInitMask() { }

MathInitMask &MathInitMask::operator=(const MathInitMask &other) {
    for (int i = 0; i < COUNT; i++) {
        m_flags[i] = other.m_flags[i];
    }
    return *this;
}

bool MathInitMask::is_empty() const {
    for (int i = 0; i < COUNT; i++) {
        if (m_flags[i]) {
            return false;
        }
    }
    return true;
}

//
//    MathInitPass
//

MathInitPass::MathInitPass():
        m_error_handler(nullptr),
        m_math_init_none(MathInitCall::none()),
        m_math_init_undef(MathInitCall::undef()),
        m_visitor_ok(false) { }

MathInitPass::~MathInitPass() { }

void MathInitPass::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
    m_graph_builder.set_error_handler(error_handler);
    m_graph_func_sort.set_error_handler(error_handler);
    m_args_builder.set_error_handler(error_handler);
    m_func_use_map_builder.set_error_handler(error_handler);
}

bool MathInitPass::run(const std::string &input_code, std::string &output_code) {
    reset();
    if (!m_graph_builder.build(input_code, m_graph)) {
        return false;
    }
    diag_dump_graph();
    if (!pass1()) {
        return false;
    }
    diag_dump_orig_funcs();
    if (!pass2()) {
        return false;
    }
    diag_dump_flow_funcs();
    if (!pass3()) {
        return false;
    }
    diag_dump_final_funcs();
    build_key_state_map();
    if (!transform(input_code, output_code)) {
        return false;
    }
    return true;
}

void MathInitPass::reset() {
    m_graph.reset();
    m_func_use_map.reset();
    m_orig_state_map.clear();
    m_stmt_state_map.clear();
    m_func_state_map.clear();
    m_final_state_map.clear();
    m_key_state_map.clear();
    m_math_init_calls.clear();
}

bool MathInitPass::pass1() {
    // Pass 1: Create original init calls
    m_visitor_ok = true;
    CallbackStmtGraphVisitor visitor;
    visitor.set_on_start_stmt(
        [this](StmtNode *stmt) {
            if (!create_orig_inits(stmt)) {
                m_visitor_ok = false;
            }
        });
    visitor.visit(m_graph.get());
    return m_visitor_ok;
}

bool MathInitPass::create_orig_inits(StmtNode *stmt) {
    std::string type_name;
    std::string method_name;
    Stmt::StmtClass stmt_class = stmt->stmt_class();
    if (stmt_class == Stmt::StmtClass::CallExprClass) {
        StmtNode *callee = stmt->first_child();
        if (callee == nullptr) {
            error("Missing callee of function call");
            return false;
        }
        FuncNode *func_ref = callee->func_ref();
        if (func_ref == nullptr) {
            // is this legal?
            return true;
        }
        method_name = func_ref->name();
        type_name = "";
    } else if (stmt_class == Stmt::StmtClass::CXXMemberCallExprClass) {
        StmtNode *callee = stmt->first_child();
        if (callee == nullptr) {
            error("Missing callee of member call");
            return false;
        }
        method_name = callee->member_name();
        if (method_name.empty()) {
            error("Missing member name of member call");
            return false;
        }
        StmtNode *self = callee->first_child();
        if (self == nullptr) {
            error("Missing object of member call");
            return false;
        }
        type_name = self->type_name();
    } else {
        return true;
    }
    MathBuiltinId id = m_builtin_handler.map(type_name, method_name);
    if (id == MathBuiltinId::NONE) {
        return true;
    }
    MathInitState state;
    for (int i = 0; i < MathInitState::COUNT; i++) {
        MathInitFunc func = m_func_handler.map(id, i);
        if (func != MathInitFunc::NONE) {
            if (!m_args_builder.build(stmt, id, i)) {
                return false;
            }
            MathInitCall *call = create_math_init_call_with_args(func);
            state.set(i, call);
        }
    }
    enter_orig_state(stmt, state);
    return true;
}

MathInitCall *MathInitPass::create_math_init_call_with_args(MathInitFunc func) {
    int arg_count = m_args_builder.arg_count();
    MathInitCall *call = create_math_init_call(func, arg_count);
    for (int i = 0; i < arg_count; i++) {
        int param = m_args_builder.arg_param(i);
        int value = m_args_builder.arg_value(i);
        std::string code = m_args_builder.arg_code(i);
        call->set_arg(i, param, value, code);
    }
    return call;
}

MathInitCall *MathInitPass::create_math_init_call(MathInitFunc func, int arg_count) {
    MathInitCall *call = new MathInitCall(func, arg_count);
    m_math_init_calls.emplace_back(call);
    return call;
}

bool MathInitPass::pass2() {
    // Pass 2: Propagate init calls to functions and statements
    if (!m_func_use_map_builder.run(m_graph.get(), m_func_use_map)) {
        return false;
    }
    std::vector<FuncNode *> sorted_funcs;
    if (!m_graph_func_sort.run(m_graph.get(), sorted_funcs)) {
        return false;
    }
    int func_count = int(sorted_funcs.size());
    for (int i = func_count - 1; i >= 0; i--) {
        FuncNode *func = sorted_funcs[i];
        MathInitState state = eval_func_init_state(func);
        enter_func_state(func, state);
    }
    return true;
}

MathInitState MathInitPass::eval_func_init_state(FuncNode *func) {
    MathInitState state;
    StmtNode *top = func->top_stmt();
    if (top != nullptr) {
        state = eval_stmt_init_state(top);
#if 0 // TODO: Revise this [fixed 08.11.2024]
        enter_stmt_state(top, state);
#endif
        state = filter_func_init_state(func, state);
    }
    return state;
}

MathInitState MathInitPass::filter_func_init_state(
        FuncNode *func, const MathInitState &stmt_state) {
    MathInitState func_state;
    MathInitFuncUse *use = m_func_use_map.find(func);
    if (use == nullptr) {
        return func_state;
    }
    for (int i = 0; i < MathInitState::COUNT; i++) {
        MathInitCall *call = stmt_state.at(i);
        if (filter_func_init_call(func, call, use)) {
            func_state.set(i, call);
        }
    }
    return func_state;
}

bool MathInitPass::filter_func_init_call(
        FuncNode *func,
        MathInitCall *call,
        MathInitFuncUse *use) {
    if (call->is_none() || call->is_undef()) {
        return true;
    }
    int arg_count = call->arg_count();
    for (int i = 0; i < arg_count; i++) {
        int param = call->arg_param(i);
        if (param == MathInitArgConst::ARG_UNDEF) {
            continue;
        }
        if (use->arg_param(param) == nullptr) {
            return false;
        }
    }
    return true;
}

#if 0 // TODO: Revise this [fixed 08.11.2024]
MathInitState MathInitPass::eval_stmt_init_state(StmtNode *stmt) {
    MathInitState state;
    const MathInitState &orig_state = get_orig_state(stmt);
    if (!orig_state.is_empty()) {
        // no need to check children
        state = orig_state;
        check_undef_init_funcs(state);
        return state;
    }
    if (stmt->stmt_class() == Stmt::StmtClass::CallExprClass) {
        return eval_call_expr_init_state(stmt);
    }
    StmtNode *child = stmt->first_child();
    if (child == nullptr) {
        return state;
    }
    for ( ; child != nullptr; child = child->next()) {
        MathInitState child_state = eval_stmt_init_state(child);
        enter_stmt_state(child, child_state);
        combine_init_states(state, child_state);
    }
    return state;
}
#endif

MathInitState MathInitPass::eval_stmt_init_state(StmtNode *stmt) {
    MathInitState state;
    const MathInitState &orig_state = get_orig_state(stmt);
    if (!orig_state.is_empty()) {
        // no need to check children
        // always retain original state for this statement but
        // don't propagate upstream init funcs with undefined arguments
        enter_stmt_state(stmt, orig_state);
        state = orig_state;
        check_undef_init_funcs(state);
        return state;
    }
    if (stmt->stmt_class() == Stmt::StmtClass::CallExprClass) {
        state = eval_call_expr_init_state(stmt);
    } else {
        for (StmtNode *child = stmt->first_child(); child != nullptr; child = child->next()) {
            MathInitState child_state = eval_stmt_init_state(child);
            combine_init_states(state, child_state);
        }
    }
    enter_stmt_state(stmt, state);
    return state;
}

MathInitState MathInitPass::eval_call_expr_init_state(StmtNode *stmt) {
    // ACHTUNG: Nested calls are not supported
    //     Must preprocess source code to replace nested calls
    //     with sequences of calls and results in temporary variables
    MathInitState state;
    StmtNode *child = stmt->first_child();
    if (child == nullptr) {
        // must not happen
        return state;
    }
    FuncNode *func = child->func_ref();
    if (func == nullptr) {
        return state;
    }
    child = child->next();
    const MathInitState &func_state = get_func_state(func);
    for (int i = 0; i < MathInitState::COUNT; i++) {
        MathInitCall *call = func_state.at(i);
        if (!call->is_none() && !call->is_undef()) {
            call = map_call_expr_args(call, child);
        }
        state.set(i, call);
    }
    return state;
}

MathInitCall *MathInitPass::map_call_expr_args(MathInitCall *call, StmtNode *first_arg) {
    std::vector<StmtNode *> args;
    for (StmtNode *arg = first_arg; arg != nullptr; arg = arg->next()) {
        args.push_back(arg);
    }
    int num_args = int(args.size());
    std::vector<VarNode *> params(num_args);
    bool have_params = false;
    int arg_count = call->arg_count();
    for (int i = 0; i < arg_count; i++) {
        int arg_index = call->arg_param(i);
        if (arg_index == MathInitArgConst::ARG_UNDEF) {
            continue;
        }
        // by construction, argument for 'arg_index' must exist and refer to caller parameter
        if (arg_index >= num_args) {
            assert(false);
            return m_math_init_undef;
        }
        VarNode *param = args[arg_index]->decl_ref();
        if (param == nullptr || param->param_index() < 0) {
            assert(false);
            return m_math_init_undef;
        }
        params[i] = param;
        have_params = true;
    }
    if (!have_params) {
        return call;
    }
    MathInitCall *result = create_math_init_call(call->func(), arg_count);
    for (int i = 0; i < arg_count; i++) {
        VarNode *param = params[i];
        if (param != nullptr) {
            result->set_arg(
                i, 
                param->param_index(), 
                MathInitArgConst::ARG_UNDEF, 
                param->name());
        } else {
            result->set_arg(
                i, 
                call->arg_param(i), 
                call->arg_value(i), 
                call->arg_code(i));
        }
    }
    return result;
}

void MathInitPass::combine_init_states(MathInitState &state, const MathInitState &other) {
    for (int i = 0; i < MathInitState::COUNT; i++) {
        state.set(i, combine_init_calls(state.at(i), other.at(i)));
    }
}

void MathInitPass::check_undef_init_funcs(MathInitState &state) {
    for (int i = 0; i < MathInitState::COUNT; i++) {
        if (state.at(i)->any_undef_arg()) {
            state.set(i, m_math_init_undef);
        }
    }
}

MathInitCall *MathInitPass::combine_init_calls(MathInitCall *call1, MathInitCall *call2) {
    if (call1->is_undef() || call2->is_undef()) {
        return m_math_init_undef;
    }
    if (call1->is_none()) {
        return call2;
    }
    if (call2->is_none()) {
        return call1;
    }
    if (!call1->equal(call2)) {
        return m_math_init_undef;
    }
    return call1;
}

bool MathInitPass::pass3() {
    // Pass 3: Evaluate final locations of init calls
    int func_count = m_graph->func_count();
    for (int i = 0; i < func_count; i++) {
        FuncNode *func = m_graph->func_at(i);
        StmtNode *top = func->top_stmt();
        if (top == nullptr) {
            continue;
        }
        MathInitMask mask = make_top_init_mask(func);
        eval_final_init_states(top, mask);
    }
    return true;
}

MathInitMask MathInitPass::make_top_init_mask(FuncNode *func) {
    MathInitMask mask;
    if (func == m_graph->main_func()) {
        return mask;
    }
    const MathInitState &state = get_func_state(func);
    for (int i = 0; i < MathInitState::COUNT; i++) {
        MathInitCall *call = state.at(i);
#if 0 // TODO: Revise this [fixed 07.11.2024]
        if (!call->is_undef()) {
#endif
        // init calls that are not none flow out of function and therefore
        // are not considered inside this function (= masked out)
        if (!call->is_none()) {
            mask.set(i, false);
        }
    }
    return mask;
}

void MathInitPass::eval_final_init_states(StmtNode *stmt, const MathInitMask &mask) {
    if (mask.is_empty()) {
        return;
    }
    if (!may_have_final_init_state(stmt)) {
        eval_final_init_states_children(stmt, mask);
        return;
    }
    const MathInitState &state = get_stmt_state(stmt);
    MathInitState stmt_state;
    MathInitMask stmt_mask(mask);
    bool found = false;
    for (int i = 0; i < MathInitState::COUNT; i++) {
        if (!mask.at(i)) {
            continue;
        }
        MathInitCall *call = state.at(i);
        if (call->is_none()) {
            stmt_mask.set(i, false);
            continue;
        }
        if (call->is_undef()) {
            continue;
        }
        stmt_state.set(i, call);
        stmt_mask.set(i, false);
        found = true;
    }
    if (found) {
        enter_final_state(stmt, stmt_state);
    }
    if (stmt_mask.is_empty()) {
        return;
    }
    eval_final_init_states_children(stmt, stmt_mask);
}

void MathInitPass::eval_final_init_states_children(StmtNode *stmt, const MathInitMask &mask) {
    for (StmtNode *child = stmt->first_child(); child != nullptr; child = child->next()) {
        eval_final_init_states(child, mask);
    }
}

bool MathInitPass::may_have_final_init_state(StmtNode *stmt) {
    Stmt::StmtClass stmt_class = stmt->stmt_class();
    // Any other relevant classes to be added here?
    if (stmt_class == Stmt::IfStmtClass) {
        return false;
    }
    return true;
}

void MathInitPass::build_key_state_map() {
    for (std::pair<StmtNode *, MathInitState> entry: m_final_state_map) {
        StmtKey key = make_stmt_key(entry.first);
        m_key_state_map.emplace(key, entry.second);
    }
}

bool MathInitPass::transform(const std::string &input_code, std::string &output_code) {
    auto make_insert = [this](const MatchResult &result) -> std::string {
        return make_init_call_insert(result);
    };
    TransformerTool transformer_tool;
    RewriteRule rule =
        applyFirst({
            makeRule(
                compoundStmt().bind("stmt"),
                insertBefore(statements("stmt"), cat(transformer::run(make_insert)))),
            makeRule(
                stmt().bind("stmt"),
                insertBefore(statement("stmt"), cat(transformer::run(make_insert))))
            });
    transformer_tool.set_error_handler(m_error_handler);
    return transformer_tool.run(rule, input_code, output_code);
}

std::string MathInitPass::make_init_call_insert(const MatchResult &result) {
    StmtKey key = make_init_call_insert_key(result);
    auto it = m_key_state_map.find(key);
    if (it == m_key_state_map.end()) {
        return std::string();
    }
    MathInitState state = it->second;
    std::string insert;
    for (int i = 0; i < MathInitState::COUNT; i++) {
        MathInitCall *call = state.at(i);
        if (call->is_none()) {
            continue;
        }
        insert += format_math_init_call(call);
    }
    return insert;
}

StmtKey MathInitPass::make_init_call_insert_key(const MatchResult &result) {
    SourceManager &source_manager = result.Context->getSourceManager();
    const Stmt *stmt = result.Nodes.getNodeAs<Stmt>("stmt");
    Stmt::StmtClass stmt_class = stmt->getStmtClass();
    int begin_line = 0;
    int begin_col = 0;
    translate_loc(source_manager, stmt->getBeginLoc(), begin_line, begin_col);
    int end_line = 0;
    int end_col = 0;
    translate_loc(source_manager, stmt->getEndLoc(), end_line, end_col);
    StmtKey key;
    key.stmt_class = int(stmt_class);
    key.begin_line = begin_line;
    key.begin_col = begin_col;
    key.end_line = end_line;
    key.end_col = end_col;
    return key;
}

void MathInitPass::translate_loc(
        const SourceManager &source_manager,
        const SourceLocation &loc,
        int &line,
        int &col) {
    line = int(source_manager.getExpansionLineNumber(loc));
    col = int(source_manager.getExpansionColumnNumber(loc));
}

std::string MathInitPass::format_math_init_call(MathInitCall *call) {
    std::string name = get_math_init_func_name(call->func());
    std::string result = "__" + name + "_init(";
    int arg_count = call->arg_count();
    for (int i = 0; i < arg_count; i++) {
        if (i != 0) {
            result += ", ";
        }
        result += call->arg_code(i);
    }
    result += ");";
    return result;
}

const MathInitState &MathInitPass::get_orig_state(StmtNode *stmt) {
    auto it = m_orig_state_map.find(stmt);
    return (it != m_orig_state_map.end()) ? it->second : m_empty_state;
}

const MathInitState &MathInitPass::get_stmt_state(StmtNode *stmt) {
    auto it = m_stmt_state_map.find(stmt);
    return (it != m_stmt_state_map.end()) ? it->second : m_empty_state;
}

const MathInitState &MathInitPass::get_func_state(FuncNode *func) {
    auto it = m_func_state_map.find(func);
    return (it != m_func_state_map.end()) ? it->second : m_empty_state;
}

const MathInitState &MathInitPass::get_final_state(StmtNode *stmt) {
    auto it = m_final_state_map.find(stmt);
    return (it != m_final_state_map.end()) ? it->second : m_empty_state;
}

void MathInitPass::enter_orig_state(StmtNode *stmt, const MathInitState &state) {
    if (!state.is_empty()) {
        m_orig_state_map.emplace(stmt, state);
    }
}

void MathInitPass::enter_stmt_state(StmtNode *stmt, const MathInitState &state) {
    if (!state.is_empty()) {
        m_stmt_state_map.emplace(stmt, state);
    }
}

void MathInitPass::enter_func_state(FuncNode *func, const MathInitState &state) {
    if (!state.is_empty()) {
        m_func_state_map.emplace(func, state);
    }
}

void MathInitPass::enter_final_state(StmtNode *stmt, const MathInitState &state) {
    if (!state.is_empty()) {
        m_final_state_map.emplace(stmt, state);
    }
}

void MathInitPass::diag_dump_graph() {
    if (!DIAG_DUMP_ENABLE) {
        return;
    }
    DiagStmtGraphVisitor visitor;
    visitor.visit(m_graph.get());
}

void MathInitPass::diag_dump_orig_funcs() {
    if (!DIAG_DUMP_ENABLE) {
        return;
    }
    DiagAnnotStmtGraphVisitor visitor;
    visitor.set_annot_stmt(
        [this](StmtNode *stmt) -> std::string {
            return diag_annot_orig_stmt(stmt);
        });
    visitor.visit(m_graph.get());
}

std::string MathInitPass::diag_annot_orig_stmt(StmtNode *stmt) {
    const MathInitState &state = get_orig_state(stmt);
    return format_init_state(state);
}

void MathInitPass::diag_dump_flow_funcs() {
    if (!DIAG_DUMP_ENABLE) {
        return;
    }
    DiagAnnotStmtGraphVisitor visitor;
    visitor.set_annot_func(
        [this](FuncNode *func) -> std::string {
            return diag_annot_flow_func(func);
        });
    visitor.set_annot_stmt(
        [this](StmtNode *stmt) -> std::string {
            return diag_annot_flow_stmt(stmt);
        });
    visitor.visit(m_graph.get());
}

std::string MathInitPass::diag_annot_flow_func(FuncNode *func) {
    const MathInitState &state = get_func_state(func);
    return format_init_state(state);
}

std::string MathInitPass::diag_annot_flow_stmt(StmtNode *stmt) {
    const MathInitState &state = get_stmt_state(stmt);
    return format_init_state(state);
}

void MathInitPass::diag_dump_final_funcs() {
    if (!DIAG_DUMP_ENABLE) {
        return;
    }
    DiagAnnotStmtGraphVisitor visitor;
    visitor.set_annot_stmt(
        [this](StmtNode *stmt) -> std::string {
            return diag_annot_final_stmt(stmt);
        });
    visitor.visit(m_graph.get());
}

std::string MathInitPass::diag_annot_final_stmt(StmtNode *stmt) {
    const MathInitState &state = get_final_state(stmt);
    return format_init_state(state);
}

std::string MathInitPass::format_init_state(const MathInitState &state) {
    std::string result;
    for (int i = 0; i < MathInitState::COUNT; i++) {
        MathInitCall *call = state.at(i);
        if (!call->is_none()) {
            if (result.empty()) {
                result += "init [";
            } else {
                result += " ";
            }
            result += call->str();
        }
    }
    if (!result.empty()) {
        result += "]";
    }
    return result;
}

void MathInitPass::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

