// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cassert>
#include <string>
#include <functional>
#include <cstdint>

#include "clang/AST/Stmt.h"

#include "core/util.hpp"
#include "core/graph.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;

//
//    NodeBase
//

NodeBase::NodeBase(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col):
            m_graph(graph),
            m_begin_line(begin_line),
            m_begin_col(begin_col),
            m_end_line(end_line),
            m_end_col(end_col) { }

NodeBase::~NodeBase() { }

//
//    StmtNode
//

StmtNode::StmtNode(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col,
        Stmt::StmtClass stmt_class):
            NodeBase(
                graph,
                begin_line,
                begin_col,
                end_line,
                end_col),
            m_stmt_class(stmt_class),
            m_decl_ref(nullptr),
            m_func_ref(nullptr),
            m_is_bool_literal(false),
            m_bool_literal(false),
            m_is_int_literal(false),
            m_int_literal(0),
            m_is_int_const_expr(false),
            m_int_const_expr(0),
            m_prev(nullptr),
            m_next(nullptr),
            m_parent(nullptr),
            m_first_child(nullptr),
            m_last_child(nullptr) { 
        m_graph->enter_stmt(this);
}

StmtNode::~StmtNode() { }

std::string StmtNode::stmt_class_name() {
    return m_graph->get_stmt_class_name(m_stmt_class);
}

void StmtNode::add_child(StmtNode *stmt) {
    assert(stmt->m_graph == m_graph);
    stmt->m_parent = this;
    if (m_last_child == nullptr) {
        m_first_child = stmt;
    } else {
        m_last_child->m_next = stmt;
    }
    stmt->m_prev = m_last_child;
    stmt->m_next = nullptr;
    m_last_child = stmt;
}

//
//    VarNode
//

VarNode::VarNode(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col,
        const std::string &name):
            NodeBase(
                graph,
                begin_line,
                begin_col,
                end_line,
                end_col),
            m_name(name),
            m_param_index(-1) { }

VarNode::~VarNode() { }

//
//    FuncNode
//

FuncNode::FuncNode(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col,
        const std::string &name):
            NodeBase(
                graph,
                begin_line,
                begin_col,
                end_line,
                end_col),
            m_name(name),
            m_prev(nullptr),
            m_next(nullptr),
            m_top_stmt(nullptr) { 
    m_graph->enter_func(this);
}

FuncNode::~FuncNode() { }

void FuncNode::add_param(VarNode *param) {
    assert(param->graph() == m_graph);
    m_params.push_back(param);
}

//
//    StmtGraph
//

StmtGraph::StmtGraph():
        m_main_func(nullptr) { }

StmtGraph::~StmtGraph() { }

void StmtGraph::add_func(FuncNode *func) {
    assert(func->graph() == this);
    m_funcs.push_back(func);
}

void StmtGraph::enter_func(FuncNode *func) {
    m_func_nodes.emplace_back(func);
}

void StmtGraph::enter_var(VarNode *var) {
    m_var_nodes.emplace_back(var);
}

void StmtGraph::enter_stmt(StmtNode *stmt) {
    m_stmt_nodes.emplace_back(stmt);
}

void StmtGraph::enter_stmt_class_name(Stmt::StmtClass stmt_class, const std::string &name) {
    m_stmt_class_name_map.emplace(stmt_class, name);
}

std::string StmtGraph::get_stmt_class_name(Stmt::StmtClass stmt_class) {
    return m_stmt_class_name_map[stmt_class];
}

//
//    StmtGraphVisitor
//

StmtGraphVisitor::StmtGraphVisitor() { }

StmtGraphVisitor::~StmtGraphVisitor() { }

void StmtGraphVisitor::visit(StmtGraph *graph) {
    start_graph(graph);
    int func_count = graph->func_count();
    for (int i = 0; i < func_count; i++) {
        FuncNode *func = graph->func_at(i);
        visit_func(func);
    }
    end_graph(graph);
}

void StmtGraphVisitor::visit_func(FuncNode *func) {
    start_func(func);
    int param_count = func->param_count();
    for (int i = 0; i < param_count; i++) {
        VarNode *param = func->param_at(i);
        visit_param(param);
    }
    // function top statement must always be present in well formed graph
    StmtNode *top = func->top_stmt();
    if (top != nullptr) {
        visit_stmt(top);
    }
    end_func(func);
}

void StmtGraphVisitor::visit_param(VarNode *param) {
    access_param(param);
}

void StmtGraphVisitor::visit_stmt(StmtNode *stmt) {
    start_stmt(stmt);
    for (StmtNode *child = stmt->first_child(); child != nullptr; child = child->next()) {
        visit_stmt(child);
    }
    end_stmt(stmt);
}

void StmtGraphVisitor::start_graph(StmtGraph *graph) { }

void StmtGraphVisitor::end_graph(StmtGraph *graph) { }

void StmtGraphVisitor::start_func(FuncNode *func) { }

void StmtGraphVisitor::end_func(FuncNode *func) { }

void StmtGraphVisitor::access_param(VarNode *param) { }

void StmtGraphVisitor::start_stmt(StmtNode *stmt) { }

void StmtGraphVisitor::end_stmt(StmtNode *stmt) { }

//
//    CallbackStmtGraphVisitor
//

CallbackStmtGraphVisitor::CallbackStmtGraphVisitor():
        m_on_start_graph(nullptr),
        m_on_end_graph(nullptr),
        m_on_start_func(nullptr),
        m_on_end_func(nullptr),
        m_on_access_param(nullptr),
        m_on_start_stmt(nullptr),
        m_on_end_stmt(nullptr) {}

CallbackStmtGraphVisitor::~CallbackStmtGraphVisitor() { }

void CallbackStmtGraphVisitor::start_graph(StmtGraph *graph) {
    if (m_on_start_graph != nullptr) {
        m_on_start_graph(graph);
    }
}

void CallbackStmtGraphVisitor::end_graph(StmtGraph *graph) {
    if (m_on_end_graph != nullptr) {
        m_on_end_graph(graph);
    }
}

void CallbackStmtGraphVisitor::start_func(FuncNode *func) {
    if (m_on_start_func != nullptr) {
        m_on_start_func(func);
    }
}

void CallbackStmtGraphVisitor::end_func(FuncNode *func) {
    if (m_on_end_func != nullptr) {
        m_on_end_func(func);
    }
}

void CallbackStmtGraphVisitor::access_param(VarNode *param) {
    if (m_on_access_param != nullptr) {
        m_on_access_param(param);
    }
}

void CallbackStmtGraphVisitor::start_stmt(StmtNode *stmt) {
    if (m_on_start_stmt != nullptr) {
        m_on_start_stmt(stmt);
    }
}

void CallbackStmtGraphVisitor::end_stmt(StmtNode *stmt) {
    if (m_on_end_stmt != nullptr) {
        m_on_end_stmt(stmt);
    }
}

//
//    DiagStmtGraphVisitor
//

DiagStmtGraphVisitor::DiagStmtGraphVisitor():
        m_level(0) { }

DiagStmtGraphVisitor::~DiagStmtGraphVisitor() { }

void DiagStmtGraphVisitor::start_graph(StmtGraph *graph) {
    printf("Graph\n");
    m_level++;
}

void DiagStmtGraphVisitor::end_graph(StmtGraph *graph) {
    m_level--;
}

void DiagStmtGraphVisitor::start_func(FuncNode *func) {
    print_leader();
    printf("FunctionDecl [%s] ", func->name().c_str());
    print_loc(func);
    printf("\n");
    m_level++;
}

void DiagStmtGraphVisitor::end_func(FuncNode *func) {
    m_level--;
}

void DiagStmtGraphVisitor::access_param(VarNode *param) {
    print_leader();
    printf("ParmVarDecl [%d] [%s] ", param->param_index(), param->name().c_str());
    print_loc(param);
    printf("\n");
}

void DiagStmtGraphVisitor::start_stmt(StmtNode *stmt) {
    std::string name = stmt->stmt_class_name();
    print_leader();
    printf("%s ", name.c_str());
    print_loc(stmt);
    std::string code = stmt->code();
    if (!code.empty()) {
        printf(" code [%s]", code.c_str());
    }
    std::string type_name = stmt->type_name();
    if (!type_name.empty()) {
        printf(" type [%s]", type_name.c_str());
    }
    std::string member_name = stmt->member_name();
    if (!member_name.empty()) {
        printf(" member [%s]", member_name.c_str());
    }
    VarNode *decl_ref = stmt->decl_ref();
    if (decl_ref != nullptr) {
        printf(" decl [%s]", decl_ref->name().c_str());
    }
    FuncNode *func_ref = stmt->func_ref();
    if (func_ref != nullptr) {
        printf(" func [%s]", func_ref->name().c_str());
    }
    if (stmt->is_bool_literal()) {
        printf(" bool [%s]", stmt->bool_literal() ? "true" : "false");
    }
    if (stmt->is_int_literal()) {
        printf(" int [%d]", stmt->int_literal());
    }
    if (stmt->is_int_const_expr()) {
        printf(" const [%d]", stmt->int_const_expr());
    }
    printf("\n");
    m_level++;
}

void DiagStmtGraphVisitor::end_stmt(StmtNode *stmt) {
    m_level--;
}

void DiagStmtGraphVisitor::print_loc(NodeBase *node) {
    printf("at [%d:%d .. %d:%d]", 
        node->begin_line(), 
        node->begin_col(), 
        node->end_line(), 
        node->end_col());
}

void DiagStmtGraphVisitor::print_leader() {
    for (int i = 0; i < m_level * INDENT; i++) {
        fputc(' ', stdout);
    }
}

//
//    DiagAnnotStmtGraphVisitor
//

DiagAnnotStmtGraphVisitor::DiagAnnotStmtGraphVisitor():
        m_annot_func(nullptr),
        m_annot_param(nullptr),
        m_annot_stmt(nullptr),
        m_level(0) { }

DiagAnnotStmtGraphVisitor::~DiagAnnotStmtGraphVisitor() { }

void DiagAnnotStmtGraphVisitor::start_graph(StmtGraph *graph) {
    printf("Graph\n");
    m_level++;
}

void DiagAnnotStmtGraphVisitor::end_graph(StmtGraph *graph) {
    m_level--;
}

void DiagAnnotStmtGraphVisitor::start_func(FuncNode *func) {
    print_leader();
    printf("FunctionDecl [%s] ", func->name().c_str());
    print_loc(func);
    if (m_annot_func) {
        std::string annot = m_annot_func(func);
        if (!annot.empty()) {
            printf(" %s", annot.c_str());
        }
    }
    printf("\n");
    m_level++;
}

void DiagAnnotStmtGraphVisitor::end_func(FuncNode *func) {
    m_level--;
}

void DiagAnnotStmtGraphVisitor::access_param(VarNode *param) {
    print_leader();
    printf("ParmVarDecl [%d] [%s] ", param->param_index(), param->name().c_str());
    print_loc(param);
    if (m_annot_param) {
        std::string annot = m_annot_param(param);
        if (!annot.empty()) {
            printf(" %s", annot.c_str());
        }
    }
    printf("\n");
}

void DiagAnnotStmtGraphVisitor::start_stmt(StmtNode *stmt) {
    std::string name = stmt->stmt_class_name();
    print_leader();
    printf("%s ", name.c_str());
    print_loc(stmt);
    if (m_annot_stmt) {
        std::string annot = m_annot_stmt(stmt);
        if (!annot.empty()) {
            printf(" %s", annot.c_str());
        }
    }
    printf("\n");
    m_level++;
}

void DiagAnnotStmtGraphVisitor::end_stmt(StmtNode *stmt) {
    m_level--;
}

void DiagAnnotStmtGraphVisitor::print_loc(NodeBase *node) {
    printf("at [%d:%d .. %d:%d]", 
        node->begin_line(), 
        node->begin_col(), 
        node->end_line(), 
        node->end_col());
}

void DiagAnnotStmtGraphVisitor::print_leader() {
    for (int i = 0; i < m_level * INDENT; i++) {
        fputc(' ', stdout);
    }
}

//
//    StmtKey
//

StmtKey make_stmt_key(StmtNode *stmt) {
    StmtKey key;
    key.stmt_class = int(stmt->stmt_class());
    key.begin_line = stmt->begin_line();
    key.begin_col = stmt->begin_col();
    key.end_line = stmt->end_line();
    key.end_col = stmt->end_col();
    return key;
}

size_t StmtKeyHash::operator()(const StmtKey &key) const noexcept {
    size_t h = std::hash<int>()(key.stmt_class);
    h = hash_combine(h, std::hash<int>()(key.begin_line));
    h = hash_combine(h, std::hash<int>()(key.begin_col));
    h = hash_combine(h, std::hash<int>()(key.end_line));
    h = hash_combine(h, std::hash<int>()(key.end_col));
    return h;
}

bool StmtKeyEqual::operator()(const StmtKey &key1, const StmtKey &key2) const {
    return (key1.stmt_class == key2.stmt_class &&
        key1.begin_line == key2.begin_line &&
        key1.begin_col == key2.begin_col &&
        key1.end_line == key2.end_line &&
        key1.end_col == key2.end_col);
}

} // namespace front
} // namespace tanto
} // namespace ronin

