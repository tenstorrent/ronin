// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <utility>

#include "clang/AST/Stmt.h"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;

class StmtNode;
class VarNode;
class FuncNode;
class StmtGraph;

class NodeBase {
public:
    NodeBase(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col);
    ~NodeBase();
public:
    StmtGraph *graph() {
        return m_graph;
    }
    int begin_line() {
        return m_begin_line;
    }
    int begin_col() {
        return m_begin_col;
    }
    int end_line() {
        return m_end_line;
    }
    int end_col() {
        return m_end_col;
    }
protected:
    StmtGraph *m_graph;
    int m_begin_line;
    int m_begin_col;
    int m_end_line;
    int m_end_col;
};

class StmtNode: public NodeBase {
public:
    StmtNode(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col,
        Stmt::StmtClass stmt_class);
    ~StmtNode();
public:
    Stmt::StmtClass stmt_class() {
        return m_stmt_class;
    }
    std::string stmt_class_name();
    void set_code(const std::string &code) {
        m_code = code;
    }
    std::string code() {
        return m_code;
    }
    void set_type_name(const std::string &type_name) {
        m_type_name = type_name;
    }
    std::string type_name() {
        return m_type_name;
    }
    void set_member_name(const std::string &member_name) {
        m_member_name = member_name;
    }
    std::string member_name() {
        return m_member_name;
    }
    void set_decl_ref(VarNode *decl_ref) {
        m_decl_ref = decl_ref;
    }
    VarNode *decl_ref() {
        return m_decl_ref;
    }
    void set_func_ref(FuncNode *func_ref) {
        m_func_ref = func_ref;
    }
    FuncNode *func_ref() {
        return m_func_ref;
    }
    void set_bool_literal(bool value) {
        m_is_bool_literal = true;
        m_bool_literal = value;
    }
    bool is_bool_literal() {
        return m_is_bool_literal;
    }
    bool bool_literal() {
        return m_bool_literal;
    }
    void set_int_literal(int value) {
        m_is_int_literal = true;
        m_int_literal = value;
    }
    bool is_int_literal() {
        return m_is_int_literal;
    }
    int int_literal() {
        return m_int_literal;
    }
    void set_int_const_expr(int value) {
        m_is_int_const_expr = true;
        m_int_const_expr = value;
    }
    bool is_int_const_expr() {
        return m_is_int_const_expr;
    }
    int int_const_expr() {
        return m_int_const_expr;
    }
    StmtNode *prev() {
        return m_prev;
    }
    StmtNode *next() {
        return m_next;
    }
    StmtNode *parent() {
        return m_parent;
    }
    StmtNode *first_child() {
        return m_first_child;
    }
    StmtNode *last_child() {
        return m_last_child;
    }
    void add_child(StmtNode *stmt);
private:
    Stmt::StmtClass m_stmt_class;
    std::string m_code;
    std::string m_type_name;
    std::string m_member_name;
    VarNode *m_decl_ref;
    FuncNode *m_func_ref;
    bool m_is_bool_literal;
    bool m_bool_literal;
    bool m_is_int_literal;
    int m_int_literal;
    bool m_is_int_const_expr;
    int m_int_const_expr;
    StmtNode *m_prev;
    StmtNode *m_next;
    StmtNode *m_parent;
    StmtNode *m_first_child;
    StmtNode *m_last_child;
};

class VarNode: public NodeBase {
public:
    VarNode(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col,
        const std::string &name);
    ~VarNode();
public:
    std::string name() {
        return m_name;
    }
    void set_param_index(int param_index) {
        m_param_index = param_index;
    }
    int param_index() {
        return m_param_index;
    }
public:
    std::string m_name;
    int m_param_index;
};

class FuncNode: public NodeBase {
public:
    FuncNode(
        StmtGraph *graph,
        int begin_line,
        int begin_col,
        int end_line,
        int end_col,
        const std::string &name);
    ~FuncNode();
public:
    std::string name() {
        return m_name;
    }
    FuncNode *prev() {
        return m_prev;
    }
    FuncNode *next() {
        return m_next;
    }
    int param_count() {
        return int(m_params.size());
    }
    VarNode *param_at(int index) {
        return m_params[index];
    }
    void set_top_stmt(StmtNode *stmt) {
        m_top_stmt = stmt;
    }
    StmtNode *top_stmt() {
        return m_top_stmt;
    }
    void add_param(VarNode *param);
public:
    std::string m_name;
    FuncNode *m_prev;
    FuncNode *m_next;
    std::vector<VarNode *> m_params;
    StmtNode *m_top_stmt;
};

class StmtGraph {
public:
    StmtGraph();
    ~StmtGraph();
public:
    void set_main_func(FuncNode *func) {
        m_main_func = func;
    }
    FuncNode *main_func() {
        return m_main_func;
    }
    int func_count() {
        return int(m_funcs.size());
    }
    FuncNode *func_at(int index) {
        return m_funcs[index];
    }
    void add_func(FuncNode *func);
    void enter_func(FuncNode *func);
    void enter_var(VarNode *var);
    void enter_stmt(StmtNode *stmt);
    void enter_stmt_class_name(Stmt::StmtClass stmt_class, const std::string &name);
    std::string get_stmt_class_name(Stmt::StmtClass stmt_class);
private:
    std::vector<std::unique_ptr<FuncNode>> m_func_nodes;
    std::vector<std::unique_ptr<VarNode>> m_var_nodes;
    std::vector<std::unique_ptr<StmtNode>> m_stmt_nodes;
    std::unordered_map<Stmt::StmtClass, std::string> m_stmt_class_name_map;
    FuncNode *m_main_func;
    std::vector<FuncNode *> m_funcs;
};

class StmtGraphVisitor {
public:
    StmtGraphVisitor();
    virtual ~StmtGraphVisitor();
public:
    void visit(StmtGraph *graph);
    void visit_func(FuncNode *func);
    void visit_param(VarNode *param);
    void visit_stmt(StmtNode *stmt);
public:
    virtual void start_graph(StmtGraph *graph);
    virtual void end_graph(StmtGraph *graph);
    virtual void start_func(FuncNode *func);
    virtual void end_func(FuncNode *func);
    virtual void access_param(VarNode *param);
    virtual void start_stmt(StmtNode *stmt);
    virtual void end_stmt(StmtNode *stmt);
};

class CallbackStmtGraphVisitor: public StmtGraphVisitor {
public:
    CallbackStmtGraphVisitor();
    ~CallbackStmtGraphVisitor();
public:
    void set_on_start_graph(std::function<void (StmtGraph *graph)> callback) { 
        m_on_start_graph = callback;
    }
    void set_on_end_graph(std::function<void (StmtGraph *graph)> callback) {
        m_on_end_graph = callback;
    }
    void set_on_start_func(std::function<void (FuncNode *func)> callback) {
        m_on_start_func = callback;
    }
    void set_on_end_func(std::function<void (FuncNode *func)> callback) {
        m_on_end_func = callback;
    }
    void set_on_access_param(std::function<void (VarNode *param)> callback) {
        m_on_access_param = callback;
    }
    void set_on_start_stmt(std::function<void (StmtNode *stmt)> callback) {
        m_on_start_stmt = callback;
    }
    void set_on_end_stmt(std::function<void (StmtNode *stmt)> callback) {
        m_on_end_stmt = callback;
    }
public:
    void start_graph(StmtGraph *graph) override;
    void end_graph(StmtGraph *graph) override;
    void start_func(FuncNode *func) override;
    void end_func(FuncNode *func) override;
    void access_param(VarNode *param) override;
    void start_stmt(StmtNode *stmt) override;
    void end_stmt(StmtNode *stmt) override;
private:
    std::function<void (StmtGraph *graph)> m_on_start_graph;
    std::function<void (StmtGraph *graph)> m_on_end_graph;
    std::function<void (FuncNode *func)> m_on_start_func;
    std::function<void (FuncNode *func)> m_on_end_func;
    std::function<void (VarNode *param)> m_on_access_param;
    std::function<void (StmtNode *stmt)> m_on_start_stmt;
    std::function<void (StmtNode *stmt)> m_on_end_stmt;
};

class DiagStmtGraphVisitor: public StmtGraphVisitor {
public:
    DiagStmtGraphVisitor();
    ~DiagStmtGraphVisitor();
public:
    void start_graph(StmtGraph *graph) override;
    void end_graph(StmtGraph *graph) override;
    void start_func(FuncNode *func) override;
    void end_func(FuncNode *func) override;
    void access_param(VarNode *param) override;
    void start_stmt(StmtNode *stmt) override;
    void end_stmt(StmtNode *stmt) override;
private:
    void print_loc(NodeBase *node);
    void print_leader();
private:
    static constexpr int INDENT = 2;
private:
    int m_level;
};

class DiagAnnotStmtGraphVisitor: public StmtGraphVisitor {
public:
    DiagAnnotStmtGraphVisitor();
    ~DiagAnnotStmtGraphVisitor();
public:
    void set_annot_func(std::function<std::string (FuncNode *)> annot) {
        m_annot_func = annot;
    }
    void set_annot_param(std::function<std::string (VarNode *)> annot) {
        m_annot_param = annot;
    }
    void set_annot_stmt(std::function<std::string (StmtNode *)> annot) {
        m_annot_stmt = annot;
    }
public:
    void start_graph(StmtGraph *graph) override;
    void end_graph(StmtGraph *graph) override;
    void start_func(FuncNode *func) override;
    void end_func(FuncNode *func) override;
    void access_param(VarNode *param) override;
    void start_stmt(StmtNode *stmt) override;
    void end_stmt(StmtNode *stmt) override;
private:
    void print_loc(NodeBase *node);
    void print_leader();
private:
    static constexpr int INDENT = 2;
private:
    std::function<std::string (FuncNode *)> m_annot_func;
    std::function<std::string (VarNode *)> m_annot_param;
    std::function<std::string (StmtNode *)> m_annot_stmt;
    int m_level;
};

struct StmtKey {
    int stmt_class;
    int begin_line;
    int begin_col;
    int end_line;
    int end_col;
};

StmtKey make_stmt_key(StmtNode *stmt);

struct StmtKeyHash {
    size_t operator()(const StmtKey &key) const noexcept;
};

struct StmtKeyEqual {
    bool operator()(const StmtKey &key1, const StmtKey &key2) const;
};

template<typename T>
using StmtKeyMap = std::unordered_map<StmtKey, T, StmtKeyHash, StmtKeyEqual>;

} // namespace front
} // namespace tanto
} // namespace ronin

