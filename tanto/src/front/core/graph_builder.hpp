// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

#include "core/error.hpp"
#include "core/tooling.hpp"
#include "core/graph.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;

class StmtGraphBuilder {
public:
    StmtGraphBuilder();
    ~StmtGraphBuilder();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool build(const std::string &input_code, std::unique_ptr<StmtGraph> &graph);
private:
    void reset();
    bool collect_stmts(const std::string &input_code);
    void parse_match_result(const MatchResult &result);
    void parse_func_params(
        const MatchResult &result, 
        const FunctionDecl *ast_func, 
        FuncNode *func);
    void parse_stmt_attrs(
        const MatchResult &result, 
        const Stmt *ast_stmt,
        StmtNode *stmt,
        FuncNode *func);
    void parse_decl_stmt(
        const MatchResult &result, 
        const DeclStmt *ast_stmt,
        StmtNode *stmt,
        FuncNode *func);
    void sort_funcs();
    void sort_stmts();
    bool fill_graph();
    bool fill_stmts();
    bool fill_funcs();
    FuncNode *enter_func(const FunctionDecl *ast_func, const SourceManager &source_manager);
    VarNode *enter_var(const VarDecl *ast_var, const SourceManager &source_manager);
    StmtNode *enter_stmt(const Stmt *ast_stmt, const SourceManager &source_manager);
    StmtNode *enter_stmt_var(const VarDecl *ast_var, const SourceManager &source_manager);
    static void translate_loc(
        const SourceManager &source_manager,
        const SourceLocation &loc,
        int &line,
        int &col);
    void error(const std::string &text);
private:
    struct StmtEntry {
        StmtNode *stmt;
        StmtNode *parent;
        FuncNode *func;
    };
private:
    ErrorHandler *m_error_handler;
    MatcherTool m_matcher_tool;
    std::unique_ptr<StmtGraph> m_graph;
    std::vector<StmtEntry> m_stmt_entries;
    std::vector<FuncNode *> m_funcs;
    std::unordered_map<const FunctionDecl *, FuncNode *> m_func_map;
    std::unordered_map<const VarDecl *, VarNode *> m_var_map;
    std::unordered_map<const Stmt *, StmtNode *> m_stmt_map;
    std::unordered_map<const VarDecl *, StmtNode *> m_var_stmt_map;
};

} // namespace front
} // namespace tanto
} // namespace ronin

