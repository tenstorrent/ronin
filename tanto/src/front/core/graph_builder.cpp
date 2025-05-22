// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <memory>
#include <optional>
#include <algorithm>

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

#include "core/error.hpp"
#include "core/tooling.hpp"
#include "core/graph.hpp"
#include "core/graph_builder.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;

namespace {

int cmp_locs(int a_line, int a_col, int b_line, int b_col) {
    int cmp = a_line - b_line;
    if (cmp != 0) {
        return cmp;
    }
    return a_col - b_col;
}

int cmp_nodes(NodeBase *a, NodeBase *b) {
    // node with lower begin location goes first
    int cmp = cmp_locs(a->begin_line(), a->begin_col(), b->begin_line(), b->begin_col());
    if (cmp != 0) {
        return cmp;
    }
    // same begin locations: node with higher end location goes first
    cmp = cmp_locs(a->end_line(), a->end_col(), b->end_line(), b->end_col());
    return (-cmp);
}

std::string format_stmt(const Stmt *stmt, ASTContext *context) {
    std::string result;
    llvm::raw_string_ostream os(result);
    PrintingPolicy policy(context->getLangOpts());
    stmt->printPretty(os, nullptr, policy);
    return os.str();
}

bool get_int_literal_value(const IntegerLiteral *int_literal, int &value) {
    value = 0;
    llvm::APInt ap_int = int_literal->getValue();
    std::optional<int64_t> sext_value = ap_int.trySExtValue();
    if (!sext_value) {
        return false;
    }
    int64_t value64 = *sext_value;
    value = int(value64);
    if (int64_t(value) != value64) {
        return false;
    }
    return true;
}

bool get_int_const_expr_value(const Expr *expr, ASTContext *context, int &value) {
    value = 0;
    std::optional<llvm::APSInt> aps_int = expr->getIntegerConstantExpr(*context);
    if (!aps_int) {
        return false;
    }
    std::optional<int64_t> sext_value = (*aps_int).trySExtValue();
    if (!sext_value) {
        return false;
    }
    int64_t value64 = *sext_value;
    value = int(value64);
    if (int64_t(value) != value64) {
        return false;
    }
    return true;
}

} // namespace

//
//    StmtGraphBuilder
//

StmtGraphBuilder::StmtGraphBuilder():
        m_error_handler(nullptr) { }

StmtGraphBuilder::~StmtGraphBuilder() { }

void StmtGraphBuilder::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
    m_matcher_tool.set_error_handler(error_handler);
}

bool StmtGraphBuilder::build(const std::string &input_code, std::unique_ptr<StmtGraph> &graph) {
    reset();
    if (!collect_stmts(input_code)) {
        return false;
    }
    sort_funcs();
    sort_stmts();
    if (!fill_graph()) {
        return false;
    }
    graph.reset(m_graph.release());
    return true;
}

void StmtGraphBuilder::reset() {
    m_graph.reset(new StmtGraph());
    m_stmt_entries.clear();
    m_funcs.clear();
    m_func_map.clear();
    m_var_map.clear();
    m_stmt_map.clear();
    m_var_stmt_map.clear();
}

bool StmtGraphBuilder::collect_stmts(const std::string &input_code) {
    auto matcher =
        traverse(TK_IgnoreUnlessSpelledInSource, 
            stmt(
                hasAncestor(functionDecl().bind("func")),
                optionally(hasParent(stmt().bind("parent"))),
                optionally(expr(hasType(cxxRecordDecl().bind("recordDecl")))),
                optionally(declRefExpr(hasDeclaration(varDecl().bind("varDecl")))),
                optionally(declRefExpr(hasDeclaration(functionDecl().bind("funcDecl")))),
                optionally(hasParent(varDecl().bind("initParent"))),
                optionally(cxxBoolLiteral().bind("boolLiteral")),
                optionally(integerLiteral().bind("intLiteral")),
                optionally(expr().bind("expr"))
            ).bind("stmt"));
    m_matcher_tool.reset_code(input_code);
    std::vector<MatchResult> results = m_matcher_tool.match(matcher);
    for (MatchResult &result: results) {
        parse_match_result(result);
    }
    return true;
}

void StmtGraphBuilder::parse_match_result(const MatchResult &result) {
    SourceManager &source_manager = result.Context->getSourceManager();
    const Stmt *ast_stmt = result.Nodes.getNodeAs<Stmt>("stmt");
    const Stmt *ast_parent = result.Nodes.getNodeAs<Stmt>("parent");
    const FunctionDecl *ast_func = result.Nodes.getNodeAs<FunctionDecl>("func");
    StmtNode *stmt = enter_stmt(ast_stmt, source_manager);
    StmtNode *parent = 
        (ast_parent != nullptr) ? 
            enter_stmt(ast_parent, source_manager) : 
            nullptr;
    FuncNode *func = enter_func(ast_func, source_manager);
    parse_func_params(result, ast_func, func);
    parse_stmt_attrs(result, ast_stmt, stmt, func);
    m_stmt_entries.emplace_back(StmtEntry{stmt, parent, func});
}

void StmtGraphBuilder::parse_func_params(
        const MatchResult &result, 
        const FunctionDecl *ast_func, 
        FuncNode *func) {
    unsigned num_params = ast_func->getNumParams();
    if (func->param_count() == num_params) {
        // already have parameters (if any)
        return;
    }
    SourceManager &source_manager = result.Context->getSourceManager();
    for (unsigned i = 0; i < num_params; i++) {
        const ParmVarDecl *ast_param = ast_func->getParamDecl(i);
        VarNode *param = enter_var(ast_param, source_manager);
        param->set_param_index(i);
        func->add_param(param);
    }
}

void StmtGraphBuilder::parse_stmt_attrs(
        const MatchResult &result, 
        const Stmt *ast_stmt,
        StmtNode *stmt,
        FuncNode *func) {
    // to save space, store code only for call arguments
    const CallExpr *call_expr = result.Nodes.getNodeAs<CallExpr>("parent");
    if (call_expr != nullptr) {
        std::string code = format_stmt(ast_stmt, result.Context);
        stmt->set_code(code);
    }
    const CXXRecordDecl *record_decl = result.Nodes.getNodeAs<CXXRecordDecl>("recordDecl");
    if (record_decl != nullptr) {
        std::string name = record_decl->getNameAsString();
        stmt->set_type_name(name);
    }
    const MemberExpr *member_expr = result.Nodes.getNodeAs<MemberExpr>("stmt");
    if (member_expr != nullptr) {
        std::string name = member_expr->getMemberDecl()->getNameAsString();
        stmt->set_member_name(name);
    }
    const VarDecl *var_decl = result.Nodes.getNodeAs<VarDecl>("varDecl");
    if (var_decl != nullptr) {
        SourceManager &source_manager = result.Context->getSourceManager();
        VarNode *var = enter_var(var_decl, source_manager);
        stmt->set_decl_ref(var);
    }
    const FunctionDecl *func_decl = result.Nodes.getNodeAs<FunctionDecl>("funcDecl");
    if (func_decl != nullptr) {
        SourceManager &source_manager = result.Context->getSourceManager();
        FuncNode *func = enter_func(func_decl, source_manager);
        stmt->set_func_ref(func);
    }
    const DeclStmt *decl_stmt = result.Nodes.getNodeAs<DeclStmt>("stmt");
    if (decl_stmt != nullptr) {
        parse_decl_stmt(result, decl_stmt, stmt, func);
    }
    const VarDecl *init_parent = result.Nodes.getNodeAs<VarDecl>("initParent");
    if (init_parent != nullptr) {
        auto it = m_var_stmt_map.find(init_parent);
        if (it != m_var_stmt_map.end()) {
            StmtNode *stmt_var = it->second;
            m_stmt_entries.emplace_back(StmtEntry{stmt, stmt_var, func});
        }
        // report error if parent not found? (must not happen)
    }
    const CXXBoolLiteralExpr *bool_literal = 
        result.Nodes.getNodeAs<CXXBoolLiteralExpr>("boolLiteral");
    if (bool_literal != nullptr) {
        bool value = bool_literal->getValue();
        stmt->set_bool_literal(value);
    }
    const IntegerLiteral *int_literal = result.Nodes.getNodeAs<IntegerLiteral>("intLiteral");
    if (int_literal != nullptr) {
        int value = 0;
        if (get_int_literal_value(int_literal, value)) {
            stmt->set_int_literal(value);
        }
    }
    const Expr *expr = result.Nodes.getNodeAs<Expr>("expr");
    if (expr != nullptr) {
        int value = 0;
        if (get_int_const_expr_value(expr, result.Context, value)) {
            stmt->set_int_const_expr(value);
        }
    }
}

void StmtGraphBuilder::parse_decl_stmt(
        const MatchResult &result, 
        const DeclStmt *ast_stmt,
        StmtNode *stmt,
        FuncNode *func) {
    SourceManager &source_manager = result.Context->getSourceManager();
    for (const Decl *ast_decl: ast_stmt->decls()) {
        const VarDecl *ast_var = dyn_cast<VarDecl>(ast_decl);
        if (ast_var == nullptr) {
            continue;
        }
        StmtNode *stmt_var = enter_stmt_var(ast_var, source_manager);
        m_stmt_entries.emplace_back(StmtEntry{stmt_var, stmt, func});
        m_var_stmt_map.emplace(ast_var, stmt_var);
    }
}

void StmtGraphBuilder::sort_funcs() {
    for (auto &entry: m_func_map) {
        m_funcs.push_back(entry.second);
    }
    std::sort(
        m_funcs.begin(),
        m_funcs.end(),
        [](FuncNode *a, FuncNode *b) -> bool {
            return (cmp_nodes(a, b) < 0);
        });
}

void StmtGraphBuilder::sort_stmts() {
    std::sort(
        m_stmt_entries.begin(),
        m_stmt_entries.end(),
        [](const StmtEntry &a, const StmtEntry &b) -> bool {
            return (cmp_nodes(a.stmt, b.stmt) < 0);
        });
}

bool StmtGraphBuilder::fill_graph() {
    if (!fill_stmts()) {
        return false;
    }
    if (!fill_funcs()) {
        return false;
    }
    return true;
}

bool StmtGraphBuilder::fill_stmts() {
    for (StmtEntry &entry: m_stmt_entries) {
        StmtNode *stmt = entry.stmt;
        StmtNode *parent = entry.parent;
        FuncNode *func = entry.func;
        if (parent != nullptr) {
            parent->add_child(stmt);
        } else {
            if (stmt->stmt_class() == Stmt::StmtClass::CompoundStmtClass) {
                if (func->top_stmt() != nullptr) {
                    error("Duplicate top level statement, function " + func->name());
                    return false;
                }
                func->set_top_stmt(stmt);
            }
            // ACHTUNG: Initailizer declarations in 'for' statements are represented
            //     as "faux" DeclStmt and their children experessions have null parents.
            //     For the time being, such children will be lost and not added to graph.
            //     (How to fix this in the future?)
        }
    }
    return true;
}

bool StmtGraphBuilder::fill_funcs() {
    for (FuncNode *func: m_funcs) {
        if (func->top_stmt() == nullptr) {
            continue;
        }
        m_graph->add_func(func);
        if (func->name() == "kernel") {
            if (m_graph->main_func() != nullptr) {
                error("Duplicate main function");
                return false;
            }
            m_graph->set_main_func(func);
        }
    }
    if (m_graph->main_func() == nullptr) {
        error("Missing main function");
        return false;
    }
    return true;
}

FuncNode *StmtGraphBuilder::enter_func(
        const FunctionDecl *ast_func, const SourceManager &source_manager) {
    auto it = m_func_map.find(ast_func);
    if (it != m_func_map.end()) {
        return it->second;
    }
    int begin_line = 0;
    int begin_col = 0;
    translate_loc(source_manager, ast_func->getBeginLoc(), begin_line, begin_col);
    int end_line = 0;
    int end_col = 0;
    translate_loc(source_manager, ast_func->getEndLoc(), end_line, end_col);
    std::string name = ast_func->getNameAsString();
    FuncNode *func = 
        new FuncNode(
            m_graph.get(),
            begin_line,
            begin_col,
            end_line,
            end_col,
            name);
    m_func_map.emplace(ast_func, func);
    return func;
}

VarNode *StmtGraphBuilder::enter_var(
        const VarDecl *ast_var, const SourceManager &source_manager) {
    auto it = m_var_map.find(ast_var);
    if (it != m_var_map.end()) {
        return it->second;
    }
    int begin_line = 0;
    int begin_col = 0;
    translate_loc(source_manager, ast_var->getBeginLoc(), begin_line, begin_col);
    int end_line = 0;
    int end_col = 0;
    translate_loc(source_manager, ast_var->getEndLoc(), end_line, end_col);
    std::string name = ast_var->getNameAsString();
    VarNode *var = 
        new VarNode(
            m_graph.get(),
            begin_line,
            begin_col,
            end_line,
            end_col,
            name);
    m_var_map.emplace(ast_var, var);
    return var;
}

StmtNode *StmtGraphBuilder::enter_stmt(
        const Stmt *ast_stmt, const SourceManager &source_manager) {
    auto it = m_stmt_map.find(ast_stmt);
    if (it != m_stmt_map.end()) {
        return it->second;
    }
    int begin_line = 0;
    int begin_col = 0;
    translate_loc(source_manager, ast_stmt->getBeginLoc(), begin_line, begin_col);
    int end_line = 0;
    int end_col = 0;
    translate_loc(source_manager, ast_stmt->getEndLoc(), end_line, end_col);
    Stmt::StmtClass stmt_class = ast_stmt->getStmtClass();
    StmtNode *stmt =
        new StmtNode(
            m_graph.get(),
            begin_line,
            begin_col,
            end_line,
            end_col,
            stmt_class);
    m_stmt_map.emplace(ast_stmt, stmt);
    m_graph->enter_stmt_class_name(stmt_class, ast_stmt->getStmtClassName());
    return stmt;
}

StmtNode *StmtGraphBuilder::enter_stmt_var(
        const VarDecl *ast_var, const SourceManager &source_manager) {
    // build fictionary statement node
    int begin_line = 0;
    int begin_col = 0;
    translate_loc(source_manager, ast_var->getBeginLoc(), begin_line, begin_col);
    int end_line = 0;
    int end_col = 0;
    translate_loc(source_manager, ast_var->getEndLoc(), end_line, end_col);
    std::string name = ast_var->getNameAsString();
    VarNode *var = 
        new VarNode(
            m_graph.get(),
            begin_line,
            begin_col,
            end_line,
            end_col,
            name);
    m_var_map.emplace(ast_var, var);
    // abuse DeclRefExprClass
    Stmt::StmtClass stmt_class = Stmt::StmtClass::DeclRefExprClass;
    StmtNode *stmt =
        new StmtNode(
            m_graph.get(),
            begin_line,
            begin_col,
            end_line,
            end_col,
            stmt_class);
    // no update of m_stmt_map
    m_graph->enter_stmt_class_name(stmt_class, "DeclRefExpr");
    stmt->set_decl_ref(var);
    return stmt;
}

void StmtGraphBuilder::translate_loc(
        const SourceManager &source_manager,
        const SourceLocation &loc,
        int &line,
        int &col) {
    line = int(source_manager.getExpansionLineNumber(loc));
    col = int(source_manager.getExpansionColumnNumber(loc));
}

void StmtGraphBuilder::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

