// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <functional>

#include "clang/AST/Stmt.h"

#include "core/error.hpp"
#include "core/tooling.hpp"
#include "core/dead_code_pass.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;

namespace {

// ACHTUNG: Unify with "graph_builder.cpp"?
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

bool match_const_expr_if(const IfStmt &stmt, ASTContext &context, bool value) {
    const Expr *cond = stmt.getCond();
    if (cond == nullptr) {
        return false;
    }
    int int_value = 0;
    if (!get_int_const_expr_value(cond, &context, int_value)) {
        return false;
    }
    bool bool_value = (int_value != 0);
    return (bool_value == value);
}

std::function<bool (const IfStmt &stmt, ASTContext &context)> 
        make_match_const_expr_if(bool value) {
    return [value](const IfStmt &stmt, ASTContext &context) -> bool {
        return match_const_expr_if(stmt, context, value);
    };
}

} // namespace

//
//    DeadCodePass
//

DeadCodePass::DeadCodePass():
        m_error_handler(nullptr) { 
    create_rules();        
}

DeadCodePass::~DeadCodePass() { }

void DeadCodePass::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
}

bool DeadCodePass::run(const std::string &input_code, std::string &output_code) {
    std::string pass_input;
    std::string pass_output;
    pass_input = input_code;
    if (!rewrite(m_pass1_rule, pass_input, pass_output)) {
        return false;
    }
    pass_input = pass_output;
    if (!rewrite(m_pass2_rule, pass_input, pass_output)) {
        return false;
    }
    output_code = pass_output;
    return true;
}

bool DeadCodePass::rewrite(
        RewriteRule rule,
        const std::string &input_code, 
        std::string &output_code) {
    TransformerTool transformer_tool;
    transformer_tool.set_error_handler(m_error_handler);
    if (!transformer_tool.run(rule, input_code, output_code)) {
        return false;
    }
    return true;
}

void DeadCodePass::create_rules() {
    m_pass1_rule =
        applyFirst({
            makeRule(
                ifStmt(
                    hasCondition(expr()),
                    unless(hasConditionVariableStatement(declStmt())),
                    unless(hasInitStatement(stmt())),
                    customMatch(make_match_const_expr_if(true)),
                    hasThen(stmt().bind("then")),
                    hasElse(stmt().bind("else"))
                ).bind("stmt"),
                {
                    remove(between(before(statement("stmt")), before(statement("then")))),
                    remove(between(after(statement("then")), after(statement("else"))))
                }),
            makeRule(
                ifStmt(
                    hasCondition(expr()),
                    unless(hasConditionVariableStatement(declStmt())),
                    unless(hasInitStatement(stmt())),
                    customMatch(make_match_const_expr_if(true)),
                    hasThen(stmt().bind("then"))
                ).bind("stmt"),
                remove(between(before(statement("stmt")), before(statement("then"))))),
            makeRule(
                ifStmt(
                    hasCondition(expr()),
                    unless(hasConditionVariableStatement(declStmt())),
                    unless(hasInitStatement(stmt())),
                    customMatch(make_match_const_expr_if(false)),
                    hasThen(stmt().bind("then")),
                    hasElse(stmt().bind("else"))
                ).bind("stmt"),
                remove(between(before(statement("stmt")), before(statement("else"))))),
            makeRule(
                ifStmt(
                    hasCondition(expr()),
                    unless(hasConditionVariableStatement(declStmt())),
                    unless(hasInitStatement(stmt())),
                    customMatch(make_match_const_expr_if(false)),
                    hasThen(stmt().bind("then"))
                ).bind("stmt"),
                remove(statement("stmt"))),
        });
    m_pass2_rule = 
        makeRule(
            compoundStmt(
                hasParent(compoundStmt()),
                unless(hasAnySubstatement(declStmt()))
            ).bind("stmt"),
            {
                // remove "stmt" outer braces
                remove(between(before(statement("stmt")), before(statements("stmt")))),
                remove(between(after(statements("stmt")), after(statement("stmt"))))
            });
}

void DeadCodePass::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

