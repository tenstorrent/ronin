// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cstdint>

#include "core/error.hpp"
#include "core/tooling.hpp"
#include "core/query.hpp"

namespace ronin {
namespace tanto {
namespace front {

namespace {

bool parse_kernel_param_type(const std::string &source, DataType &type) {
    size_t pos = source.find("<");
    if (pos == std::string::npos) {
        if (source == "int32") {
            type = DataType::INT32;
        } else if (source == "uint32") {
            type = DataType::UINT32;
        } else if (source == "float") {
            type = DataType::FLOAT;
        } else if (source == "semaphore") {
            type = DataType::SEMAPHORE;
        } else {
            return false;
        }
    } else {
        std::string base = source.substr(0, pos);
        if (base == "global") {
            type = DataType::GLOBAL;
        } else if (base == "local") {
            type = DataType::LOCAL;
        } else if (base == "pipe") {
            type = DataType::PIPE;
        } else {
            return false;
        }
    }
    return true;
}

} // namespace

//
//    Query
//

Query::Query():
        m_error_handler(nullptr) { }

Query::~Query() { }

void Query::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
    m_matcher_tool.set_error_handler(error_handler);
}

bool Query::run(const std::string &input_code) {
    if (!m_matcher_tool.reset_code(input_code)) {
        return false;
    }
    if (!query_kernel_params()) {
        return false;
    }
#if 0
    if (!traverse_stmts()) {
        return false;
    }
#endif
#if 0
    if (!test_custom()) {
        return false;
    }
#endif
#if 0
    if (!run_tests()) {
        return false;
    }
#endif
    return true;
}

bool Query::query_kernel_params() {
    m_kernel_params.clear();
    auto matcher = 
        functionDecl(
            hasName("kernel"),
            forEachDescendant(
                parmVarDecl(hasType(qualType().bind("type"))).bind("param")));
    auto results = m_matcher_tool.match(matcher);
    auto select_name = cat(name("param"));
    auto select_type = describe("type");
    for (auto &result: results) {
        std::string name;
        if (!m_matcher_tool.eval_stencil(select_name, result, name)) {
            return false;
        }
        std::string type;
        if (!m_matcher_tool.eval_stencil(select_type, result, type)) {
            return false;
        }
        DataType data_type;
        if (!parse_kernel_param_type(type, data_type)) {
            error("Invalid kernel parameter type: " + type);
            return false;
        }
        m_kernel_params.emplace_back(name, data_type);
    }
    return true;
}

bool Query::traverse_stmts() {
    auto matcher =
        traverse(TK_IgnoreUnlessSpelledInSource, 
            stmt(
                hasAncestor(functionDecl().bind("func")),
                optionally(hasParent(stmt().bind("parent")))
            ).bind("stmt"));
    auto results = m_matcher_tool.match(matcher);
    printf("[Stmts]\n");
    const FunctionDecl *curr_func = nullptr;
    for (auto &result: results) {
        SourceManager &source_manager = result.Context->getSourceManager();
        const FunctionDecl *func = result.Nodes.getNodeAs<FunctionDecl>("func");
        if (curr_func != func) {
            printf("[Func] %s\n", func->getNameAsString().c_str());
        }
        const Stmt *stmt = result.Nodes.getNodeAs<Stmt>("stmt");
        SourceLocation begin_loc = stmt->getBeginLoc();
        int begin_line = int(source_manager.getExpansionLineNumber(begin_loc));
        int begin_col = int(source_manager.getExpansionColumnNumber(begin_loc));
        const char *class_name = stmt->getStmtClassName();
        int64_t id = stmt->getID(*result.Context);
        const Stmt *parent = result.Nodes.getNodeAs<Stmt>("parent");
        int64_t parent_id = 0;
        if (parent != nullptr) {
            parent_id = parent->getID(*result.Context);
        }
        printf("At [%d:%d] %s %zd (parent %zd)\n", 
            begin_line, begin_col, class_name, size_t(id), size_t(parent_id));
    }
    return true;
}

bool Query::test_custom() {
    bool ok = true;
    printf("---- Custom 1\n");
    auto match_func1 = [](const Stmt &stmt, ASTContext &context) -> bool {
        return (stmt.getStmtClass() == Stmt::StmtClass::MemberExprClass);
    };
    ok &= eval_match(stmt(customMatch<Stmt>(match_func1)).bind("id"), node("id"));
    return ok;
}

bool Query::run_tests() {
    bool ok = true;
    std::string id("id");
    auto node_id = node(id);
#if 0
    printf("---- Query 1\n");
    ok &= eval_match(memberExpr().bind(id), node_id);
    printf("---- Query 2\n");
    ok &= eval_match(
        callExpr(
            callee(
                memberExpr()
            )
        ).bind(id),
        node_id);
    printf("---- Query 3\n");
    ok &= eval_match(
        cxxMemberCallExpr(
            on(
               expr(
                   hasType(
                       cxxRecordDecl(hasName("pipe"))
                   )
               )
            )
        ).bind(id),
        node_id);
    printf("---- Query 4\n");
    ok &= eval_match(
        cxxMemberCallExpr(
            on(
               expr(
                   hasType(
                       cxxRecordDecl(hasName("pipe"))
                   )
               )
            ),
            callee(
                cxxMethodDecl(hasName("wait_front"))
            )
        ).bind(id),
        node_id);
    printf("---- Query 5\n");
    ok &= eval_match(
        cxxMemberCallExpr(
            on(
               expr(
                   hasType(
                       cxxRecordDecl(hasName("pipe"))
                   )
               )
            ),
            callee(
                cxxMethodDecl(hasName("wait_front"))
            ),
            hasArgument(0, expr().bind("arg0"))
        ).bind(id),
        node("arg0"));
    printf("---- Query 6\n");
    ok &= eval_match(
        cxxMemberCallExpr(
            on(
               expr(
                   hasType(
                       cxxRecordDecl(hasName("pipe"))
                   )
               )
            ),
            callee(
                cxxMethodDecl(hasName("wait_front"))
            ),
            hasArgument(0, expr().bind("arg0"))
        ).bind(id),
        statement("id"));
#endif
    printf("---- Query 7\n");
    ok &= eval_match(
        declStmt(
            hasSingleDecl(varDecl(hasType(cxxRecordDecl(hasName("math"))))),
            hasParent(compoundStmt().bind("parent"))
        ).bind(id),
        statement("id"));
    printf("---- Query 8\n");
    ok &= eval_match(
        functionDecl(
            hasAnyParameter(parmVarDecl(
                hasType(cxxRecordDecl(hasName("math")))).bind("param"))
        ).bind("id"),
        node("param"));
    return ok;
}

bool Query::print_match_results(
        const std::vector<MatchResult> &results,
        RangeSelector selector) {
    bool ok = true;
    std::string text;
    for (auto &result: results) {
        ok &= m_matcher_tool.select_range(selector, result, text);
        printf("[%s]\n", text.c_str());
    }
    return ok;
}

void Query::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

