// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <cstdint>

#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "core/matchers.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace ast_matchers;

namespace {

auto make_data_type_matcher() {
    return anyOf(
        qualType(asString("unsigned int")).bind("T_uint32"),
        qualType(asString("float")).bind("T_float"),
        recordType(hasDeclaration(cxxRecordDecl(hasName("bfloat16")))).bind("T_bfloat16")
    );
} 

auto make_dist_matcher() {
    return anyOf(
        templateArgument(equalsIntegralValue("0")).bind("DIST_linear"),
        templateArgument(equalsIntegralValue("1")).bind("DIST_block"),
        templateArgument(equalsIntegralValue("2")).bind("DIST_cyclic")
    );
}

auto make_dram_matcher() {
    return anyOf(
        templateArgument(equalsIntegralValue("0")).bind("DRAM_false"),
        templateArgument(equalsIntegralValue("1")).bind("DRAM_true")
    );
}

} // namespace

//
//    Matcher factories
//

// function calls

StatementMatcher make_func_call_0_matcher(const std::string &func_name) {
    // <stmt> ::= func();
    return callExpr(
        callee(functionDecl(hasName(func_name))),
        argumentCountIs(0)
    ).bind("stmt");
}

StatementMatcher make_func_call_1_matcher(const std::string &func_name) {
    // <stmt> ::= func(<arg0>);
    return callExpr(
        callee(functionDecl(hasName(func_name))),
        argumentCountIs(1),
        hasArgument(0, expr().bind("arg0"))
    ).bind("stmt");
}

StatementMatcher make_func_call_2_matcher(const std::string &func_name) {
    // <stmt> ::= func(<arg0>);
    return callExpr(
        callee(functionDecl(hasName(func_name))),
        argumentCountIs(2),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1"))
    ).bind("stmt");
}

StatementMatcher make_func_call_3_matcher(const std::string &func_name) {
    // <stmt> ::= func(<arg0>);
    return callExpr(
        callee(functionDecl(hasName(func_name))),
        argumentCountIs(3),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2"))
    ).bind("stmt");
}

// member calls

StatementMatcher make_member_call_0_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method();
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(0)
    ).bind("stmt");
}

StatementMatcher make_member_call_1_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(1),
        hasArgument(0, expr().bind("arg0"))
    ).bind("stmt");
}

StatementMatcher make_member_call_2_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(2),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1"))
    ).bind("stmt");
}

StatementMatcher make_member_call_3_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(3),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2"))
    ).bind("stmt");
}

StatementMatcher make_member_call_4_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(4),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3"))
    ).bind("stmt");
}

StatementMatcher make_member_call_5_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>, <arg4>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(5),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4"))
    ).bind("stmt");
}

StatementMatcher make_member_call_6_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>, <arg4>, <arg5>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(6),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4")),
        hasArgument(5, expr().bind("arg5"))
    ).bind("stmt");
}

StatementMatcher make_member_call_7_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>, <arg4>, <arg5>, <arg6>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(7),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4")),
        hasArgument(5, expr().bind("arg5")),
        hasArgument(6, expr().bind("arg6"))
    ).bind("stmt");
}

StatementMatcher make_member_call_8_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>, <arg4>, <arg5>, <arg6>, <arg7>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(8),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4")),
        hasArgument(5, expr().bind("arg5")),
        hasArgument(6, expr().bind("arg6")),
        hasArgument(7, expr().bind("arg7"))
    ).bind("stmt");
}

StatementMatcher make_member_call_9_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>, <arg4>, <arg5>, <arg6>, <arg7>, <arg8>);
    return cxxMemberCallExpr(
        on(expr(hasType(cxxRecordDecl(hasName(self_type)))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(9),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4")),
        hasArgument(5, expr().bind("arg5")),
        hasArgument(6, expr().bind("arg6")),
        hasArgument(7, expr().bind("arg7")),
        hasArgument(8, expr().bind("arg8"))
    ).bind("stmt");
}

//
//    Special matchers capturing "typename T" of "self" template class
//

StatementMatcher make_member_call_1_with_t_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>);
    auto t = make_data_type_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(1),
        hasArgument(0, expr().bind("arg0"))
    ).bind("stmt");
}

StatementMatcher make_member_call_2_with_t_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>);
    auto t = make_data_type_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(2),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1"))
    ).bind("stmt");
}

StatementMatcher make_member_call_4_with_t_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>);
    auto t = make_data_type_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(4),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr().bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3"))
    ).bind("stmt");
}

//
//    Special matchers capturing "typename T" of "self" and matching type of "arg1"
//

StatementMatcher make_member_call_3_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>);
    auto t = make_data_type_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(3),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr(
            hasType(cxxRecordDecl(hasName(arg1_type)))
        ).bind("arg1")),
        hasArgument(2, expr().bind("arg2"))
    ).bind("stmt");
}

StatementMatcher make_member_call_4_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>);
    auto t = make_data_type_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(4),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr(
            hasType(cxxRecordDecl(hasName(arg1_type)))
        ).bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3"))
    ).bind("stmt");
}

StatementMatcher make_member_call_6_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>);
    auto t = make_data_type_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(6),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr(
            hasType(cxxRecordDecl(hasName(arg1_type)))
        ).bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4")),
        hasArgument(5, expr().bind("arg5"))
    ).bind("stmt");
}

StatementMatcher make_member_call_9_with_t_arg1_matcher(
        const std::string &self_type,
        const std::string &method_name,
        const std::string &arg1_type) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>);
    auto t = make_data_type_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(9),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr(
            hasType(cxxRecordDecl(hasName(arg1_type)))
        ).bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4")),
        hasArgument(5, expr().bind("arg5")),
        hasArgument(6, expr().bind("arg6")),
        hasArgument(7, expr().bind("arg7")),
        hasArgument(8, expr().bind("arg8"))
    ).bind("stmt");
}

//
//    Special matchers capturing "typename T" of "self" and "DIST" / "DRAM" of "arg1"
//

StatementMatcher make_member_call_4_with_t_dist_dram_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>);
    auto t = make_data_type_matcher();
    auto dist = make_dist_matcher();
    auto dram = make_dram_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(4),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr(
            hasType(classTemplateSpecializationDecl(
                hasTemplateArgument(1, dist))),
            hasType(classTemplateSpecializationDecl(
                hasTemplateArgument(2, dram)))
        ).bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3"))
    ).bind("stmt");
}

StatementMatcher make_member_call_5_with_t_dist_dram_matcher(
        const std::string &self_type,
        const std::string &method_name) {
    // <stmt> ::= <self>.method(<arg0>, <arg1>, <arg2>, <arg3>, <arg4>);
    auto t = make_data_type_matcher();
    auto dist = make_dist_matcher();
    auto dram = make_dram_matcher();
    return cxxMemberCallExpr(
        on(expr(hasType(classTemplateSpecializationDecl(
            hasName(self_type),
            hasTemplateArgument(0, templateArgument(refersToType(t)))
        ))).bind("self")),
        callee(cxxMethodDecl(hasName(method_name))),
        argumentCountIs(5),
        hasArgument(0, expr().bind("arg0")),
        hasArgument(1, expr(
            hasType(classTemplateSpecializationDecl(
                hasTemplateArgument(1, dist))),
            hasType(classTemplateSpecializationDecl(
                hasTemplateArgument(2, dram)))
        ).bind("arg1")),
        hasArgument(2, expr().bind("arg2")),
        hasArgument(3, expr().bind("arg3")),
        hasArgument(4, expr().bind("arg4"))
    ).bind("stmt");
}

} // namespace front
} // namespace tanto
} // namespace ronin

