// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <functional>
#include <cstdint>

#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

#include "core/matchers.hpp"
#include "core/rules.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace transformer;

//
//    RuleFactory
//

RuleFactory::RuleFactory():
        m_write_mode(false) { }

RuleFactory::~RuleFactory() { }

void RuleFactory::set_write_mode(bool write_mode) {
    m_write_mode = write_mode;
}

// style

RewriteRule RuleFactory::make_tidy_if_stmt_rule() {
    // ACHTUNG: Add braces to bodies of other relevant control flow statements
    auto match1 =
        ifStmt(
            hasThen(stmt(unless(compoundStmt())).bind("then")),
            hasElse(stmt(unless(compoundStmt())).bind("else")));
    auto match2 = ifStmt(hasThen(stmt(unless(compoundStmt())).bind("then")));
    auto match3 = ifStmt(hasElse(stmt(unless(compoundStmt())).bind("else")));
    return applyFirst({
        makeRule(
            match1,
            {
                insertBefore(statement("then"), cat("{")),
                insertAfter(statement("then"), cat("}")),
                insertBefore(statement("else"), cat("{")),
                insertAfter(statement("else"), cat("}"))
            }),
        makeRule(
            match2,
            {
                insertBefore(statement("then"), cat("{")),
                insertAfter(statement("then"), cat("}"))
            }),
        makeRule(
            match3,
            {
                insertBefore(statement("else"), cat("{")),
                insertAfter(statement("else"), cat("}"))
            })
        });
}

RewriteRule RuleFactory::make_cleanup_if_stmt_rule() {
    return makeRule(
        ifStmt(
            hasElse(
                compoundStmt(
                    statementCountIs(1),
                    hasAnySubstatement(ifStmt())
                ).bind("stmt")
            )),
        {
            // remove "stmt" outer braces
            remove(between(before(statement("stmt")), before(statements("stmt")))),
            remove(between(after(statements("stmt")), after(statement("stmt"))))
        }
    );
}

// top level

RewriteRule RuleFactory::make_param_rule(std::function<uint32_t ()> get_value) {
    auto wrap_get_value = [get_value](const MatchFinder::MatchResult &) -> Expected<std::string> {
        return std::to_string(get_value());
    };
    return makeRule(
        varDecl(
            allOf(hasGlobalStorage(), unless(isStaticLocal())),
            hasType(cxxRecordDecl(hasName("param")))
        ).bind("decl"),
        changeTo(
            node("decl"),
            cat("static constexpr ", "uint32", " ", name("decl"), " = ", 
                "uint32(", run(wrap_get_value), ");")
        ));
}

// functions

RewriteRule RuleFactory::make_parm_global_rule() {
    return makeRule(
        parmVarDecl(hasType(cxxRecordDecl(hasName("global")))).bind("param"),
        changeTo(
            node("param"),
            cat("Global ", name("param"))));
}

RewriteRule RuleFactory::make_parm_local_rule() {
    return makeRule(
        parmVarDecl(hasType(cxxRecordDecl(hasName("local")))).bind("param"),
        changeTo(
            node("param"),
            cat("Local ", name("param"))));
}

RewriteRule RuleFactory::make_parm_semaphore_rule() {
    return makeRule(
        parmVarDecl(hasType(cxxRecordDecl(hasName("semaphore")))).bind("param"),
        changeTo(
            node("param"),
            cat("Semaphore ", name("param"))));
}

RewriteRule RuleFactory::make_parm_pipe_rule() {
    return makeRule(
        parmVarDecl(hasType(cxxRecordDecl(hasName("pipe")))).bind("param"),
        changeTo(
            node("param"),
            cat("Pipe ", name("param"))));
}

RewriteRule RuleFactory::make_parm_math_rule() {
    return makeRule(
        parmVarDecl(hasType(cxxRecordDecl(hasName("math")))).bind("param"),
        remove(node("param")));
}

// common

RewriteRule RuleFactory::make_pipe_set_frame_rule() {
    // self.set_frame(tiles)
    //     =>
    // self.frame_size = tiles;
    return makeRule(
        make_member_call_1_matcher("pipe", "set_frame"),
        changeTo(
            statement("stmt"),
            cat(access("self", "frame_size"), " = ", expression("arg0"), ";")));
}

RewriteRule RuleFactory::make_pipe_wait_front_rule() {
    // self.wait_front();
    //     =>
    // cb_wait_front(self.cb_id, self.frame_size);
    return makeRule(
        make_member_call_0_matcher("pipe", "wait_front"),
        changeTo(
            statement("stmt"), 
            cat(
                "cb_wait_front(", 
                    access("self", "cb_id"), ", ", 
                    access("self", "frame_size"), ");")));
}

RewriteRule RuleFactory::make_pipe_pop_front_rule() {
    // self.pop_front();
    //     =>
    // cb_wait_front(self.cb_id, self.frame_size);
    return makeRule(
        make_member_call_0_matcher("pipe", "pop_front"),
        changeTo(
            statement("stmt"), 
            cat(
                "cb_pop_front(", 
                    access("self", "cb_id"), ", ",
                    access("self", "frame_size"), ");")));
}

RewriteRule RuleFactory::make_pipe_reserve_back_rule() {
    // self.reserve_back();
    //     =>
    // cb_reserve_back(self.cb_id, self.frame_size);
    return makeRule(
        make_member_call_0_matcher("pipe", "reserve_back"),
        changeTo(
            statement("stmt"), 
            cat(
                "cb_reserve_back(", 
                    access("self", "cb_id"), ", ", 
                    access("self", "frame_size"), ");")));
}

RewriteRule RuleFactory::make_pipe_push_back_rule() {
    // self.push_back();
    //     =>
    // cb_push_back(self.cb_id, self.frame_size);
    return makeRule(
        make_member_call_0_matcher("pipe", "push_back"),
        changeTo(
            statement("stmt"), 
            cat(
                "cb_push_back(", 
                    access("self", "cb_id"), ", ",
                    access("self", "frame_size"), ");")));
}

} // namespace front
} // namespace tanto
} // namespace ronin

