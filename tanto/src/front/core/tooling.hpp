// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "clang/Frontend/ASTUnit.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Transformer/Transformer.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

#include "core/error.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace tooling;
using namespace transformer;
using namespace ast_matchers;

using MatchResult = MatchFinder::MatchResult;

template<typename T>
class CustomMatcher: public internal::MatcherInterface<T> {
public:
    explicit CustomMatcher(std::function<bool (const T &, ASTContext &)> match_func):
        m_match_func(match_func) { }
    ~CustomMatcher() { }
public:
    bool matches(
            const T &node,
            internal::ASTMatchFinder *finder,
            internal::BoundNodesTreeBuilder *builder) const override {
        return m_match_func(node, finder->getASTContext());
    }
private:
    std::function<bool (const T &, ASTContext &)> m_match_func;
};

template<typename T>
inline internal::Matcher<T> customMatch(
        std::function<bool (const T &, ASTContext &)> match_func) {
    return internal::Matcher<T>(new CustomMatcher<T>(match_func));
} 

//
//    MatcherTool
//

class MatcherTool {
public:
    MatcherTool();
    ~MatcherTool();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool reset_code(const std::string &code);
    template<typename M> 
    std::vector<MatchResult> match(M matcher) {
        std::vector<MatchResult> result;
        ASTContext &context = m_ast_unit->getASTContext();
        TraversalKindScope scope(context, TK_IgnoreUnlessSpelledInSource);
        auto matches = ast_matchers::match(matcher, context);
        for (auto match: matches) {
            result.push_back(MatchResult(match, &context));
        }
        return result;
    }
    bool select_range(
        RangeSelector selector, 
        const MatchResult &match_result, 
        std::string &text);
    bool eval_stencil(
        const Stencil &stencil,
        const MatchResult &match_result, 
        std::string &text);
private:
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    std::unique_ptr<clang::ASTUnit> m_ast_unit;
};

//
//    TransformerTool
//

class TransformerTool {
public:
    TransformerTool();
    ~TransformerTool();
public:
    void set_error_handler(ErrorHandler *error_handler);
    bool run(
        RewriteRule rule, 
        const std::string &input, 
        std::string &result);
    bool run(
        RewriteRuleWith<std::string> rule,
        const std::string &input, 
        std::string &result);
    Transformer::ChangeSetConsumer consumer();
    std::function<void(Expected<TransformerResult<std::string>>)> consumer_with_string_metadata();
    bool rewrite(const std::string &input, std::string &result);
    void add_file(const std::string &filename, const std::string &content);
private:
    void error(const std::string &text);
private:
    ErrorHandler *m_error_handler;
    int m_error_count;
    // Transformers are referenced by MatchFinder.
    std::vector<std::unique_ptr<Transformer>> m_transformers;
    clang::ast_matchers::MatchFinder m_match_finder;
    AtomicChanges m_changes;
    std::vector<std::string> m_string_metadata; 
    FileContentMappings m_file_contents; 
};

//
//    Code formatting
//

bool format_code(const std::string &input, std::string &result);

} // namespace front
} // namespace tanto
} // namespace ronin

