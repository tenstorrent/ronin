// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

#include "llvm/Support/raw_ostream.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

#include "core/tooling.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;

//
//    MatcherTool
//

MatcherTool::MatcherTool():
        m_error_handler(nullptr) { }

MatcherTool::~MatcherTool() { }

void MatcherTool::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
}

bool MatcherTool::reset_code(const std::string &code) {
    m_ast_unit.reset(tooling::buildASTFromCode(code).release());
    if (m_ast_unit == nullptr) {
        error("AST construction failed");
        return false;
    }
    ASTContext &context = m_ast_unit->getASTContext();
    if (context.getDiagnostics().hasErrorOccurred()) {
        error("Compilation error");
        return false;
    }
    return true;
}

bool MatcherTool::select_range(
        RangeSelector selector, 
        const MatchResult &match_result, 
        std::string &text) {
    Expected<CharSourceRange> range = selector(match_result);
    if (!range) {
        error("Range selector error: " + llvm::toString(range.takeError()));
        return false;
    }
    text = tooling::getText(*range, *match_result.Context);
    return true;
}

bool MatcherTool::eval_stencil(
        const Stencil &stencil,
        const MatchResult &match_result, 
        std::string &text) {
    Expected<std::string> result = stencil->eval(match_result);
    if (!result) {
        error("Stencil error: " + llvm::toString(result.takeError()));
        return false;
    }
    text = result.get();
    return true;
}

void MatcherTool::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

//
//    TransformerTool
//

TransformerTool::TransformerTool():
        m_error_handler(nullptr),
        m_error_count(0) { }

TransformerTool::~TransformerTool() { }

void TransformerTool::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
}

bool TransformerTool::run(
        RewriteRule rule, 
        const std::string &input, 
        std::string &result) {
    m_error_count = 0;
    m_transformers.push_back(std::make_unique<Transformer>(std::move(rule), consumer()));
    m_transformers.back()->registerMatchers(&m_match_finder);
    return rewrite(input, result);
}

bool TransformerTool::run(
        RewriteRuleWith<std::string> rule,
        const std::string &input, 
        std::string &result) {
    m_error_count = 0;
    m_transformers.push_back(
        std::make_unique<Transformer>(
            std::move(rule), consumer_with_string_metadata()));
    m_transformers.back()->registerMatchers(&m_match_finder); 
    return rewrite(input, result);
}

Transformer::ChangeSetConsumer TransformerTool::consumer() {
    return [this](Expected<MutableArrayRef<AtomicChange>> c) {
        if (c) {
            m_changes.insert(
                m_changes.end(), 
                std::make_move_iterator(c->begin()),
                std::make_move_iterator(c->end()));
        } else {
            error("Error generating changes: " + llvm::toString(c.takeError()));
            m_error_count++;
        }
    }; 
}

std::function<void(Expected<TransformerResult<std::string>>)> 
        TransformerTool::consumer_with_string_metadata() {
    return [this](Expected<TransformerResult<std::string>> c) { 
        if (c) {
            m_changes.insert(
                m_changes.end(),
                std::make_move_iterator(c->Changes.begin()),
                std::make_move_iterator(c->Changes.end()));
            m_string_metadata.push_back(std::move(c->Metadata));
        } else {
            error("Error generating changes: " + llvm::toString(c.takeError()));
            m_error_count++;
        }
    }; 
}

bool TransformerTool::rewrite(const std::string &input, std::string &result) {
    auto factory = newFrontendActionFactory(&m_match_finder);
    if (!runToolOnCodeWithArgs(
            factory->create(), 
            input, 
            std::vector<std::string>(), 
            "input.cc",
            "clang-tool", 
            std::make_shared<PCHContainerOperations>(),
            m_file_contents)) {
        error("Running transformer tool failed");
        return false;
    }
    if (m_error_count != 0) {
        error("Generating changes failed");
        return false;
    }
    auto changed_code = applyAtomicChanges("input.cc", input, m_changes, ApplyChangesSpec());
    if (!changed_code) {
        error("Applying changes failed: " + llvm::toString(changed_code.takeError()));
        return false;
    }
    result = changed_code.get();
    return true;
}

void TransformerTool::add_file(const std::string &filename, const std::string &content) {
    m_file_contents.emplace_back(std::string(filename), std::string(content)); 
}

void TransformerTool::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

//
//    Code formatting
//

bool format_code(const std::string &input, std::string &result) {
    std::vector<Range> ranges(1, Range(0, input.size()));
    auto style = format::getLLVMStyle();
    const auto replacements = format::reformat(style, input, ranges);
    auto formatted = applyAllReplacements(input, replacements);
    if (!formatted) {
        result.clear();
        return false;
    }
    result = formatted.get();
    return true;
}

} // namespace front
} // namespace tanto
} // namespace ronin

