// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace ronin {
namespace tanto {
namespace front {

class ErrorHandler {
public:
    ErrorHandler() { }
    ~ErrorHandler() { }
public:
    void reset() {
        m_errors.clear();
    }
    void error(const std::string text) {
        m_errors.push_back(text);
    }
    const std::vector<std::string> get_errors() {
        return m_errors;
    }
private:
    std::vector<std::string> m_errors;
};

} // namespace front
} // namespace tanto
} // namespace ronin

