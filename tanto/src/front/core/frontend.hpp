// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "core/common.hpp"
#include "core/error.hpp"
#include "core/query.hpp"
#include "core/transform.hpp"
#include "core/dead_code_pass.hpp"
#include "core/math_init_pass.hpp"

namespace ronin {
namespace tanto {
namespace front {

class Frontend {
public:
    Frontend();
    ~Frontend();
public:
    void add_define(const std::string &name, const std::string &value);
    void add_param(uint32_t index, uint32_t value);
    bool compile(
        FrontendMode mode,
        const std::string &input_code, 
        std::string &output_code);
    const std::vector<std::string> get_errors() {
        return m_error_handler.get_errors();
    }
private:
    bool setup_transform();
    bool compile_compute(const std::string &input_code, std::string &output_code);
    bool compile_dataflow(
        const std::string &input_code, 
        std::string &output_code,
        bool write_mode);
    std::string make_full_input(const std::string &input_code, bool with_init);
    bool finalize_compute(const std::string &input_code, std::string &output_code);
    bool finalize_dataflow(const std::string &input_code, std::string &output_code);
    void extract_spdx_header(
        const std::string &input_code, 
        std::string &spdx_header, 
        std::string &input_nospdx);
    std::string build_kernel_main_body(bool compute);
    std::string build_defines();
    void error(const std::string text) {
        m_error_handler.error(text);
    }
private:
    ErrorHandler m_error_handler;
    std::vector<std::pair<std::string, std::string>> m_defines;
    std::vector<std::pair<uint32_t, uint32_t>> m_params;
    Query m_query;
    Transform m_transform;
    DeadCodePass m_dead_code_pass;
    MathInitPass m_math_init_pass;
};

} // namespace front
} // namespace tanto
} // namespace ronin

