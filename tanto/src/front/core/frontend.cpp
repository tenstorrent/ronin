// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <utility>
#include <sstream>

#include "core/common.hpp"
#include "core/builtin.hpp"
#include "core/builtin_init.hpp"
#include "core/frontend.hpp"

/*
Mapping of standard objects

struct Global {
    uint32_t addr;
    uint32_t log2_page_size;
};
struct Local {
    uint32_t addr;
};
struct Pipe {
    uint32_t cb_id;
    uint32_t frame_size;
};
struct Semaphore {
    uint32_t addr;
};
*/

namespace ronin {
namespace tanto {
namespace front {

namespace {

std::string start_code_mark("// ---- CODE\n");

std::string remove_header(const std::string &code) {
    std::string result = code;
    size_t it = result.find(start_code_mark);
    if (it != std::string::npos) {
        result.erase(0, it + start_code_mark.size()); 
    }
    return result;
}

std::string remove_lf(const std::string &code) {
    std::stringstream result;
    size_t size = code.size();
    for (size_t i = 0; i < size; i++) {
        char ch = code[i];
        if (ch != '\r') {
            result << ch;
        }
    }
    return result.str();
}

} // namespace

//
//    Frontend
//

Frontend::Frontend() { 
    m_query.set_error_handler(&m_error_handler);
    m_transform.set_error_handler(&m_error_handler);
    m_dead_code_pass.set_error_handler(&m_error_handler);
    m_math_init_pass.set_error_handler(&m_error_handler);
}

Frontend::~Frontend() { }

void Frontend::add_define(const std::string &name, const std::string &value) {
    m_defines.emplace_back(name, value);
}

void Frontend::add_param(uint32_t index, uint32_t value) {
    m_params.emplace_back(index, value);
}

bool Frontend::compile(
        FrontendMode mode,
        const std::string &input_code, 
        std::string &output_code) {
    if (!setup_transform()) {
        return false;
    }
    if (mode == FrontendMode::COMPUTE) {
        return compile_compute(input_code, output_code);
    } else if (mode == FrontendMode::READ || mode == FrontendMode::WRITE) {
        bool write_mode = (mode == FrontendMode::WRITE);
        return compile_dataflow(input_code, output_code, write_mode);
    } else {
        assert(false);
    }
    return false;
}

bool Frontend::setup_transform() {
    m_transform.reset();
    for (std::pair<uint32_t, uint32_t> entry: m_params) {
        if (!m_transform.add_param(entry.first, entry.second)) {
            return false;
        }
    }
    return true;
}

bool Frontend::compile_compute(const std::string &input_code, std::string &output_code) {
    bool ok = true;
    std::string full_input = make_full_input(input_code, true);
    ok = m_query.run(full_input);
    if (!ok) {
        return false;
    }
    std::string pass_input;
    std::string pass_output;
    pass_input = full_input;
    ok = m_transform.pass1(pass_input, pass_output);
    if (!ok) {
        return false;
    }
    pass_input = pass_output;
    ok = m_dead_code_pass.run(pass_input, pass_output);
    if (!ok) {
        return false;
    }
    pass_input = pass_output;
    ok = m_math_init_pass.run(pass_input, pass_output);
    if (!ok) {
        return false;
    }
    pass_input = pass_output;
    ok = m_transform.pass2_compute(pass_input, pass_output);
    if (!ok) {
        return false;
    }
    pass_input = remove_header(pass_output);
    ok = finalize_compute(pass_input, pass_output);
    if (!ok) {
        return false;
    }
    ok = format_code(pass_output, output_code);
    if (!ok) {
        return false;
    }
    return true;
}

bool Frontend::compile_dataflow(
        const std::string &input_code, 
        std::string &output_code,
        bool write_mode) {
    bool ok = true;
    std::string full_input = make_full_input(input_code, false);
    ok = m_query.run(full_input);
    if (!ok) {
        return false;
    }
    std::string pass1_code;
    ok = m_transform.pass1(full_input, pass1_code);
    if (!ok) {
        return false;
    }
    std::string pass2_code;
    ok = m_transform.pass2_dataflow(pass1_code, pass2_code, write_mode);
    if (!ok) {
        return false;
    }
    pass2_code = remove_header(pass2_code);
    std::string final_code;
    ok = finalize_dataflow(pass2_code, final_code);
    if (!ok) {
        return false;
    }
    ok = format_code(final_code, output_code);
    if (!ok) {
        return false;
    }
    return true;
}

std::string Frontend::make_full_input(const std::string &input_code, bool with_init) {
    std::string result = get_builtin_header();
    result += "\n";
    if (with_init) {
        result += get_builtin_init_header();
        result += "\n";
    }
    result += build_defines();
    result += "\n";
    result += start_code_mark;
    result += "\n";
    result += remove_lf(input_code);
    return result;
}

bool Frontend::finalize_compute(const std::string &input_code, std::string &output_code) {
    std::string spdx_header;
    std::string input_nospdx;
    extract_spdx_header(input_code, spdx_header, input_nospdx);
    output_code.clear();
    output_code += spdx_header;
    output_code += "\n";
    output_code += "#include \"tanto/compute.h\"\n";
    output_code += "\n";
    output_code += build_defines();
    output_code += "\n";
    output_code += "namespace NAMESPACE {\n";
    output_code += input_nospdx;
    output_code += "void MAIN {\n";
    output_code += build_kernel_main_body(true);
    output_code += "}\n";
    output_code += "} // NAMESPACE\n";
    return true;
}

bool Frontend::finalize_dataflow(const std::string &input_code, std::string &output_code) {
    std::string spdx_header;
    std::string input_nospdx;
    extract_spdx_header(input_code, spdx_header, input_nospdx);
    output_code.clear();
    output_code += spdx_header;
    output_code += "\n";
    output_code += "#include \"tanto/dataflow.h\"\n";
    output_code += "\n";
    output_code += build_defines();
    output_code += "\n";
    output_code += input_nospdx;
    output_code += "void kernel_main() {\n";
    output_code += build_kernel_main_body(false);
    output_code += "}\n";
    return true;
}

void Frontend::extract_spdx_header(
        const std::string &input_code, 
        std::string &spdx_header, 
        std::string &input_nospdx) {
    size_t size = input_code.size();
    size_t pos = 0;
    while (pos < size && input_code[pos] == '\n') {
        pos++;
    }
    uint32_t start = pos;
    if (input_code.find("// SPDX-", pos) != pos) {
        spdx_header = "";
        input_nospdx = input_code;
        return;
    }
    for ( ; ; ) {
        size_t eol = input_code.find('\n', pos);
        if (eol == std::string::npos) {
            pos = size;
            break;
        }
        pos = eol + 1;
        if (pos + 1 >= size || input_code[pos] != '/' || input_code[pos + 1] != '/') {
            break;
        }
    }
    if (pos >= size) {
        // must not happen
        spdx_header = input_code;
        input_nospdx = "";
    } else {
        spdx_header = input_code.substr(start, pos);
        input_nospdx = input_code.substr(pos);
    }
}

std::string Frontend::build_kernel_main_body(bool compute) {
    std::stringstream result;
    int k = 0;
    int count = m_query.kernel_param_count();
    for (int i = 0; i < count; i++) {
        std::string name = m_query.kernel_param_name(i);
        DataType type = m_query.kernel_param_type(i);
        switch (type) {
        case DataType::INT32:
            result << "int32 " << name << " = get_arg_val<int32>(" << k << ");\n";
            k++;
            break;
        case DataType::UINT32:
            result << "uint32 " << name << " = get_arg_val<uint32>(" << k << ");\n";
            k++;
            break;
        case DataType::FLOAT:
            result << "float " << name << " = get_arg_val<float>(" << k << ");\n";
            k++;
            break;
        case DataType::GLOBAL:
            result << "Global " << name << ";\n";
            result << name << ".addr = get_arg_val<uint32>(" << k << ");\n";
            result << name << ".log2_page_size = get_arg_val<uint32>(" << (k + 1) << ");\n";
            k += 2;
            break;
        case DataType::LOCAL:
            result << "Local " << name << ";\n";
            result << name << ".addr = get_arg_val<uint32>(" << k << ");\n";
            k++;
            break;
        case DataType::SEMAPHORE:
            result << "Semaphore " << name << ";\n";
#if 0 // TODO: Revise this
            result << name << ".addr = get_arg_val<uint32>(" << k << ");\n";
#endif
            result << name << ".addr = tanto_get_semaphore(get_arg_val<uint32>(" << k << "));\n";
            k++;
            break;
        case DataType::PIPE:
            result << "Pipe " << name << ";\n";
            result << name << ".cb_id = get_arg_val<uint32>(" << k << ");\n";
            result << name << ".frame_size = get_arg_val<uint32>(" << (k + 1) << ");\n";
            k += 2;
            break;
        default:
            assert(false);
            break;
        }
    }
    // It is much easier to insert global compute initialization here
    // rather than introduce separate rule and then propagate call through passes 
    if (compute) {
        result << "tanto_compute_init();\n";
    }
    result << "kernel(";
    for (int i = 0; i < count; i++) {
        std::string name = m_query.kernel_param_name(i);
        if (i != 0) {
            result << ", ";
        }
        result << name;
    }
    result << ");\n";
    return result.str();
}

std::string Frontend::build_defines() {
    std::stringstream result;
    for (auto &entry: m_defines) {
        result << "#define " << entry.first << " " << entry.second << "\n";
    }
    return result.str();
}

} // namespace front
} // namespace tanto
} // namespace ronin

