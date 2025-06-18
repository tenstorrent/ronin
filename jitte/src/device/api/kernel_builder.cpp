// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>

#include "whisper/linker/linker.hpp"

#include "riscv/builtin_compute.hpp"
#include "riscv/builtin_compute_tanto.hpp"
#include "riscv/builtin_dataflow.hpp"
#include "riscv/builtin_dataflow_tanto.hpp"
#include "riscv/builtin_stdlib.hpp"

#include "api/kernel_builder.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

using ::riscv::linker::Linker;

std::string quote_define_value(const std::string &value) {
    // TODO: Replace this temporary placeholder with correct implementation
    bool must_quote = (value.find(" ") != std::string::npos);
    if (must_quote) {
        return "\"" + value + "\"";
    } else {
        return value;
    }
}

//
//    KernelLinker
//

class KernelLinker {
public:
    KernelLinker(bool is_compute);
    ~KernelLinker();
public:
    void link(
        const std::string &fname, 
        uint32_t code_base,
        std::vector<uint8_t> &result,
        uint32_t &start_pc);
private:
    void add_builtins();
private:
    bool m_is_compute;
    std::unique_ptr<Linker> m_linker;
    bool m_have_builtins;
};

KernelLinker::KernelLinker(bool is_compute):
        m_is_compute(is_compute),
        m_linker(Linker::create()),
        m_have_builtins(false) { }

KernelLinker::~KernelLinker() { }

void KernelLinker::link(
        const std::string &fname, 
        uint32_t code_base,
        std::vector<uint8_t> &result,
        uint32_t &start_pc) {
    // deferred 'add_builtins' to avoid timing condlict during global construction
    // (global KernelBuilder in 'tt_metal/emulator' constructed before builtin maps in 'riscv')
    if (!m_have_builtins) {
        add_builtins();
        m_have_builtins = true;
    }
    uint64_t code_base_u64 = uint64_t(code_base);
    uint64_t start_pc_u64 = 0;
    m_linker->link(fname, code_base_u64, result, start_pc_u64);
    start_pc = uint32_t(start_pc_u64);
}

void KernelLinker::add_builtins() {
    static constexpr uint32_t BUILTIN_MASK = uint64_t(1) << 30;
    if (m_is_compute) {
        for (auto &entry: get_compute_builtin_map()) {
            std::string name = entry.second.first;
            uint32_t id = uint32_t(entry.first);
            m_linker->add_builtin(name, BUILTIN_MASK | id);
        }
        for (auto &entry: get_compute_tanto_builtin_map()) {
            std::string name = entry.second.first;
            uint32_t id = uint32_t(entry.first);
            m_linker->add_builtin(name, BUILTIN_MASK | id);
        }
    } else {
        for (auto &entry: get_dataflow_builtin_map()) {
            std::string name = entry.second.name;
            uint32_t id = uint32_t(entry.first);
            m_linker->add_builtin(name, BUILTIN_MASK | id);
        }
        for (auto &entry: get_dataflow_tanto_builtin_map()) {
            std::string name = entry.second.name;
            uint32_t id = uint32_t(entry.first);
            m_linker->add_builtin(name, BUILTIN_MASK | id);
        }
    }
    for (auto &entry: get_stdlib_builtin_map()) {
        std::string name = entry.second.first;
        uint32_t id = uint32_t(entry.first);
        m_linker->add_builtin(name, BUILTIN_MASK | id);
    }
}

//
//    KernelBuilderImpl
//

class KernelBuilderImpl: public KernelBuilder {
public:
    KernelBuilderImpl();
    ~KernelBuilderImpl();
public:
    void configure(
        const std::string &cpp_cmd_base,
        const std::vector<std::pair<std::string, std::string>> &prefix_map,
        const std::string &src_base_dir,
        const std::string &temp_dir) override;
    void build(
        const std::string &name,
        bool is_compute,
        const std::string &defines,
        uint32_t code_base,
        std::vector<uint8_t> &code, 
        uint32_t &start_pc) override;
private:
    std::string make_obj_path();
    std::string make_cpp_cmd(
        const std::string &kernel_name,
        const std::string &defines,
        const std::string &obj_path);
    std::string map_kernel_name(const std::string &name);
    std::string make_compile_time_arg_name(int index);
    void run_cpp_cmd(const std::string &cpp_cmd);
private:
    KernelLinker m_compute_linker;
    KernelLinker m_dataflow_linker;
    std::string m_cpp_cmd_base;
    std::vector<std::pair<std::string, std::string>> m_prefix_map;
    std::string m_src_base_dir;
    std::string m_temp_dir;
};

KernelBuilderImpl::KernelBuilderImpl():
        m_compute_linker(true),
        m_dataflow_linker(false) { }

KernelBuilderImpl::~KernelBuilderImpl() { }

void KernelBuilderImpl::configure(
        const std::string &cpp_cmd_base,
        const std::vector<std::pair<std::string, std::string>> &prefix_map,
        const std::string &src_base_dir,
        const std::string &temp_dir) {
    m_cpp_cmd_base = cpp_cmd_base;
    m_prefix_map = prefix_map;
    m_src_base_dir = src_base_dir,
    m_temp_dir = temp_dir;
}

void KernelBuilderImpl::build(
        const std::string &name,
        bool is_compute,
        const std::string &defines,
        uint32_t code_base,
        std::vector<uint8_t> &code, 
        uint32_t &start_pc) {
    std::string obj_path = make_obj_path();
    std::string cpp_cmd = make_cpp_cmd(name, defines, obj_path);
    run_cpp_cmd(cpp_cmd);
    if (is_compute) {
        m_compute_linker.link(obj_path, code_base, code, start_pc);
    } else {
        m_dataflow_linker.link(obj_path, code_base, code, start_pc);
    }
}

std::string KernelBuilderImpl::make_obj_path() {
    return m_temp_dir + "kernel.o";
}

std::string KernelBuilderImpl::make_cpp_cmd(
        const std::string &kernel_name,
        const std::string &defines,
        const std::string &obj_path) {
    std::string file_name = map_kernel_name(kernel_name);
    return m_cpp_cmd_base + " -o " + obj_path + " " +
        defines + " " + m_src_base_dir + file_name;
}

std::string KernelBuilderImpl::map_kernel_name(const std::string &name) {
    std::string from;
    std::string to;
    size_t from_len = 0;
    bool found = false;
    for (auto &entry: m_prefix_map) {
        from = entry.first;
        from_len = from.size();
        if (name.substr(0, from_len) == from) {
            to = entry.second;
            found = true;
            break;
        }
    }
    if (!found) {
        return name;
    }
    return to + name.substr(from_len);
}

std::string KernelBuilderImpl::make_compile_time_arg_name(int index) {
    return "KERNEL_COMPILE_TIME_ARG_" + std::to_string(index);
}

void KernelBuilderImpl::run_cpp_cmd(const std::string &cpp_cmd) {
printf("@@@ ---- CPP_CMD [%s]\n", cpp_cmd.c_str());
    int stat = std::system(cpp_cmd.c_str());
    if (stat != 0) {
        throw std::runtime_error("Error in std::system, status " + std::to_string(stat));
    }
}

} // namespace

//
//    KernelBuilder
//

KernelBuilder *KernelBuilder::create() {
    return new KernelBuilderImpl();
}

} // namespace device
} // namespace metal
} // namespace tt

