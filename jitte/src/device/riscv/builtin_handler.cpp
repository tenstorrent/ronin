// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/machine.hpp"

#include "riscv/compute_handler.hpp"
#include "riscv/compute_tanto_handler.hpp"
#include "riscv/dataflow_handler.hpp"
#include "riscv/dataflow_tanto_handler.hpp"
#include "riscv/stdlib_handler.hpp"
#include "riscv/builtin_compute.hpp"
#include "riscv/builtin_compute_tanto.hpp"
#include "riscv/builtin_dataflow.hpp"
#include "riscv/builtin_dataflow_tanto.hpp"
#include "riscv/builtin_stdlib.hpp"
#include "riscv/builtin_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

static constexpr bool DIAG_REPORT_CALL_ENABLED = false;

namespace {

void report_call(Riscv32Core *core, const std::string &name, int count) {
    if (!DIAG_REPORT_CALL_ENABLED) {
        return;
    }
    printf("[CALL] %s(", name.c_str());
    for (int i = 0; i < count; i++) {
        if (i != 0) {
            printf(", ");
        }
        printf("%d", core->get_arg(i));
    }
    printf(")\n");
}

} // namespace

//
//    BuiltinHandler
//

BuiltinHandler::BuiltinHandler(Machine *machine):
        m_compute_handler(machine),
        m_compute_tanto_handler(machine),
        m_dataflow_handler(machine),
        m_dataflow_tanto_handler(machine),
        m_stdlib_handler(machine) { }

BuiltinHandler::~BuiltinHandler() { }

void BuiltinHandler::call(Riscv32Core *core, int id) {
    auto &compute_builtin_map = get_compute_builtin_map();
    auto it_compute = compute_builtin_map.find(ComputeBuiltinId(id));
    if (it_compute != compute_builtin_map.end()) {
        auto &entry = it_compute->second;
        std::string name = entry.first;
        int count = entry.second;
        report_call(core, name, count);
        m_compute_handler.call(core, id);
        return;
    }
    auto &compute_tanto_builtin_map = get_compute_tanto_builtin_map();
    auto it_compute_tanto = compute_tanto_builtin_map.find(ComputeTantoBuiltinId(id));
    if (it_compute_tanto != compute_tanto_builtin_map.end()) {
        auto &entry = it_compute_tanto->second;
        std::string name = entry.first;
        int count = entry.second;
        report_call(core, name, count);
        m_compute_tanto_handler.call(core, id);
        return;
    }
    auto &dataflow_builtin_map = get_dataflow_builtin_map();
    auto it_dataflow = dataflow_builtin_map.find(DataflowBuiltinId(id));
    if (it_dataflow != dataflow_builtin_map.end()) {
        auto &entry = it_dataflow->second;
        std::string name = entry.name;
        int count = entry.count;
        report_call(core, name, count);
        m_dataflow_handler.call(core, id);
        return;
    }
    auto &dataflow_tanto_builtin_map = get_dataflow_tanto_builtin_map();
    auto it_dataflow_tanto = dataflow_tanto_builtin_map.find(DataflowTantoBuiltinId(id));
    if (it_dataflow_tanto != dataflow_tanto_builtin_map.end()) {
        auto &entry = it_dataflow_tanto->second;
        std::string name = entry.name;
        int count = entry.count;
        report_call(core, name, count);
        m_dataflow_tanto_handler.call(core, id);
        return;
    }
    auto &stdlib_builtin_map = get_stdlib_builtin_map();
    auto it_stdlib = stdlib_builtin_map.find(StdlibBuiltinId(id));
    if (it_stdlib != stdlib_builtin_map.end()) {
        auto &entry = it_stdlib->second;
        std::string name = entry.first;
        int count = entry.second;
        report_call(core, name, count);
        m_stdlib_handler.call(core, id);
        return;
    }
    // TODO: Implement coroutine-friendly error handling
    fprintf(stderr, "[ERROR] Unsupported builtin ID: %d\n", id);
    exit(1);
}

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

