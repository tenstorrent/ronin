
#include <cstdint>
#include <cassert>

#include "whisper/riscv/riscv32.hpp"

#include "core/kernel_structs.hpp"
#include "core/dataflow_api.hpp"
#include "core/machine.hpp"

#include "riscv/builtin_dataflow_tanto.hpp"
#include "riscv/dataflow_tanto_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

namespace {

using ::riscv::core::Riscv32Core;

// Tanto extensions

void noc_async_read_global_dram(Dataflow *api, Riscv32Core *core) {
    uint32_t dst_addr = core->get_arg(0);
    uint32_t src_addr = core->get_arg(1);
    uint32_t src_log2_page_size = core->get_arg(2);
    uint32_t src_offset = core->get_arg(3);
    uint32_t len_bytes = core->get_arg(4);
    api->noc_async_read_global_dram(
        dst_addr,
        src_addr,
        src_log2_page_size,
        src_offset,
        len_bytes);
}

void noc_async_read_global_l1(Dataflow *api, Riscv32Core *core) {
    uint32_t dst_addr = core->get_arg(0);
    uint32_t src_addr = core->get_arg(1);
    uint32_t src_log2_page_size = core->get_arg(2);
    uint32_t src_offset = core->get_arg(3);
    uint32_t len_bytes = core->get_arg(4);
    api->noc_async_read_global_l1(
        dst_addr,
        src_addr,
        src_log2_page_size,
        src_offset,
        len_bytes);
}

void noc_async_write_global_dram(Dataflow *api, Riscv32Core *core) {
    uint32_t src_addr = core->get_arg(0);
    uint32_t dst_addr = core->get_arg(1);
    uint32_t dst_log2_page_size = core->get_arg(2);
    uint32_t dst_offset = core->get_arg(3);
    uint32_t len_bytes = core->get_arg(4);
    api->noc_async_write_global_dram(
        src_addr,
        dst_addr,
        dst_log2_page_size,
        dst_offset,
        len_bytes);
}

void noc_async_write_global_l1(Dataflow *api, Riscv32Core *core) {
    uint32_t src_addr = core->get_arg(0);
    uint32_t dst_addr = core->get_arg(1);
    uint32_t dst_log2_page_size = core->get_arg(2);
    uint32_t dst_offset = core->get_arg(3);
    uint32_t len_bytes = core->get_arg(4);
    api->noc_async_write_global_l1(
        src_addr,
        dst_addr,
        dst_log2_page_size,
        dst_offset,
        len_bytes);
}

} // namespace

//
//    DataflowTantoHandler
//

DataflowTantoHandler::DataflowTantoHandler(Machine *machine):
        m_machine(machine) { }

DataflowTantoHandler::~DataflowTantoHandler() { }

#define DECL_BUILTIN(name, count, result) \
    case DataflowTantoBuiltinId::name: \
        name(api, core); \
        break;

void DataflowTantoHandler::call(Riscv32Core *core, int id) {
    Dataflow *api = m_machine->get_dataflow_api();
    switch (DataflowTantoBuiltinId(id)) {
DATAFLOW_TANTO_BUILTINS
    default:
        assert(false);
        break;
    }
}

#undef DECL_BUILTIN

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

