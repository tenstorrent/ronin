
#include <cstdint>
#include <cassert>

#include "whisper/riscv/riscv32.hpp"

#include "core/kernel_structs.hpp"
#include "core/dataflow_api.hpp"
#include "core/machine.hpp"

#include "riscv/builtin_dataflow.hpp"
#include "riscv/dataflow_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

namespace {

using ::riscv::core::Riscv32Core;

uint32_t get_hi32(uint64_t value) {
    return uint32_t(value >> 32);
}

uint32_t get_lo32(uint64_t value) {
    return uint32_t(value & 0xffffffff);
}

uint64_t make_u64(uint32_t lo, uint32_t hi) {
    return ((uint64_t(hi) << 32) | uint64_t(lo));
}
uint64_t get_arg64(Riscv32Core *core, int index) {
    return make_u64(core->get_arg(index), core->get_arg(index + 1));
}

void set_ret64(Riscv32Core *core, int index, uint64_t value) {
    core->set_ret(index, get_lo32(value));
    core->set_ret(index + 1, get_hi32(value));
}

uint32_t *get_arg_ptr32(Riscv32Core *core, int index) {
    return reinterpret_cast<uint32_t *>(core->map_addr(core->get_arg(index)));
}

void get_arg_uint32(Dataflow *api, Riscv32Core *core) {
    int arg_idx = int(core->get_arg(0));
    uint32_t ret = api->get_arg_uint32(arg_idx);
    core->set_ret(0, ret);
}

void cb_push_back(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t num_pages = core->get_arg(1);
    api->cb_push_back(operand, num_pages);
}

void cb_pop_front(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t num_pages = core->get_arg(1);
    api->cb_pop_front(operand, num_pages);
}

void get_tile_size(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t ret = api->get_tile_size(operand);
    core->set_ret(0, ret);
}

void get_dataformat(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    DataFormat ret = api->get_dataformat(operand);
    core->set_ret(0, uint32_t(ret));
}

void get_write_ptr(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t ret = api->get_write_ptr(operand);
    core->set_ret(0, ret);
}

void get_read_ptr(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t ret = api->get_read_ptr(operand);
    core->set_ret(0, ret);
}

void wait_for_sync_register_value(Dataflow *api, Riscv32Core *core) {
    uint32_t addr = core->get_arg(0);
    int32_t val = int32_t(core->get_arg(1));
    api->wait_for_sync_register_value(addr, val);
}

void cb_reserve_back(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t num_pages = core->get_arg(1);
    api->cb_reserve_back(operand, num_pages);
}

void cb_wait_front(Dataflow *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t num_pages = core->get_arg(1);
    api->cb_wait_front(operand, num_pages);
}

// NOC transfers
// simple APIs

void get_noc_multicast_addr(Dataflow *api, Riscv32Core *core) {
    uint32_t noc_x_start = core->get_arg(0);
    uint32_t noc_y_start = core->get_arg(1);
    uint32_t noc_x_end = core->get_arg(2);
    uint32_t noc_y_end = core->get_arg(3);
    uint32_t addr = core->get_arg(4);
    uint64_t ret = 
        api->get_noc_multicast_addr(
            noc_x_start,
            noc_y_start,
            noc_x_end,
            noc_y_end,
            addr);
    set_ret64(core, 0, ret);
}

void get_noc_addr_remote(Dataflow *api, Riscv32Core *core) {
    uint32_t noc_x = core->get_arg(0);
    uint32_t noc_y = core->get_arg(1);
    uint32_t addr = core->get_arg(2);
    uint64_t ret = api->get_noc_addr_remote(noc_x, noc_y, addr);
    set_ret64(core, 0, ret);
}

void get_noc_addr_helper(Dataflow *api, Riscv32Core *core) {
    uint32_t noc_xy = core->get_arg(0);
    uint32_t addr = core->get_arg(1);
    uint64_t ret = api->get_noc_addr_helper(noc_xy, addr);
    set_ret64(core, 0, ret);
}

void get_dram_noc_addr(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    uint32_t page_size = core->get_arg(1);
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t offset = core->get_arg(3);
    uint64_t ret = 
        api->get_dram_noc_addr(
            id, 
            page_size, 
            bank_base_address, 
            offset);
    set_ret64(core, 0, ret);
}

void get_l1_noc_addr(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    uint32_t page_size = core->get_arg(1);
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t offset = core->get_arg(3);
    uint64_t ret =
        api->get_l1_noc_addr(
            id, 
            page_size, 
            bank_base_address, 
            offset);
    set_ret64(core, 0, ret);
}

void get_system_memory_noc_addr(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    uint32_t page_size = core->get_arg(1);
    uint32_t base_addr = core->get_arg(2);
    uint32_t offset = core->get_arg(3);
    uint64_t ret =
        api->get_system_memory_noc_addr(
            id, 
            page_size, 
            base_addr, 
            offset);
    set_ret64(core, 0, ret);
}

void get_noc_addr_interleaved(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t page_size = core->get_arg(3);
    uint32_t offset = core->get_arg(4);
    uint64_t ret = 
        api->get_noc_addr_interleaved(
            id, 
            dram,
            bank_base_address,
            page_size,
            offset);
    set_ret64(core, 0, ret);
}

void get_noc_addr_interleaved_pow2(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t log_base_2_of_page_size = core->get_arg(3);
    uint32_t offset = core->get_arg(4);
    uint64_t ret =
        api->get_noc_addr_interleaved_pow2(
            id, 
            dram,
            bank_base_address,
            log_base_2_of_page_size,
            offset);
    set_ret64(core, 0, ret);
}

void get_noc_addr_interleaved_fast(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0); 
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t page_size = core->get_arg(3);
    DataFormat data_format = DataFormat(core->get_arg(4));
    uint32_t offset = core->get_arg(5);
    uint64_t ret =
        api->get_noc_addr_interleaved_fast(
            id, 
            dram,
            bank_base_address,
            page_size,
            data_format,
            offset);
    set_ret64(core, 0, ret);
}

void get_noc_addr_local(Dataflow *api, Riscv32Core *core) {
    uint32_t addr = core->get_arg(0);
    uint64_t ret = api->get_noc_addr_local(addr);
    set_ret64(core, 0, ret);
}

void noc_async_read(Dataflow *api, Riscv32Core *core) {
    uint64_t src_noc_addr = get_arg64(core, 0);
    uint32_t dst_local_l1_addr = core->get_arg(2);
    uint32_t size = core->get_arg(3);
    api->noc_async_read(
        src_noc_addr, 
        dst_local_l1_addr, 
        size);
}

void noc_async_read_one_packet(Dataflow *api, Riscv32Core *core) {
    uint64_t src_noc_addr = get_arg64(core, 0);
    uint32_t dst_local_l1_addr = core->get_arg(2);
    uint32_t size = core->get_arg(3);
    api->noc_async_read_one_packet(
        src_noc_addr, 
        dst_local_l1_addr, 
        size);
}

void noc_async_read_one_packet_set_state(Dataflow *api, Riscv32Core *core) {
    uint64_t src_noc_addr = get_arg64(core, 0);
    uint32_t size = core->get_arg(2);
    api->noc_async_read_one_packet_set_state(src_noc_addr, size);
}

void noc_async_read_one_packet_with_state(Dataflow *api, Riscv32Core *core) {
    uint32_t src_noc_addr = core->get_arg(0);
    uint32_t dst_local_l1_addr = core->get_arg(1);
    bool inc_num_issued = bool(core->get_arg(2));
    api->noc_async_read_one_packet_with_state(
        src_noc_addr, 
        dst_local_l1_addr,
        inc_num_issued);
}

void noc_async_read_set_state(Dataflow *api, Riscv32Core *core) {
    uint64_t src_noc_addr = get_arg64(core, 0);
    api->noc_async_read_set_state(src_noc_addr);
}

void noc_async_read_with_state(Dataflow *api, Riscv32Core *core) {
    uint32_t src_noc_addr = core->get_arg(0);
    uint32_t dst_local_l1_addr = core->get_arg(1);
    uint32_t size = core->get_arg(2);
    bool inc_num_issued = bool(core->get_arg(3));
    api->noc_async_read_with_state(
        src_noc_addr, 
        dst_local_l1_addr, 
        size,
        inc_num_issued);
}

void noc_async_read_inc_num_issued(Dataflow *api, Riscv32Core *core) {
    uint32_t num_issued_reads_inc = core->get_arg(0);
    api->noc_async_read_inc_num_issued(num_issued_reads_inc);
}

void noc_async_write_one_packet(Dataflow *api, Riscv32Core *core) {
    uint32_t src_local_l1_addr = core->get_arg(0);
    uint64_t dst_noc_addr = get_arg64(core, 1);
    uint32_t size = core->get_arg(3);
    api->noc_async_write_one_packet(
        src_local_l1_addr, 
        dst_noc_addr, 
        size);
}

void noc_async_write_one_packet_set_state(Dataflow *api, Riscv32Core *core) {
    uint64_t dst_noc_addr = get_arg64(core, 0);
    uint32_t size = core->get_arg(2);
    bool non_posted = bool(core->get_arg(3));
    api->noc_async_write_one_packet_set_state(
        dst_noc_addr, 
        size,
        non_posted);
}

void noc_async_write_one_packet_with_state(Dataflow *api, Riscv32Core *core) {
    uint32_t src_local_l1_addr = core->get_arg(0);
    uint32_t dst_noc_addr = core->get_arg(1);
    bool non_posted = bool(core->get_arg(2));
    api->noc_async_write_one_packet_with_state(
        src_local_l1_addr, 
        dst_noc_addr,
        non_posted);
}

void noc_async_read_tile(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t page_size = core->get_arg(3);
    DataFormat data_format = DataFormat(core->get_arg(4));
    uint32_t dst_local_l1_addr = core->get_arg(5);
    uint32_t offset = core->get_arg(6);
    api->noc_async_read_tile(
        id, 
        dram,
        bank_base_address,
        page_size,
        data_format,
        dst_local_l1_addr, 
        offset);
}

void noc_async_read_page(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t log_base_2_of_page_size = core->get_arg(3);
    uint32_t dest_addr = core->get_arg(4);
    uint32_t offset = core->get_arg(5);
    api->noc_async_read_page(
        id, 
        dram,
        bank_base_address,
        log_base_2_of_page_size,
        dest_addr, 
        offset);
}

void noc_async_read_partial_page(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t log_base_2_of_page_size = core->get_arg(3);
    uint32_t dest_addr = core->get_arg(4);
    uint32_t size = core->get_arg(5);
    uint32_t offset = core->get_arg(6);
    api->noc_async_read_partial_page(
        id, 
        dram,
        bank_base_address,
        log_base_2_of_page_size,
        dest_addr, 
        size, 
        offset);
}

void noc_async_write(Dataflow *api, Riscv32Core *core) {
    uint32_t src_local_l1_addr = core->get_arg(0);
    uint64_t dst_noc_addr = get_arg64(core, 1);
    uint32_t size = core->get_arg(3);
    api->noc_async_write(
        src_local_l1_addr, 
        dst_noc_addr, 
        size);
}

void noc_async_write_tile(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t page_size = core->get_arg(3);
    DataFormat data_format = DataFormat(core->get_arg(4));
    uint32_t src_local_l1_addr = core->get_arg(5);
    api->noc_async_write_tile(
        id, 
        dram,
        bank_base_address,
        page_size,
        data_format,
        src_local_l1_addr);
}

void noc_async_write_page(Dataflow *api, Riscv32Core *core) {
    uint32_t id = core->get_arg(0);
    bool dram = bool(core->get_arg(1));
    uint32_t bank_base_address = core->get_arg(2);
    uint32_t log_base_2_of_page_size = core->get_arg(3);
    uint32_t src_addr = core->get_arg(4);
    uint32_t write_size_bytes = core->get_arg(5);
    uint32_t offset = core->get_arg(6);
    api->noc_async_write_page(
        id, 
        dram,
        bank_base_address,
        log_base_2_of_page_size,
        src_addr, 
        write_size_bytes, 
        offset);
}

void get_semaphore(Dataflow *api, Riscv32Core *core) {
    uint32_t semaphore_id = core->get_arg(0);
    uint32_t ret = api->get_semaphore(semaphore_id);
    core->set_ret(0, ret);
}

void noc_semaphore_set_remote(Dataflow *api, Riscv32Core *core) {
    uint32_t src_local_l1_addr = core->get_arg(0);
    uint64_t dst_noc_addr = get_arg64(core, 1);
    api->noc_semaphore_set_remote(src_local_l1_addr, dst_noc_addr);
}

void noc_async_write_multicast(Dataflow *api, Riscv32Core *core) {
    uint32_t src_local_l1_addr = core->get_arg(0);
    uint64_t dst_noc_addr_multicast = get_arg64(core, 1);
    uint32_t size = core->get_arg(3);
    uint32_t num_dests = core->get_arg(4);
    api->noc_async_write_multicast(
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        num_dests);
}

void noc_semaphore_set_multicast(Dataflow *api, Riscv32Core *core) {
    uint32_t src_local_l1_addr = core->get_arg(0);
    uint64_t dst_noc_addr_multicast = get_arg64(core, 1);
    uint32_t num_dests = core->get_arg(3);
    api->noc_semaphore_set_multicast(
        src_local_l1_addr, 
        dst_noc_addr_multicast, 
        num_dests);
}

void noc_async_write_multicast_loopback_src(Dataflow *api, Riscv32Core *core) {
    uint32_t src_local_l1_addr = core->get_arg(0);
    uint64_t dst_noc_addr_multicast = get_arg64(core, 1);
    uint32_t size = core->get_arg(3);
    uint32_t num_dests = core->get_arg(4);
    api->noc_async_write_multicast_loopback_src(
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        num_dests);
}

void noc_async_read_barrier(Dataflow *api, Riscv32Core *core) {
    api->noc_async_read_barrier();
}

void noc_async_write_barrier(Dataflow *api, Riscv32Core *core) {
    api->noc_async_write_barrier();
}

void noc_semaphore_wait(Dataflow *api, Riscv32Core *core) {
    uint32_t *sem_addr = get_arg_ptr32(core, 0);
    uint32_t val = core->get_arg(1);
    api->noc_semaphore_wait(sem_addr, val);
}

void noc_semaphore_set(Dataflow *api, Riscv32Core *core) {
    uint32_t *sem_addr = get_arg_ptr32(core, 0);
    uint32_t val = core->get_arg(1);
    api->noc_semaphore_set(sem_addr, val);
}

void noc_semaphore_inc(Dataflow *api, Riscv32Core *core) {
    uint64_t addr = get_arg64(core, 0);
    uint32_t incr = core->get_arg(2);
    api->noc_semaphore_inc(addr, incr);
}

void noc_fast_read(Dataflow *api, Riscv32Core *core) {
    uint32_t src_addr = core->get_arg(0);
    uint32_t dest_addr = core->get_arg(1);
    api->noc_fast_read(src_addr, dest_addr);
}

// optimized NOC transfer APIs

void noc_fast_read_set_src_xy(Dataflow *api, Riscv32Core *core) {
    uint64_t src_addr = get_arg64(core, 0);
    api->noc_fast_read_set_src_xy(src_addr);
}

void noc_fast_read_set_len(Dataflow *api, Riscv32Core *core) {
    uint32_t len_bytes = core->get_arg(0);
    api->noc_fast_read_set_len(len_bytes);
}

void noc_fast_read_inc_num_issued(Dataflow *api, Riscv32Core *core) {
    uint32_t num_issued = core->get_arg(0);
    api->noc_fast_read_inc_num_issued(num_issued);
}

void noc_fast_write(Dataflow *api, Riscv32Core *core) {
    uint32_t src_addr = core->get_arg(0);
    uint64_t dest_addr = get_arg64(core, 1);
    api->noc_fast_write(src_addr, dest_addr);
}

void noc_fast_write_set_cmd_field(Dataflow *api, Riscv32Core *core) {
    uint32_t vc = core->get_arg(0);
    bool mcast = bool(core->get_arg(1));
    bool linked = bool(core->get_arg(2));
    api->noc_fast_write_set_cmd_field(vc, mcast, linked);
}

void noc_fast_write_set_dst_xy(Dataflow *api, Riscv32Core *core) {
    uint64_t dest_addr = get_arg64(core, 0);
    api->noc_fast_write_set_dst_xy(dest_addr);
}

void noc_fast_write_set_len(Dataflow *api, Riscv32Core *core) {
    uint32_t len_bytes = core->get_arg(0);
    api->noc_fast_write_set_len(len_bytes);
}

void noc_fast_write_inc_num_dests(Dataflow *api, Riscv32Core *core) {
    uint32_t num_issued = core->get_arg(0);
    api->noc_fast_write_inc_num_dests(num_issued);
}

// Command queue APIs

void cq_wait_front(Dataflow *api, Riscv32Core *core) {
    api->cq_wait_front();
}

void notify_host_of_cq_read_pointer(Dataflow *api, Riscv32Core *core) {
    api->notify_host_of_cq_read_pointer();
}

void cq_pop_front(Dataflow *api, Riscv32Core *core) {
    uint32_t cmd_size_B = core->get_arg(0);
    api->cq_pop_front(cmd_size_B);
}

} // namespace

//
//    DataflowHandler
//

DataflowHandler::DataflowHandler(Machine *machine):
        m_machine(machine) { }

DataflowHandler::~DataflowHandler() { }

#define DECL_BUILTIN(name, count, result) \
    case DataflowBuiltinId::name: \
        name(api, core); \
        break;

void DataflowHandler::call(Riscv32Core *core, int id) {
    Dataflow *api = m_machine->get_dataflow_api();
    switch (DataflowBuiltinId(id)) {
DATAFLOW_BUILTINS
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

