#pragma once

#include <string>
#include <unordered_map>

//
//    Generic list of dataflow builtins
//

#define DATAFLOW_BUILTINS \
    DECL_BUILTIN(get_arg_uint32, 1, 1) \
    DECL_BUILTIN(cb_push_back, 2, 0) \
    DECL_BUILTIN(cb_pop_front, 2, 0) \
    DECL_BUILTIN(get_tile_size, 1, 1) \
    DECL_BUILTIN(get_dataformat, 1, 1) \
    DECL_BUILTIN(get_write_ptr, 1, 1) \
    DECL_BUILTIN(get_read_ptr, 1, 1) \
    DECL_BUILTIN(wait_for_sync_register_value, 2, 0) \
    DECL_BUILTIN(cb_reserve_back, 2, 0) \
    DECL_BUILTIN(cb_wait_front, 2, 0) \
    DECL_BUILTIN(get_noc_multicast_addr, 5, 2) \
    DECL_BUILTIN(get_noc_addr_remote, 3, 2) \
    DECL_BUILTIN(get_noc_addr_helper, 2, 2) \
    DECL_BUILTIN(get_dram_noc_addr, 4, 2) \
    DECL_BUILTIN(get_l1_noc_addr, 4, 2) \
    DECL_BUILTIN(get_system_memory_noc_addr, 4, 2) \
    DECL_BUILTIN(get_noc_addr_interleaved, 5, 2) \
    DECL_BUILTIN(get_noc_addr_interleaved_pow2, 5, 2) \
    DECL_BUILTIN(get_noc_addr_interleaved_fast, 6, 2) \
    DECL_BUILTIN(get_noc_addr_local, 1, 2) \
    DECL_BUILTIN(noc_async_read, 4, 0) \
    DECL_BUILTIN(noc_async_read_one_packet, 4, 0) \
    DECL_BUILTIN(noc_async_read_one_packet_set_state, 3, 0) \
    DECL_BUILTIN(noc_async_read_one_packet_with_state, 3, 0) \
    DECL_BUILTIN(noc_async_read_set_state, 2, 0) \
    DECL_BUILTIN(noc_async_read_with_state, 4, 0) \
    DECL_BUILTIN(noc_async_read_inc_num_issued, 1, 0) \
    DECL_BUILTIN(noc_async_write_one_packet, 4, 0) \
    DECL_BUILTIN(noc_async_write_one_packet_set_state, 4, 0) \
    DECL_BUILTIN(noc_async_write_one_packet_with_state, 3, 0) \
    DECL_BUILTIN(noc_async_read_tile, 7, 0) \
    DECL_BUILTIN(noc_async_read_page, 6, 0) \
    DECL_BUILTIN(noc_async_read_partial_page, 7, 0) \
    DECL_BUILTIN(noc_async_write, 4, 0) \
    DECL_BUILTIN(noc_async_write_tile, 6, 0) \
    DECL_BUILTIN(noc_async_write_page, 7, 0) \
    DECL_BUILTIN(get_semaphore, 1, 1) \
    DECL_BUILTIN(noc_semaphore_set_remote, 3, 0) \
    DECL_BUILTIN(noc_async_write_multicast, 5, 0) \
    DECL_BUILTIN(noc_semaphore_set_multicast, 4, 0) \
    DECL_BUILTIN(noc_async_write_multicast_loopback_src, 5, 0) \
    DECL_BUILTIN(noc_async_read_barrier, 0, 0) \
    DECL_BUILTIN(noc_async_write_barrier, 0, 0) \
    DECL_BUILTIN(noc_semaphore_wait, 2, 0) \
    DECL_BUILTIN(noc_semaphore_set, 2, 0) \
    DECL_BUILTIN(noc_semaphore_inc, 3, 0) \
    DECL_BUILTIN(noc_fast_read, 2, 0) \
    DECL_BUILTIN(noc_fast_read_set_src_xy, 2, 0) \
    DECL_BUILTIN(noc_fast_read_set_len, 1, 0) \
    DECL_BUILTIN(noc_fast_read_inc_num_issued, 1, 0) \
    DECL_BUILTIN(noc_fast_write, 3, 0) \
    DECL_BUILTIN(noc_fast_write_set_cmd_field, 3, 0) \
    DECL_BUILTIN(noc_fast_write_set_dst_xy, 2, 0) \
    DECL_BUILTIN(noc_fast_write_set_len, 1, 0) \
    DECL_BUILTIN(noc_fast_write_inc_num_dests, 1, 0) \
    DECL_BUILTIN(cq_wait_front, 0, 0) \
    DECL_BUILTIN(notify_host_of_cq_read_pointer, 0, 0) \
    DECL_BUILTIN(cq_pop_front, 1, 0)

//
//    Dataflow builtin enumeration
//

#define DECL_BUILTIN(name, count, result) name,

enum class DataflowBuiltinId {
    START = 1024,
DATAFLOW_BUILTINS
};

#undef DECL_BUILTIN

// public functions

struct DataflowBuiltinEntry {
    std::string name;
    int count;
    int result;
};

std::unordered_map<DataflowBuiltinId, DataflowBuiltinEntry> &get_dataflow_builtin_map();

