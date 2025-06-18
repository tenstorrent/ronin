// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "arch/noc_arch.hpp"

#include "core/kernel_structs.hpp"
#include "core/sync.hpp"
#include "core/memory.hpp"
#include "core/cb_api.hpp"
#include "core/noc_api.hpp"
#include "core/dataflow_api.hpp"

namespace tt {
namespace metal {
namespace device {

class DataflowImpl: public Dataflow {
public:
    DataflowImpl(
        Sync *sync,
        Memory *l1,
        CB *cb,
        NocArch *noc_arch,
        Noc *noc,
        uint32_t noc_index,
        uint32_t my_x,
        uint32_t my_y);
    ~DataflowImpl();
public:
    void reset() override;
    uint32_t get_arg_uint32(int arg_idx) override;
    void cb_push_back(uint32_t operand, uint32_t num_pages) override;
    void cb_pop_front(uint32_t operand, uint32_t num_pages) override;
    int32_t get_tile_size(uint32_t operand) override;
    DataFormat get_dataformat(uint32_t operand) override;
    uint32_t get_write_ptr(uint32_t operand) override;
    uint32_t get_read_ptr(uint32_t operand) override;
    void wait_for_sync_register_value(uint32_t addr, int32_t val) override;
    void cb_reserve_back(uint32_t operand, uint32_t num_pages) override;
    void cb_wait_front(uint32_t operand, uint32_t num_pages) override;
    // NOC transfers
    // simple APIs
    uint64_t get_noc_multicast_addr(
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t addr) override;
    uint64_t get_noc_addr_remote(uint32_t noc_x, uint32_t noc_y, uint32_t addr) override;
    uint64_t get_noc_addr_helper(uint32_t noc_xy, uint32_t addr) override;
    uint64_t get_dram_noc_addr(
        uint32_t id, 
        uint32_t page_size, 
        uint32_t bank_base_address, 
        uint32_t offset) override;
    uint64_t get_l1_noc_addr(
        uint32_t id, 
        uint32_t page_size, 
        uint32_t bank_base_address, 
        uint32_t offset) override;
    uint64_t get_system_memory_noc_addr(
        uint32_t id, 
        uint32_t page_size, 
        uint32_t base_addr, 
        uint32_t offset) override;
    uint64_t get_noc_addr_interleaved(
        uint32_t id, 
//        const InterleavedAddrGen<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        uint32_t offset) override;
    uint64_t get_noc_addr_interleaved_pow2(
        uint32_t id, 
//        const InterleavedPow2AddrGen<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t offset) override;
    uint64_t get_noc_addr_interleaved_fast(
        uint32_t id, 
//        const InterleavedAddrGenFast<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        DataFormat data_format,
        uint32_t offset) override;
    uint64_t get_noc_addr_local(uint32_t addr) override;
    void noc_async_read(
        uint64_t src_noc_addr, 
        uint32_t dst_local_l1_addr, 
        uint32_t size) override;
    void noc_async_read_one_packet(
        uint64_t src_noc_addr, 
        uint32_t dst_local_l1_addr, 
        uint32_t size) override;
    void noc_async_read_one_packet_set_state(uint64_t src_noc_addr, uint32_t size) override;
    void noc_async_read_one_packet_with_state(
        uint32_t src_noc_addr, 
        uint32_t dst_local_l1_addr,
        bool inc_num_issued) override;
    void noc_async_read_set_state(uint64_t src_noc_addr) override;
    void noc_async_read_with_state(
        uint32_t src_noc_addr, 
        uint32_t dst_local_l1_addr, 
        uint32_t size,
        bool inc_num_issued) override;
    void noc_async_read_inc_num_issued(uint32_t num_issued_reads_inc) override;
    void noc_async_write_one_packet(
        uint32_t src_local_l1_addr, 
        uint64_t dst_noc_addr, 
        uint32_t size) override;
    void noc_async_write_one_packet_set_state(
        uint64_t dst_noc_addr, 
        uint32_t size,
        bool non_posted) override;
    void noc_async_write_one_packet_with_state(
        uint32_t src_local_l1_addr, 
        uint32_t dst_noc_addr,
        bool non_posted) override;
    void noc_async_read_tile(
        uint32_t id, 
//        const InterleavedAddrGenFast<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        DataFormat data_format,
        uint32_t dst_local_l1_addr, 
        uint32_t offset) override;
    // converted from method
    void noc_async_read_page(
        uint32_t id, 
//        const InterleavedPow2AddrGenFast &s,
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t dest_addr, 
        uint32_t offset) override;
    // converted from method
    void noc_async_read_partial_page(
        uint32_t id, 
//        const InterleavedPow2AddrGenFast &s,
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t dest_addr, 
        uint32_t size, 
        uint32_t offset) override;
    void noc_async_write(
        uint32_t src_local_l1_addr, 
        uint64_t dst_noc_addr, 
        uint32_t size) override;
    void noc_async_write_tile(
        uint32_t id, 
//        const InterleavedAddrGenFast<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        DataFormat data_format,
        uint32_t src_local_l1_addr) override;
    // converted from method
    void noc_async_write_page(
        uint32_t id, 
//        const InterleavedPow2AddrGenFast &s,
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t src_addr, 
        uint32_t write_size_bytes, 
        uint32_t offset) override;
    uint32_t get_semaphore(uint32_t semaphore_id) override;
    void noc_semaphore_set_remote(uint32_t src_local_l1_addr, uint64_t dst_noc_addr) override;
    void noc_async_write_multicast(
        uint32_t src_local_l1_addr,
        uint64_t dst_noc_addr_multicast,
        uint32_t size,
        uint32_t num_dests) override;
    void noc_semaphore_set_multicast(
        uint32_t src_local_l1_addr, 
        uint64_t dst_noc_addr_multicast, 
        uint32_t num_dests) override;
    void noc_async_write_multicast_loopback_src(
        uint32_t src_local_l1_addr,
        uint64_t dst_noc_addr_multicast,
        uint32_t size,
        uint32_t num_dests) override;
    void noc_async_read_barrier() override;
    void noc_async_write_barrier() override;
    void noc_semaphore_wait(volatile uint32_t *sem_addr, uint32_t val) override;
    void noc_semaphore_set(volatile uint32_t *sem_addr, uint32_t val) override;
    void noc_semaphore_inc(uint64_t addr, uint32_t incr) override;
    void noc_fast_read(uint32_t src_addr, uint32_t dest_addr) override;
    // optimized NOC transfer APIs
    void noc_fast_read_set_src_xy(uint64_t src_addr) override;
    void noc_fast_read_set_len(uint32_t len_bytes) override;
    void noc_fast_read_inc_num_issued(uint32_t num_issued) override;
    void noc_fast_write(uint32_t src_addr, uint64_t dest_addr) override;
    void noc_fast_write_set_cmd_field(uint32_t vc, bool mcast, bool linked) override;
    void noc_fast_write_set_dst_xy(uint64_t dest_addr) override;
    void noc_fast_write_set_len(uint32_t len_bytes) override;
    void noc_fast_write_inc_num_dests(uint32_t num_issued) override;
    // Command queue APIs
    void cq_wait_front() override;
    void notify_host_of_cq_read_pointer() override;
    void cq_pop_front(uint32_t cmd_size_B) override;
    // Tanto extensions
    void noc_async_read_global_dram(
        uint32_t dst_addr,
        uint32_t src_addr,
        uint32_t src_log2_page_size,
        uint32_t src_offset,
        uint32_t len_bytes) override;
    void noc_async_read_global_l1(
        uint32_t dst_addr,
        uint32_t src_addr,
        uint32_t src_log2_page_size,
        uint32_t src_offset,
        uint32_t len_bytes) override;
    void noc_async_write_global_dram(
        uint32_t src_addr,
        uint32_t dst_addr,
        uint32_t dst_log2_page_size,
        uint32_t dst_offset,
        uint32_t len_bytes) override;
    void noc_async_write_global_l1(
        uint32_t src_addr,
        uint32_t dst_addr,
        uint32_t dst_log2_page_size,
        uint32_t dst_offset,
        uint32_t len_bytes) override;
private:
    uint64_t get_noc_addr_global_dram(
        uint32_t base_addr, 
        uint32_t log2_page_size, 
        uint32_t offset);
    uint64_t get_noc_addr_global_l1(
        uint32_t base_addr, 
        uint32_t log2_page_size, 
        uint32_t offset);
    void noc_fast_read_global(
        uint32_t dst_addr, 
        uint64_t src_addr, 
        uint32_t len_bytes);
    void noc_fast_write_global(
        uint32_t src_addr, 
        uint64_t dst_addr, 
        uint32_t len_bytes);
    uint32_t enc_noc_x(uint32_t x);
    uint32_t enc_noc_y(uint32_t y);
private:
    Sync *m_sync;
    Memory *m_l1;
    CB *m_cb;
    NocArch *m_noc_arch;
    Noc *m_noc;
    uint32_t m_noc_index;
    uint32_t m_my_x;
    uint32_t m_my_y;
    uint32_t m_num_dram_banks;
    uint32_t m_num_l1_banks;
    uint32_t m_noc_size_x;
    uint32_t m_noc_size_y;
    uint32_t m_pcie_core_noc_encoding;
};

} // namespace device
} // namespace metal
} // namespace tt

