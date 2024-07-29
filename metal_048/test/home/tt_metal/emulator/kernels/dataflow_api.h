#pragma once

#include <cstdint>

#include "kernel_structs.h"
#include "hostdevcommon/common_values.hpp"

#define FORCE_INLINE inline
#define ALWI inline

#define tt_l1_ptr

#define API extern "C"

// Borrowed from "hw/inc/<ARCH_NAME>/dev_mem_map.h"
// Generally arch-specific but same for GS and WH B0

#define MEM_ZEROS_BASE 2048
#define MEM_ZEROS_SIZE 512

// DataFormat borrowed from "hw/inc/wormhole/wormhole_b0_defines/tensix_types.h"

enum class DataFormat {
    Float32   = 0,
    Float16   = 1,
    Bfp8      = 2,
    Bfp4      = 3,
    Bfp2      = 11,
    Float16_b = 5,
    Bfp8_b    = 6,
    Bfp4_b    = 7,
    Bfp2_b    = 15,
    Lf8       = 10,
    UInt16    = 12,
    Int8      = 14,
    UInt8     = 30,
    Int32     = 8,
    Int16     = 9,
    Tf32      = 4,
    testMan7  = 0x82,       // intermediate format for testing: 7bit mantissa (6+hidden)
    testMan2  = 0x8A,       // intermediate format for testing: 2bit mantissa (2+hidden)
    Invalid   = 0xff
};

#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_ ## arg_idx

API uint32_t get_arg_uint32(int arg_idx);

template <typename T>
T get_arg_val(int arg_idx) {
    // only uint32_t is supported
    return static_cast<T>(get_arg_uint32(arg_idx));
}

API void cb_push_back(int32_t operand, int32_t num_pages);
API void cb_pop_front(int32_t operand, int32_t num_pages);
API int32_t get_tile_size(int32_t operand);
API DataFormat get_dataformat(int32_t operand);
API uint32_t get_write_ptr(uint32_t operand);
API uint32_t get_read_ptr(uint32_t operand);
API void wait_for_sync_register_value(uint32_t addr, int32_t val);
API void cb_reserve_back(int32_t operand, int32_t num_pages);
API void cb_wait_front(int32_t operand, int32_t num_pages);

// NOC transfers
// simple APIs
API uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start,
    uint32_t noc_y_start,
    uint32_t noc_x_end,
    uint32_t noc_y_end,
    uint32_t addr);

API uint64_t get_noc_addr_remote(uint32_t noc_x, uint32_t noc_y, uint32_t addr);

inline uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr) {
    return get_noc_addr_remote(noc_x, noc_y, addr);
}

API uint64_t get_noc_addr_helper(uint32_t noc_xy, uint32_t addr);
API uint64_t get_dram_noc_addr(
    uint32_t id, 
    uint32_t page_size, 
    uint32_t bank_base_address, 
    uint32_t offset = 0);
API uint64_t get_l1_noc_addr(
    uint32_t id, 
    uint32_t page_size, 
    uint32_t bank_base_address, 
    uint32_t offset = 0);
API uint64_t get_system_memory_noc_addr(
    uint32_t id, 
    uint32_t page_size, 
    uint32_t base_addr, 
    uint32_t offset = 0);

API uint64_t get_noc_addr_interleaved(
    uint32_t id,
    bool dram,
    uint32_t bank_base_address,
    uint32_t page_size,
    uint32_t offset);

template <bool DRAM>
struct InterleavedAddrGen {
    uint32_t bank_base_address;
    uint32_t page_size;

    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) const {
        return get_noc_addr_interleaved(id, DRAM, bank_base_address, page_size, offset);
    }
};

API uint64_t get_noc_addr_interleaved_pow2(
    uint32_t id, 
    bool dram,
    uint32_t bank_base_address,
    uint32_t log_base_2_of_page_size,
    uint32_t offset);

template <bool DRAM>
struct InterleavedPow2AddrGen {
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size;

    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) const {
        return get_noc_addr_interleaved_pow2(
            id, DRAM, bank_base_address, log_base_2_of_page_size, offset);
    }
};

API uint64_t get_noc_addr_interleaved_fast(
    uint32_t id, 
    bool dram,
    uint32_t bank_base_address,
    uint32_t page_size,
    DataFormat data_format,
    uint32_t offset);
API void noc_async_read_tile(
    uint32_t id, 
    bool dram,
    uint32_t bank_base_address,
    uint32_t page_size,
    DataFormat data_format,
    uint32_t dst_local_l1_addr, 
    uint32_t offset);
API void noc_async_write_tile(
    uint32_t id, 
    bool dram,
    uint32_t bank_base_address,
    uint32_t page_size,
    DataFormat data_format,
    uint32_t src_local_l1_addr);

template <bool DRAM>
struct InterleavedAddrGenFast {
    uint32_t bank_base_address;
    uint32_t page_size;
    DataFormat data_format;

    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) const {
        return get_noc_addr_interleaved_fast(
            id, DRAM, bank_base_address, page_size, data_format, offset);
    }
    void noc_async_read_tile(uint32_t id, uint32_t dest_addr, uint32_t offset = 0) const {
        ::noc_async_read_tile(
            id, 
            DRAM,
            bank_base_address,
            page_size,
            data_format, 
            dest_addr, 
            offset);
    }
    void noc_async_write_tile(uint32_t id, uint32_t src_addr) const {
        ::noc_async_write_tile(
            id, 
            DRAM,
            bank_base_address,
            page_size,
            data_format,
            src_addr);
    }
};

API void noc_async_read_page(
    uint32_t id, 
    bool dram,
    uint32_t bank_base_address,
    uint32_t log_base_2_of_page_size,
    uint32_t dest_addr, 
    uint32_t offset);
API void noc_async_read_partial_page(
    uint32_t id, 
    bool dram,
    uint32_t bank_base_address,
    uint32_t log_base_2_of_page_size,
    uint32_t dest_addr, 
    uint32_t size, 
    uint32_t offset);
API void noc_async_write_page(
    uint32_t id, 
    bool dram,
    uint32_t bank_base_address,
    uint32_t log_base_2_of_page_size,
    uint32_t src_addr, 
    uint32_t write_size_bytes, 
    uint32_t offset);

template <bool DRAM>
struct InterleavedPow2AddrGenFast {
    uint32_t bank_base_address;
    uint32_t log_base_2_of_page_size;

    void noc_async_read_page(uint32_t id, uint32_t dest_addr, uint32_t offset = 0) const {
        ::noc_async_read_page(
            id, 
            DRAM,
            bank_base_address,
            log_base_2_of_page_size,
            dest_addr, 
            offset);
    }
    void noc_async_read_partial_page(
            uint32_t id, 
            uint32_t dest_addr, 
            uint32_t size, 
            uint32_t offset) const {
        ::noc_async_read_partial_page(
            id, 
            DRAM,
            bank_base_address,
            log_base_2_of_page_size,
            dest_addr, 
            size, 
            offset);
    }
    void noc_async_write_page(
            uint32_t id, 
            uint32_t src_addr, 
            uint32_t write_size_bytes, 
            uint32_t offset = 0) const {
        ::noc_async_write_page(
            id, 
            DRAM,
            bank_base_address,
            log_base_2_of_page_size,
            src_addr, 
            write_size_bytes, 
            offset);
    }
};

template <bool DRAM>
uint64_t get_noc_addr(uint32_t id, const InterleavedAddrGen<DRAM> &s, uint32_t offset = 0) {
    return s.get_noc_addr(id, offset);
}

template <bool DRAM>
uint64_t get_noc_addr(uint32_t id, const InterleavedPow2AddrGen<DRAM> &s, uint32_t offset = 0) {
    return s.get_noc_addr(id, offset);
}

template <bool DRAM>
uint64_t get_noc_addr(uint32_t id, const InterleavedAddrGenFast<DRAM> &s, uint32_t offset = 0) {
    return s.get_noc_addr(id, offset);
}

API uint64_t get_noc_addr_local(uint32_t addr);

inline uint64_t get_noc_addr(uint32_t addr) {
    return get_noc_addr_local(addr);
}

API void noc_async_read(
    uint64_t src_noc_addr, 
    uint32_t dst_local_l1_addr,
    uint32_t size);
API void noc_async_read_one_packet(
    uint64_t src_noc_addr, 
    uint32_t dst_local_l1_addr, 
    uint32_t size);
API void noc_async_read_one_packet_set_state(uint64_t src_noc_addr, uint32_t size);

API void noc_async_read_one_packet_with_state(
    uint32_t src_noc_addr, 
    uint32_t dst_local_l1_addr,
    bool inc_num_issued);

template <bool inc_num_issued = true>
void noc_async_read_one_packet_with_state(uint32_t src_noc_addr, uint32_t dst_local_l1_addr) {
    noc_async_read_one_packet_with_state(src_noc_addr, dst_local_l1_addr, inc_num_issued);
}

API void noc_async_read_set_state(uint64_t src_noc_addr);

API void noc_async_read_with_state(
    uint32_t src_noc_addr, 
    uint32_t dst_local_l1_addr, 
    uint32_t size,
    bool inc_num_issued);

template <bool inc_num_issued = true>
void noc_async_read_with_state(uint32_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size) {
    noc_async_read_with_state(src_noc_addr, dst_local_l1_addr, size, inc_num_issued);
}

API void noc_async_read_inc_num_issued(uint32_t num_issued_reads_inc);
API void noc_async_write_one_packet(
    uint32_t src_local_l1_addr, 
    uint64_t dst_noc_addr, 
    uint32_t size);

API void noc_async_write_one_packet_set_state(
    uint64_t dst_noc_addr, 
    uint32_t size,
    bool non_posted);

template <bool non_posted = true>
void noc_async_write_one_packet_set_state(uint64_t dst_noc_addr, uint32_t size) {
    noc_async_write_one_packet_set_state(dst_noc_addr, size, non_posted);
 }

API void noc_async_write_one_packet_with_state(
    uint32_t src_local_l1_addr, 
    uint32_t dst_noc_addr,
    bool non_posted);

template <bool non_posted = true>
void noc_async_write_one_packet_with_state(uint32_t src_local_l1_addr, uint32_t dst_noc_addr) {
    noc_async_write_one_packet_with_state(src_local_l1_addr, dst_noc_addr, non_posted);
}

template <bool DRAM>
void noc_async_read_tile(
        uint32_t id, 
        const InterleavedAddrGenFast<DRAM> &s, 
        uint32_t dst_local_l1_addr, 
        uint32_t offset = 0) {
    s.noc_async_read_tile(id, dst_local_l1_addr, offset);
}

API void noc_async_write(
    uint32_t src_local_l1_addr, 
    uint64_t dst_noc_addr, 
    uint32_t size);

template <bool DRAM>
void noc_async_write_tile(
        uint32_t id, 
        const InterleavedAddrGenFast<DRAM> &s, 
        uint32_t src_local_l1_addr) {
    s.noc_async_write_tile(id, src_local_l1_addr);
}

API uint32_t get_semaphore(uint32_t semaphore_id);
API void noc_semaphore_set_remote(uint32_t src_local_l1_addr, uint64_t dst_noc_addr);
API void noc_async_write_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests);
API void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr, 
    uint64_t dst_noc_addr_multicast, 
    uint32_t num_dests);
API void noc_async_write_multicast_loopback_src(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests);
API void noc_async_read_barrier();
API void noc_async_write_barrier();
API void noc_semaphore_wait(volatile uint32_t *sem_addr, uint32_t val);
API void noc_semaphore_set(volatile uint32_t *sem_addr, uint32_t val);
API void noc_semaphore_inc(uint64_t addr, uint32_t incr);

// optimized NOC transfer APIs
API void noc_fast_read(uint32_t src_addr, uint32_t dest_addr);
API void noc_fast_read_set_src_xy(uint64_t src_addr);
API void noc_fast_read_set_len(uint32_t len_bytes);
API void noc_fast_read_inc_num_issued(uint32_t num_issued);
API void noc_fast_write(uint32_t src_addr, uint64_t dest_addr);
API void noc_fast_write_set_cmd_field(uint32_t vc, bool mcast, bool linked);
API void noc_fast_write_set_dst_xy(uint64_t dest_addr);
API void noc_fast_write_set_len(uint32_t len_bytes);
API void noc_fast_write_inc_num_dests(uint32_t num_issued);

// Command queue APIs
API void cq_wait_front();
API void notify_host_of_cq_read_pointer();
API void cq_pop_front(uint32_t cmd_size_B);

enum class BufferType {
    DRAM = 0,
    L1 = 1,
    SYSTEM_MEMORY = 2
};

class Buffer {
private:
    uint32_t bank_base_address;
    uint32_t page_size_;
    uint64_t (*get_noc_addr_helper)(uint32_t, uint32_t, uint32_t, uint32_t);
    BufferType type;

    void set_type(BufferType type) {
        this->type = type;
        switch (type) {
        case BufferType::DRAM:
            this->get_noc_addr_helper = get_dram_noc_addr; 
            break;
        case BufferType::L1:
            this->get_noc_addr_helper = get_l1_noc_addr; 
            break;
        case BufferType::SYSTEM_MEMORY:
            this->get_noc_addr_helper = get_system_memory_noc_addr; 
            break;
        }
    }
    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) {
        return this->get_noc_addr_helper(id, this->page_size_, this->bank_base_address, offset);
    }
public:
    Buffer(BufferType type, uint32_t bank_base_address, uint32_t page_size) {
        this->set_type(type);
        this->bank_base_address = bank_base_address;
        this->page_size_ = page_size;
    }
    uint32_t page_size() { 
        return this->page_size_; 
    }
    void noc_async_write_buffer(uint32_t src, uint32_t id, uint32_t num_pages, uint32_t offset) {
        if (this->type == BufferType::SYSTEM_MEMORY) {
            noc_async_write(src, this->get_noc_addr(id, offset), this->page_size_ * num_pages);
        } else {
            for (uint32_t i = 0; i < num_pages; i++) {
                uint64_t address = this->get_noc_addr(id + i, offset);
                noc_async_write(src, address, this->page_size_);
                src += this->page_size_;
            }
        }
    }
    void noc_async_read_buffer(uint32_t dst, uint32_t id, uint32_t num_pages, uint32_t offset) {
        if (this->type == BufferType::SYSTEM_MEMORY) {
            noc_async_read(this->get_noc_addr(id, offset), dst, this->page_size_ * num_pages);
        } else {
            for (uint32_t i = 0; i < num_pages; i++) {
                uint64_t address = this->get_noc_addr(id + i, offset);
                noc_async_read(address, dst, this->page_size_);
                dst += this->page_size_;
            }
        }
    }
};

// conventional main function for linker

void kernel_main();

extern "C" int main() {
    kernel_main();
    return 0;
}

