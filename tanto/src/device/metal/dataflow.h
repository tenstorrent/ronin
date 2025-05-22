// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "dataflow_api.h"

typedef uint32_t uint32;

struct Global {
    uint32 addr;
    uint32 log2_page_size;
};

struct Local {
    uint32 addr;
};

struct Pipe {
    uint32 cb_id;
    uint32 frame_size;
};

struct Semaphore {
    uint32 addr;
};

#define tanto_get_semaphore(x) get_semaphore(x)

FORCE_INLINE uint64_t __get_noc_addr_global_dram(
        uint32_t base_addr, 
        uint32_t log2_page_size, 
        uint32_t offset) {
    uint32_t id = offset >> log2_page_size;
    offset -= id << log2_page_size;
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
    uint32_t bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(id);
    uint32_t addr = udivsi3_const_divisor<NUM_DRAM_BANKS>(id) << log2_page_size;
#else
    uint32_t bank_id = id & (NUM_DRAM_BANKS - 1);
    uint32_t addr = (id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) << log2_page_size;
#endif
    addr += base_addr + offset;
    addr += bank_to_dram_offset[bank_id];
    uint32_t noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
    return get_noc_addr_helper(noc_xy, addr);
}

FORCE_INLINE uint64_t __get_noc_addr_global_l1(
        uint32_t base_addr, 
        uint32_t log2_page_size, 
        uint32_t offset) {
    uint32_t id = offset >> log2_page_size;
    offset -= id << log2_page_size;
#ifdef IS_NOT_POW2_NUM_L1_BANKS
    uint32_t bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
    uint32_t addr = udivsi3_const_divisor<NUM_L1_BANKS>(id) << log2_page_size;
#else
    uint32_t bank_id = id & (NUM_L1_BANKS - 1);
    uint32_t addr = (id >> LOG_BASE_2_OF_NUM_L1_BANKS) << log2_page_size;
#endif
    addr += base_addr + offset;
    addr += bank_to_l1_offset[bank_id];
    uint32_t noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
    return get_noc_addr_helper(noc_xy, addr);
}

FORCE_INLINE void __noc_fast_read_global(
        uint32_t dst_addr, 
        uint64_t src_addr, 
        uint32_t len_bytes) {
    uint32_t addr = uint32_t(src_addr);
    uint32_t noc_xy = uint32_t(src_addr >> 32);

    while (!noc_cmd_buf_ready(noc_index, NCRISC_RD_CMD_BUF));

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, noc_xy);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    noc_reads_num_issued[noc_index] += 1;
}

FORCE_INLINE void __noc_fast_write_global(
        uint32_t src_addr, 
        uint64_t dst_addr, 
        uint32_t len_bytes) {
    uint32_t addr = uint32_t(dst_addr);
    uint32_t noc_xy = uint32_t(dst_addr >> 32);

    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF));

    uint32_t noc_cmd_field = 
        NOC_CMD_CPY | 
        NOC_CMD_WR | 
        NOC_CMD_VC_STATIC |
        NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | 
        0x0 | // (linked ? NOC_CMD_VC_LINKED : 0x0)
        0x0 | // (mcast ? (NOC_CMD_PATH_RESERVE | NOC_CMD_BRCST_PACKET) : 0x0)
        NOC_CMD_RESP_MARKED;

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID, noc_xy);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE,  len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    noc_nonposted_writes_num_issued[noc_index] += 1;
    noc_nonposted_writes_acked[noc_index] += 1;
}

FORCE_INLINE void noc_async_read_global_dram(
        uint32 dst_addr,
        uint32 src_addr,
        uint32 src_log2_page_size,
        uint32 src_offset,
        uint32 len_bytes) {
    // TODO: Optimize for simple cases
    uint32_t page_size = 1 << src_log2_page_size;
    uint32_t bytes_left = len_bytes;
    uint32_t limit = page_size - (src_offset & (page_size - 1));
    while (bytes_left != 0) {
        uint32_t xfer_bytes = (bytes_left < limit) ? bytes_left : limit;
        uint64_t noc_addr = __get_noc_addr_global_dram(src_addr, src_log2_page_size, src_offset);
        __noc_fast_read_global(dst_addr, noc_addr, xfer_bytes);
        dst_addr += xfer_bytes;
        src_offset += xfer_bytes;
        bytes_left -= xfer_bytes;
        limit = page_size;
    }
}

FORCE_INLINE void noc_async_read_global_l1(
        uint32 dst_addr,
        uint32 src_addr,
        uint32 src_log2_page_size,
        uint32 src_offset,
        uint32 len_bytes) {
    // TODO: Optimize for simple cases
    uint32_t page_size = 1 << src_log2_page_size;
    uint32_t bytes_left = len_bytes;
    uint32_t limit = page_size - (src_offset & (page_size - 1));
    while (bytes_left != 0) {
        uint32_t xfer_bytes = (bytes_left < limit) ? bytes_left : limit;
        uint64_t noc_addr = __get_noc_addr_global_l1(src_addr, src_log2_page_size, src_offset);
        __noc_fast_read_global(dst_addr, noc_addr, xfer_bytes);
        dst_addr += xfer_bytes;
        src_offset += xfer_bytes;
        bytes_left -= xfer_bytes;
        limit = page_size;
    }
}

FORCE_INLINE void noc_async_write_global_dram(
        uint32 src_addr,
        uint32 dst_addr,
        uint32 dst_log2_page_size,
        uint32 dst_offset,
        uint32 len_bytes) {
    // TODO: Optimize for simple cases
    uint32_t page_size = 1 << dst_log2_page_size;
    uint32_t bytes_left = len_bytes;
    uint32_t limit = page_size - (dst_offset & (page_size - 1));
    while (bytes_left != 0) {
        uint32_t xfer_bytes = (bytes_left < limit) ? bytes_left : limit;
        uint64_t noc_addr = __get_noc_addr_global_dram(dst_addr, dst_log2_page_size, dst_offset);
        __noc_fast_write_global(src_addr, noc_addr, xfer_bytes);
        src_addr += xfer_bytes;
        dst_offset += xfer_bytes;
        bytes_left -= xfer_bytes;
        limit = page_size;
    }
}

FORCE_INLINE void noc_async_write_global_l1(
        uint32 src_addr,
        uint32 dst_addr,
        uint32 dst_log2_page_size,
        uint32 dst_offset,
        uint32 len_bytes) {
    // TODO: Optimize for simple cases
    uint32_t page_size = 1 << dst_log2_page_size;
    uint32_t bytes_left = len_bytes;
    uint32_t limit = page_size - (dst_offset & (page_size - 1));
    while (bytes_left != 0) {
        uint32_t xfer_bytes = (bytes_left < limit) ? bytes_left : limit;
        uint64_t noc_addr = __get_noc_addr_global_l1(dst_addr, dst_log2_page_size, dst_offset);
        __noc_fast_write_global(src_addr, noc_addr, xfer_bytes);
        src_addr += xfer_bytes;
        dst_offset += xfer_bytes;
        bytes_left -= xfer_bytes;
        limit = page_size;
    }
}

FORCE_INLINE uint64_t __get_noc_addr_cyclic_dram(
        uint32_t base_addr,
        uint32_t page_id,
        uint32_t page_size,
        uint32_t offset) {
#ifdef IS_NOT_POW2_NUM_DRAM_BANKS
    uint32_t bank_id = umodsi3_const_divisor<NUM_DRAM_BANKS>(page_id);
    uint32_t addr = udivsi3_const_divisor<NUM_DRAM_BANKS>(page_id) * page_size;
#else
    uint32_t bank_id = page_id & (NUM_DRAM_BANKS - 1);
    uint32_t addr = (page_id >> LOG_BASE_2_OF_NUM_DRAM_BANKS) * page_size;
#endif
    addr += base_addr + offset;
    addr += bank_to_dram_offset[bank_id];
    uint32_t noc_xy = dram_bank_to_noc_xy[noc_index][bank_id];
    return get_noc_addr_helper(noc_xy, addr);
}

FORCE_INLINE void noc_async_read_cyclic_dram(
        uint32_t dst_addr,
        uint32_t src_addr,
        uint32_t src_page_size,
        uint32_t src_page_id,
        uint32_t src_offset,
        uint32_t len_bytes) {
    uint64_t src_noc_addr =
        __get_noc_addr_cyclic_dram(src_addr, src_page_id, src_page_size, src_offset);
    __noc_fast_read_global(dst_addr, src_noc_addr, len_bytes);
}

FORCE_INLINE void noc_async_write_cyclic_dram(
        uint32_t src_addr,
        uint32_t dst_addr,
        uint32_t dst_page_size,
        uint32_t dst_page_id,
        uint32_t dst_offset,
        uint32_t len_bytes) {
    uint64_t dst_noc_addr =
        __get_noc_addr_cyclic_dram(dst_addr, dst_page_id, dst_page_size, dst_offset);
    __noc_fast_write_global(src_addr, dst_noc_addr, len_bytes);
}


