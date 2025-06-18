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

API void print_uint32(uint32 arg);

API void noc_async_read_global_dram(
    uint32 dst_addr,
    uint32 src_addr,
    uint32 src_log2_page_size,
    uint32 src_offset,
    uint32 len_bytes);
API void noc_async_read_global_l1(
    uint32 dst_addr,
    uint32 src_addr,
    uint32 src_log2_page_size,
    uint32 src_offset,
    uint32 len_bytes);
API void noc_async_write_global_dram(
    uint32 src_addr,
    uint32 dst_addr,
    uint32 dst_log2_page_size,
    uint32 dst_offset,
    uint32 len_bytes);
API void noc_async_write_global_l1(
    uint32 src_addr,
    uint32 dst_addr,
    uint32 dst_log2_page_size,
    uint32 dst_offset,
    uint32 len_bytes);

API void noc_async_read_linear_dram(
    uint32 dst_addr,
    uint32 src_addr,
    uint32 src_log2_page_size,
    uint32 src_page_id,
    uint32 src_offset,
    uint32 len_bytes);
API void noc_async_read_block_dram(
    uint32 dst_addr,
    uint32 src_addr,
    uint32 src_page_size,
    uint32 src_page_id,
    uint32 src_offset,
    uint32 len_bytes);
API void noc_async_read_cyclic_dram(
    uint32 dst_addr,
    uint32 src_addr,
    uint32 src_page_size,
    uint32 src_page_id,
    uint32 src_offset,
    uint32 len_bytes);
API void noc_async_write_linear_dram(
    uint32 src_addr,
    uint32 dst_addr,
    uint32 dst_log2_page_size,
    uint32 dst_page_id,
    uint32 dst_offset,
    uint32 len_bytes);
API void noc_async_write_block_dram(
    uint32 src_addr,
    uint32 dst_addr,
    uint32 dst_page_size,
    uint32 dst_page_id,
    uint32 dst_offset,
    uint32 len_bytes);
API void noc_async_write_cyclic_dram(
    uint32 src_addr,
    uint32 dst_addr,
    uint32 dst_page_size,
    uint32 dst_page_id,
    uint32 dst_offset,
    uint32 len_bytes);

