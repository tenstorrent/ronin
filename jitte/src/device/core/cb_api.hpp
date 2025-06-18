// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "core/kernel_structs.hpp"
#include "core/sync.hpp"

namespace tt {
namespace metal {
namespace device {

class CB {
public:
    CB() { }
    virtual ~CB() { }
public:
    virtual void setup_read_write_interfaces(
        uint32_t cb_id,
        uint32_t fifo_addr,
        uint32_t fifo_size,
        uint32_t fifo_num_pages,
        uint32_t fifo_page_size) = 0;
    virtual void setup_data_formats(
        uint32_t cb_id, 
        DataFormat unpack_src_format,
        DataFormat unpack_dst_format,
        DataFormat pack_src_format,
        DataFormat pack_dst_format) = 0;
    virtual void cb_push_back(uint32_t cb_id, uint32_t num_pages) = 0;
    virtual void cb_pop_front(uint32_t cb_id, uint32_t num_pages) = 0;
    virtual uint32_t get_write_ptr(uint32_t cb_id) = 0;
    virtual uint32_t get_read_ptr(uint32_t cb_id) = 0;
    virtual void set_write_ptr(uint32_t cb_id, uint32_t ptr) = 0;
    virtual void set_read_ptr(uint32_t cb_id, uint32_t ptr) = 0;
    virtual void cb_reserve_back(uint32_t cb_id, uint32_t num_pages) = 0;
    virtual void cb_wait_front(uint32_t cb_id, uint32_t num_pages) = 0;
    virtual uint32_t get_write_tile_ptr(uint32_t cb_id) = 0;
    virtual void incr_write_tile_ptr(uint32_t cb_id, uint32_t num_tiles) = 0;
    virtual DataFormat get_unpack_src_format(uint32_t cb) = 0;
    virtual DataFormat get_unpack_dst_format(uint32_t cb) = 0;
    virtual DataFormat get_pack_src_format(uint32_t cb) = 0;
    virtual DataFormat get_pack_dst_format(uint32_t cb) = 0;
    virtual uint32_t get_tile_size(uint32_t cb) = 0;
public:
    static constexpr uint32_t NUM_CIRCULAR_BUFFERS = 32;
};

} // namespace device
} // namespace metal
} // namespace tt

