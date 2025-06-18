// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "core/kernel_structs.hpp"
#include "core/sync.hpp"
#include "core/cb_impl.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

uint32_t get_l1_tile_size(DataFormat format) {
    // returns tile size in 16B words
    switch (format) {
    case DataFormat::Float16_b: 
    case DataFormat::Float16:
        return 2048 >> 4;
    case DataFormat::Bfp8:
    case DataFormat::Bfp8_b:
        return (1024 >> 4) + (64 >> 4);
    case DataFormat::Float32:
        return 4096 >> 4;
    case DataFormat::Bfp4:
    case DataFormat::Bfp4_b:
        return (512 >> 4) + (64 >> 4);
    case DataFormat::Bfp2:
    case DataFormat::Bfp2_b:
        return (256 >> 4) + (64 >> 4);
    default: 
        return (1024 >> 4) + (64 >> 4);
    }
}

} // namespace

//
//    CBImpl
//

CBImpl::CBImpl(Sync *sync):
        m_sync(sync) { 
    reset_read_write_interfaces();
    reset_data_formats();
}

CBImpl::~CBImpl() { }

void CBImpl::setup_read_write_interfaces(
        uint32_t cb_id,
        uint32_t fifo_addr,
        uint32_t fifo_size,
        uint32_t fifo_num_pages,
        uint32_t fifo_page_size) {
    // NOTE: fifo_addr, fifo_size and fifo_limit in 16B words
    m_cb_interface[cb_id].fifo_size = fifo_size;
    m_cb_interface[cb_id].fifo_limit = fifo_addr + fifo_size;
    m_cb_interface[cb_id].fifo_page_size = fifo_page_size;
    m_cb_interface[cb_id].fifo_num_pages = fifo_num_pages;
    m_cb_interface[cb_id].fifo_wr_ptr = fifo_addr;
    m_cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
    m_cb_interface[cb_id].tiles_acked = 0;
    m_cb_interface[cb_id].tiles_received = 0;
    m_cb_interface[cb_id].fifo_wr_tile_ptr = 0;
}

void CBImpl::setup_data_formats(
        uint32_t cb_id, 
        DataFormat unpack_src_format,
        DataFormat unpack_dst_format,
        DataFormat pack_src_format,
        DataFormat pack_dst_format) {
    m_unpack_src_format[cb_id] = unpack_src_format;
    m_unpack_dst_format[cb_id] = unpack_dst_format;
    m_pack_src_format[cb_id] = pack_src_format;
    m_pack_dst_format[cb_id] = pack_dst_format;
    // use unpack_src_format, but because unpack_src_format == pack_dst_format, we can use either
    m_tile_size[cb_id] = get_l1_tile_size(unpack_src_format);
}

void CBImpl::cb_push_back(uint32_t cb_id, uint32_t num_pages) {
    CBInterface &cb = m_cb_interface[cb_id];

    uint32_t num_words =  num_pages * cb.fifo_page_size;

    uint32_t *tiles_received_ptr = get_cb_tiles_received_ptr(cb_id);
    tiles_received_ptr[0] += num_pages;

    cb.fifo_wr_ptr += num_words;
    // borrowed from "llk_lib/llk_op_pack.h"
    // required for implementation of 'llk_pack' primitive
    cb.fifo_wr_tile_ptr = 0; 

    // this will basically reset fifo_wr_ptr to fifo_addr -- no other wrap is legal
    // producer always writes into contiguous memory, it cannot wrap
    if (cb.fifo_wr_ptr >= cb.fifo_limit) {
        cb.fifo_wr_ptr -= cb.fifo_size;
    }
}

void CBImpl::cb_pop_front(uint32_t cb_id, uint32_t num_pages) {
    CBInterface &cb = m_cb_interface[cb_id];

    uint32_t *tiles_acked_ptr = get_cb_tiles_acked_ptr(cb_id);
    tiles_acked_ptr[0] += num_pages;

    uint32_t num_words = num_pages * cb.fifo_page_size;

    cb.fifo_rd_ptr += num_words;

    // this will basically reset fifo_rd_ptr to fifo_addr -- no other wrap is legal
    // consumer always reads from contiguous memory, it cannot wrap
    if (cb.fifo_rd_ptr >= cb.fifo_limit) {
        cb.fifo_rd_ptr -= cb.fifo_size;
    }
}

uint32_t CBImpl::get_write_ptr(uint32_t cb_id) {
    // return byte address (fifo_wr_ptr is 16B address)
    uint32_t wr_ptr_bytes = m_cb_interface[cb_id].fifo_wr_ptr << 4;
    return wr_ptr_bytes;
}

uint32_t CBImpl::get_read_ptr(uint32_t cb_id) {
    // return byte address (fifo_wr_ptr is 16B address)
    uint32_t rd_ptr_bytes = m_cb_interface[cb_id].fifo_rd_ptr << 4;
    return rd_ptr_bytes;
}

void CBImpl::set_write_ptr(uint32_t cb_id, uint32_t ptr) {
    // convert byte address (fifo_wr_ptr is 16B address)
    m_cb_interface[cb_id].fifo_wr_ptr = ptr >> 4;
}

void CBImpl::set_read_ptr(uint32_t cb_id, uint32_t ptr) {
    // convert byte address (fifo_rd_ptr is 16B address)
    m_cb_interface[cb_id].fifo_rd_ptr = ptr >> 4;
}

void CBImpl::cb_reserve_back(uint32_t cb_id, uint32_t num_pages) {
    volatile uint32_t *pages_acked_ptr = get_cb_tiles_acked_ptr(cb_id);
    volatile uint32_t *pages_received_ptr = get_cb_tiles_received_ptr(cb_id);

    // while the producer (write-side interface) is waiting for
    // space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = pages_received_ptr[0];

    auto cond = [=, this]() -> bool {
        uint32_t pages_acked = pages_acked_ptr[0];
        uint32_t free_space_pages = 
            m_cb_interface[cb_id].fifo_num_pages - (pages_received - pages_acked);
        return (free_space_pages >= num_pages);
    };

    m_sync->wait(cond);
}

void CBImpl::cb_wait_front(uint32_t cb_id, uint32_t num_pages) {
    volatile uint32_t *pages_acked_ptr = get_cb_tiles_acked_ptr(cb_id);
    volatile uint32_t *pages_received_ptr = get_cb_tiles_received_ptr(cb_id);

    // "tiles_popped" doesn't change while we wait for tiles to be pushed to CB
    uint32_t pages_acked = pages_acked_ptr[0];

    auto cond = [=]() -> bool {
        uint32_t pages_received = pages_received_ptr[0];
        uint32_t num_pages_recv = pages_received - pages_acked;
        return (num_pages_recv >= num_pages);
    };

    m_sync->wait(cond);
}

uint32_t CBImpl::get_write_tile_ptr(uint32_t cb_id) {
    return m_cb_interface[cb_id].fifo_wr_tile_ptr;
}

void CBImpl::incr_write_tile_ptr(uint32_t cb_id, uint32_t num_tiles) {
    m_cb_interface[cb_id].fifo_wr_tile_ptr += num_tiles;
}

DataFormat CBImpl::get_unpack_src_format(uint32_t cb) {
    return m_unpack_src_format[cb];
}

DataFormat CBImpl::get_unpack_dst_format(uint32_t cb) {
    return m_unpack_dst_format[cb];
}

DataFormat CBImpl::get_pack_src_format(uint32_t cb) {
    return m_pack_src_format[cb];
}

DataFormat CBImpl::get_pack_dst_format(uint32_t cb) {
    return m_pack_dst_format[cb];
}

uint32_t CBImpl::get_tile_size(uint32_t cb) {
    // L1 16B words
    uint32_t num_words = m_tile_size[cb];
    // return bytes
    return num_words << 4;
}

void CBImpl::reset_read_write_interfaces() {
    for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
        m_cb_interface[cb_id].fifo_size = 0;
        m_cb_interface[cb_id].fifo_limit = 0;
        m_cb_interface[cb_id].fifo_page_size = 0;
        m_cb_interface[cb_id].fifo_num_pages = 0;
        m_cb_interface[cb_id].fifo_wr_ptr = 0;
        m_cb_interface[cb_id].fifo_rd_ptr = 0;
        m_cb_interface[cb_id].tiles_acked = 0;
        m_cb_interface[cb_id].tiles_received = 0;
        m_cb_interface[cb_id].fifo_wr_tile_ptr = 0;
    }
}

void CBImpl::reset_data_formats() {
    for (uint32_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
        // use Float32 instead of Invalid to simplify initial testing
        m_unpack_src_format[cb_id] = DataFormat::Float32;
        m_unpack_dst_format[cb_id] = DataFormat::Float32;
        m_pack_src_format[cb_id] = DataFormat::Float32;
        m_pack_dst_format[cb_id] = DataFormat::Float32;
        m_tile_size[cb_id] = get_l1_tile_size(DataFormat::Float32);
    }
}

} // namespace device
} // namespace metal
} // namespace tt

