#pragma once

#include <cstdint>

#include "core/kernel_structs.hpp"
#include "core/sync.hpp"
#include "core/cb_api.hpp"

namespace tt {
namespace metal {
namespace device {

struct CBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit;
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;
    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;
    uint32_t tiles_acked;
    uint32_t tiles_received;
    // used by packer for in-order packing
    uint32_t fifo_wr_tile_ptr;
};

class CBImpl: public CB {
public:
    CBImpl(Sync *sync);
    ~CBImpl();
public:
    void setup_read_write_interfaces(
        uint32_t cb_id,
        uint32_t fifo_addr,
        uint32_t fifo_size,
        uint32_t fifo_num_pages,
        uint32_t fifo_page_size) override;
    void setup_data_formats(
        uint32_t cb_id, 
        DataFormat unpack_src_format,
        DataFormat unpack_dst_format,
        DataFormat pack_src_format,
        DataFormat pack_dst_format) override;
    void cb_push_back(uint32_t cb_id, uint32_t num_pages) override;
    void cb_pop_front(uint32_t cb_id, uint32_t num_pages) override;
    uint32_t get_write_ptr(uint32_t cb_id) override;
    uint32_t get_read_ptr(uint32_t cb_id) override;
    void cb_reserve_back(uint32_t cb_id, uint32_t num_pages) override;
    void cb_wait_front(uint32_t cb_id, uint32_t num_pages) override;
    uint32_t get_write_tile_ptr(uint32_t cb_id) override;
    void incr_write_tile_ptr(uint32_t cb_id, uint32_t num_tiles) override;
    DataFormat get_unpack_src_format(uint32_t cb) override;
    DataFormat get_unpack_dst_format(uint32_t cb) override;
    DataFormat get_pack_src_format(uint32_t cb) override;
    DataFormat get_pack_dst_format(uint32_t cb) override;
    uint32_t get_tile_size(uint32_t cb) override;
private:
    void reset_read_write_interfaces();
    void reset_data_formats();
    uint32_t *get_cb_tiles_acked_ptr(uint32_t cb_id) {
        return &m_cb_interface[cb_id].tiles_acked;
    }
    uint32_t *get_cb_tiles_received_ptr(uint32_t cb_id) {
        return &m_cb_interface[cb_id].tiles_received;
    }
private:
    Sync *m_sync;
    CBInterface m_cb_interface[NUM_CIRCULAR_BUFFERS];
    DataFormat m_unpack_src_format[NUM_CIRCULAR_BUFFERS];
    DataFormat m_unpack_dst_format[NUM_CIRCULAR_BUFFERS];
    DataFormat m_pack_src_format[NUM_CIRCULAR_BUFFERS];
    DataFormat m_pack_dst_format[NUM_CIRCULAR_BUFFERS];
    uint32_t m_tile_size[NUM_CIRCULAR_BUFFERS];
};

} // namespace device
} // namespace metal
} // namespace tt

