// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/util.hpp"
#include "core/metal.hpp"

#ifndef METAL_057

#include "tt_metal/impl/device/device.hpp"

#endif

namespace ronin {
namespace tanto {
namespace host {

//
//    GlobalImpl
//

GlobalImpl::GlobalImpl(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        GlobalDist dist,
        bool is_dram,
        uint32_t size,
        uint32_t page_size):
            m_device(device),
            m_data_format(data_format),
            m_dist(dist),
            m_is_dram(is_dram),
            m_size(size),
            m_page_size(page_size) { }

GlobalImpl::~GlobalImpl() { }

std::shared_ptr<GlobalImpl> GlobalImpl::create(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        bool is_dram,
        uint32_t size,
        uint32_t log2_page_size) {
    uint32_t page_size = 1 << log2_page_size;
    auto global = 
        std::make_shared<GlobalImpl>(
            device,
            data_format,
            GlobalDist::LINEAR,
            is_dram,
            size,
            page_size);
    device->add_global(global);
    global->create_impl();
    return global;
}

std::shared_ptr<GlobalImpl> GlobalImpl::create(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        GlobalDist dist,
        uint32_t size,
        uint32_t page_size) {
    validate_dist_size(dist, size, page_size);
    auto global = 
        std::make_shared<GlobalImpl>(
            device,
            data_format,
            dist,
            false,
            size,
            page_size);
    device->add_global(global);
    global->create_impl();
    return global;
}

uint32_t GlobalImpl::log2_page_size() {
    return u32_log2(m_page_size);
}

void GlobalImpl::validate_dist_size(
        GlobalDist dist,
        uint32_t size,
        uint32_t page_size) {
    if (dist == GlobalDist::LINEAR) {
        if (!is_pow2(page_size)) {
            throw Error("Page size of linear global buffer must be power of 2");
        }
    } else if (dist == GlobalDist::BLOCK) {
        throw Error("Block distribution of global buffers is not supported");
     } else {
        // sharded buffer page size in items is fixed to 1024 in this version
        if (page_size % 1024 != 0) {
            throw Error("Page size of distributed global buffer must be multiple of 1024");
        }
        if (size % page_size != 0) {
            throw Error("Size of distributed global buffer must be multiple of its page size");
        }
    }
}

void GlobalImpl::create_impl() {
    if (m_dist == GlobalDist::LINEAR) {
        create_impl_linear();
    } else {
        create_impl_dist();
    }
}

void GlobalImpl::create_impl_linear() {
    // size parameters are in items
    uint32_t item_bytes = get_item_bytes(m_data_format);
    uint32_t bytes = m_size * item_bytes;
    uint32_t page_bytes = m_page_size * item_bytes;
    if (bytes < page_bytes) {
        bytes = page_bytes;
    } else if (bytes % page_bytes != 0) {
        bytes = ((bytes + page_bytes - 1) / page_bytes) * page_bytes;
    }
    std::shared_ptr<DeviceImpl> device = m_device.lock();
    metal::BufferType buffer_type = 
        m_is_dram ? metal::BufferType::DRAM : metal::BufferType::L1;
    metal::InterleavedBufferConfig config{
        .device = device->impl(),
        .size = bytes,
        .page_size = page_bytes,
        .buffer_type = buffer_type
    };
    m_impl = metal::CreateBuffer(config);
}

void GlobalImpl::create_impl_dist() {
    // sharded buffer page size in items is fixed to 1024 in this version
    // not to be mismatched with tanto global buffer page size
    uint32_t unit_size = 1024;
    std::shared_ptr<DeviceImpl> device = m_device.lock();
#ifdef METAL_057
    // ACHTUNG: Fixed for Wormhole
    // TODO: Implement general solution
    uint32_t num_banks = 12;
#else
    uint32_t num_banks = device->impl()->num_banks(metal::BufferType::DRAM);
#endif
    uint32_t num_pages = m_size / m_page_size;
    uint32_t num_rows = (num_pages + num_banks - 1) / num_banks;
    CoreCoord dram_grid_size = device->impl()->dram_grid_size();
    assert(dram_grid_size.x * dram_grid_size.y == num_banks);
    CoreRangeSet grid(
        std::set<CoreRange>({
            CoreRange(
                CoreCoord(0, 0),
                CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))
            }));
    metal::TensorMemoryLayout memory_layout = metal::TensorMemoryLayout::BLOCK_SHARDED;
    std::array<uint32_t, 2> page_shape{1, unit_size};
    std::array<uint32_t, 2> shard_shape{num_rows, m_page_size};
    std::array<uint32_t, 2> tensor_shape{num_rows, num_banks * m_page_size / unit_size};
    uint32_t item_bytes = get_item_bytes(m_data_format);
    uint32_t bytes = num_rows * num_banks * m_page_size * item_bytes;
    uint32_t unit_bytes = unit_size * item_bytes;
    metal::ShardedBufferConfig config{
        .device = device->impl(),
        .size = bytes,
        .page_size = unit_bytes,
        .buffer_type = metal::BufferType::DRAM,
        .buffer_layout = memory_layout,
        .shard_parameters = metal::ShardSpecBuffer(
            grid,
            shard_shape,
            metal::ShardOrientation::ROW_MAJOR,
#ifndef METAL_057
            false,
#endif
            page_shape,
            tensor_shape)
    };
    m_impl = metal::CreateBuffer(config);
}

uint32_t GlobalImpl::bytes() {
    return m_impl->size();
}

uint32_t GlobalImpl::page_bytes() {
    return m_page_size * get_item_bytes(m_data_format);
}

} // namespace host
} // namespace tanto
} // namespace ronin

