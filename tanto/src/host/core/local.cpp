// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <variant>
#include <type_traits>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/util.hpp"
#include "core/metal.hpp"

namespace ronin {
namespace tanto {
namespace host {

namespace {

CoreRangeSet make_device_core_range_set(const std::shared_ptr<DeviceImpl> &device) {
#if 0 // TODO: Revise this (temporarily preset for Wormhole)
    uint32_t x, y;
    device->worker_grid_size(x, y);
#else
    uint32_t x = 8;
    uint32_t y = 8;
#endif
    return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(x - 1, y - 1))});
}

CoreRangeSet make_grid_core_range_set(const std::shared_ptr<GridImpl> &grid) {
    return std::visit(
        [](auto &&arg) -> CoreRangeSet {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                return CoreRangeSet(std::set<CoreRange>{{arg}});
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                return CoreRangeSet({arg});
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                return arg;
            }
        }, 
        grid->impl());
}

} // namespace

//
//    LocalImpl
//

LocalImpl::LocalImpl(
        const std::shared_ptr<DeviceImpl> &device,
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        DataFormat data_format,
        uint32_t size,
        LocalScope scope):
            m_device(device),
            m_program(program),
            m_grid(grid),
            m_data_format(data_format),
            m_size(size),
            m_scope(scope) { }

LocalImpl::~LocalImpl() { }

std::shared_ptr<LocalImpl> LocalImpl::create(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        uint32_t size) {
    auto local =
        std::make_shared<LocalImpl>(
            device,
            nullptr,
            nullptr,
            data_format,
            size,
            LocalScope::DEVICE);
    device->add_local(local);
    local->create_impl();
    return local;
}

std::shared_ptr<LocalImpl> LocalImpl::create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        DataFormat data_format,
        uint32_t size) {
    validate_program_grid(program, grid);
    if (grid->range_count() != 1) {
        throw Error("Grid of local buffer has more than one range");
    }
    auto local =
        std::make_shared<LocalImpl>(
            program->device(),
            program,
            grid,
            data_format,
            size,
            LocalScope::PROGRAM);
    program->add_local(local);
    // create_impl() is deferred until program launch
    return local;
}

void LocalImpl::create_impl() {
    // size parameter is in items
    uint32_t size = ((m_size + 31) / 32) * 32;
    std::shared_ptr<DeviceImpl> device = m_device.lock();
    CoreRangeSet core_set = 
        (m_grid != nullptr) ?
            make_grid_core_range_set(m_grid) :
            make_device_core_range_set(device);
    uint32_t num_cores = core_set.num_cores();
    std::array<uint32_t, 2> shard_shape{32, size / 32};
    std::array<uint32_t, 2> page_shape{32, size / 32};
    std::array<uint32_t, 2> tensor2d_shape{32, (size / 32) * num_cores};
    uint32_t item_bytes = get_item_bytes(m_data_format);
    uint32_t bytes = size * item_bytes;
    metal::ShardedBufferConfig config{
        .device = device->impl(),
        .size = bytes * num_cores,
        .page_size = bytes,
        .buffer_type = metal::BufferType::L1,
        .buffer_layout = metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = metal::ShardSpecBuffer(
            core_set,
            shard_shape,
            metal::ShardOrientation::ROW_MAJOR,
#ifndef METAL_057
            false,
#endif
            page_shape,
            tensor2d_shape)
    };
    m_impl = metal::CreateBuffer(config);
}

void LocalImpl::release_impl() {
    m_impl.reset();
}

} // namespace host
} // namespace tanto
} // namespace ronin

