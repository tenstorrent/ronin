// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/metal.hpp"

namespace ronin {
namespace tanto {
namespace host {

//
//    DeviceImpl
//

DeviceImpl::DeviceImpl(const std::shared_ptr<PlatformImpl> &platform, uint32_t id):
        m_platform(platform), 
        m_id(id),
        m_impl(nullptr) { }

DeviceImpl::~DeviceImpl() { 
    if (m_impl != nullptr) {
        metal::CloseDevice(m_impl);
    }
}

std::shared_ptr<DeviceImpl> DeviceImpl::create(
        const std::shared_ptr<PlatformImpl> &platform, uint32_t id) {
    std::shared_ptr<DeviceImpl> device = platform->find_device(id);
    if (device != nullptr) {
        if (device->m_impl == nullptr) {
            device->m_impl = metal::CreateDevice(id);
        }
        return device;
    }
    device = std::make_shared<DeviceImpl>(platform, id);
    platform->add_device(device);
    device->m_impl = metal::CreateDevice(id);
    return device;
}

void DeviceImpl::add_global(const std::shared_ptr<GlobalImpl> &global) {
    m_globals.emplace_back(global);
}

void DeviceImpl::add_local(const std::shared_ptr<LocalImpl> &local) {
    m_locals.emplace_back(local);
}

void DeviceImpl::add_program(const std::shared_ptr<ProgramImpl> &program) {
    m_programs.emplace_back(program);
}

void DeviceImpl::add_queue(const std::shared_ptr<QueueImpl> &queue) {
    m_queues.emplace_back(queue);
}

const std::shared_ptr<QueueImpl> DeviceImpl::find_queue(uint32_t id) {
    for (auto &queue: m_queues) {
        if (queue->id() == id) {
            return queue;
        }
    }
    return nullptr;
}

void DeviceImpl::dram_grid_size(uint32_t &x, uint32_t &y) {
    CoreCoord size = m_impl->dram_grid_size();
    x = uint32_t(size.x);
    y = uint32_t(size.y);
}

void DeviceImpl::worker_grid_size(uint32_t &x, uint32_t &y) {
    CoreCoord size = m_impl->compute_with_storage_grid_size();
    x = uint32_t(size.x);
    y = uint32_t(size.y);
}

void DeviceImpl::worker_core_from_logical_core(
        uint32_t logical_x,
        uint32_t logical_y,
        uint32_t &worker_x,
        uint32_t &worker_y) {
    CoreCoord worker_coord = 
        m_impl->worker_core_from_logical_core(CoreCoord(logical_x, logical_y));
    worker_x = uint32_t(worker_coord.x);
    worker_y = uint32_t(worker_coord.y);
}

void DeviceImpl::close() {
    validate_impl();
    metal::CloseDevice(m_impl);
    m_globals.clear();
    m_programs.clear();
    m_queues.clear();
    m_impl = nullptr;
}

void DeviceImpl::validate_impl() {
    if (m_impl == nullptr) {
        throw Error("Null TT-Metal device reference");
    }
}

} // namespace host
} // namespace tanto
} // namespace ronin

