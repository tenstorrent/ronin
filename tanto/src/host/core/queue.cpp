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
//    QueueImpl
//

QueueImpl::QueueImpl(const std::shared_ptr<DeviceImpl> &device, uint32_t id):
        m_device(device),
        m_id(id),
        m_impl(nullptr) { }

QueueImpl::~QueueImpl() { }

std::shared_ptr<QueueImpl> QueueImpl::create(
        const std::shared_ptr<DeviceImpl> &device, uint32_t id) {
    std::shared_ptr<QueueImpl> queue = device->find_queue(id);
    if (queue != nullptr) {
        return queue;
    }
    queue = std::make_shared<QueueImpl>(device, id);
    device->add_queue(queue);
    queue->create_impl();
    return queue;
}

void QueueImpl::enqueue_read(
        const std::shared_ptr<GlobalImpl> &global, 
        void *dst,
        bool blocking) {
    metal::EnqueueReadBuffer(*m_impl, global->impl(), dst, blocking);
}

void QueueImpl::enqueue_write(
        const std::shared_ptr<GlobalImpl> &global, 
        const void *src,
        bool blocking) {
    metal::EnqueueWriteBuffer(*m_impl, global->impl(), src, blocking);
}

void QueueImpl::enqueue_program(
        const std::shared_ptr<ProgramImpl> &program, bool blocking) {
    program->before_enqueue();
    metal::EnqueueProgram(*m_impl, program->impl(), blocking);
    program->after_enqueue();
}

void QueueImpl::finish() {
    metal::Finish(*m_impl);
}

void QueueImpl::create_impl() {
    std::shared_ptr<DeviceImpl> device = m_device.lock();
    m_impl = &device->impl()->command_queue(m_id);
}

} // namespace host
} // namespace tanto
} // namespace ronin

