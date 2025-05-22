// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"

namespace ronin {
namespace tanto {
namespace host {

//
//    ProgramImpl
//

ProgramImpl::ProgramImpl(const std::shared_ptr<DeviceImpl> &device):
        m_device(device) { }

ProgramImpl::~ProgramImpl() { }

std::shared_ptr<ProgramImpl> ProgramImpl::create(
        const std::shared_ptr<DeviceImpl> &device) {
    auto program = std::make_shared<ProgramImpl>(device);
    device->add_program(program);
    program->create_impl();
    return program;
}

void ProgramImpl::add_grid(const std::shared_ptr<GridImpl> &grid) {
    m_grids.emplace_back(grid);
}

void ProgramImpl::add_local(const std::shared_ptr<LocalImpl> &local) {
    m_locals.emplace_back(local);
}

void ProgramImpl::add_pipe(const std::shared_ptr<PipeImpl> &pipe) {
    m_pipes.emplace_back(pipe);
}

void ProgramImpl::add_semaphore(const std::shared_ptr<SemaphoreImpl> &semaphore) {
    m_semaphores.emplace_back(semaphore);
}

void ProgramImpl::add_kernel(const std::shared_ptr<KernelImpl> &kernel) {
    m_kernels.emplace_back(kernel);
}

void ProgramImpl::before_enqueue() {
    for (auto &local: m_locals) {
        if (local->scope() == LocalScope::PROGRAM) {
            local->create_impl();
            metal::AssignGlobalBufferToProgram(local->impl(), m_impl);
        }
    }
    for (auto &pipe: m_pipes) {
        pipe->update_dynamic_address();
    }
    for (auto &kernel: m_kernels) {
        kernel->set_args_impl();
    }
}

void ProgramImpl::after_enqueue() {
    for (auto &local: m_locals) {
        if (local->scope() == LocalScope::PROGRAM) {
            local->release_impl();
        }
    }
}

void ProgramImpl::create_impl() {
    m_impl = metal::CreateProgram();
}

} // namespace host
} // namespace tanto
} // namespace ronin

