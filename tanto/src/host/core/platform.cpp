// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <mutex>

#include "core/api.hpp"
#include "core/impl.hpp"

namespace ronin {
namespace tanto {
namespace host {

//
//    PlatformImpl
//

std::once_flag PlatformImpl::m_default_flag; 
std::shared_ptr<PlatformImpl> PlatformImpl::m_default;

PlatformImpl::PlatformImpl() { }

PlatformImpl::~PlatformImpl() { }

std::shared_ptr<PlatformImpl> PlatformImpl::get_default() {
    std::call_once(m_default_flag, make_default);
    return m_default;
}

void PlatformImpl::add_device(const std::shared_ptr<DeviceImpl> &device) {
    m_devices.emplace_back(device);
}

std::shared_ptr<DeviceImpl> PlatformImpl::find_device(uint32_t id) {
    for (auto &device: m_devices) {
        if (device->id() == id) {
            return device;
        }
    }
    return nullptr;
}

void PlatformImpl::make_default() {
    m_default = std::make_shared<PlatformImpl>();
}

} // namespace host
} // namespace tanto
} // namespace ronin

