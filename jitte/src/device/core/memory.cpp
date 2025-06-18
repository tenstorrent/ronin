// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "core/memory.hpp"

namespace tt {
namespace metal {
namespace device {

//
//    L1Bank
//

L1Bank::L1Bank() { }

L1Bank::~L1Bank() { }

void L1Bank::init(uint32_t size) {
    m_data.resize(size, 0);
}

uint32_t L1Bank::size() {
    return uint32_t(m_data.size());
}

uint8_t *L1Bank::map_addr(uint32_t addr) {
    return &m_data[addr];
}

//
//    DramBank
//

DramBank::DramBank() { }

DramBank::~DramBank() { }

void DramBank::init(uint32_t size) {
    m_data.resize(size, 0);
}

uint32_t DramBank::size() {
    return uint32_t(m_data.size());
}

uint8_t *DramBank::map_addr(uint32_t addr) {
    return &m_data[addr];
}

} // namespace device
} // namespace metal
} // namespace tt

