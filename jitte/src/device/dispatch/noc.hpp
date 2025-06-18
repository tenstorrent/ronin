// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"

namespace tt {
namespace metal {
namespace device {
namespace dispatch {

class Noc {
public:
    Noc(Soc *soc, NocArch *noc_arch);
    ~Noc();
public:
    uint64_t get_noc_addr_interleaved(
        bool is_dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        uint32_t id,
        uint32_t offset);
    uint64_t get_noc_addr_helper(uint32_t xy, uint32_t addr);
    void read(
        uint64_t src_noc_addr, 
        uint8_t *dst, 
        uint32_t size);
    void write(
        const uint8_t *src, 
        uint64_t dst_noc_addr, 
        uint32_t size);
    void write_multicast(
        const uint8_t *src,
        uint64_t dst_noc_addr_multicast,
        uint32_t size,
        uint32_t num_dests);
private:
    uint8_t *map_remote_addr(uint64_t noc_addr, uint32_t size);
    uint8_t *map_remote_addr(
        uint32_t x, 
        uint32_t y, 
        uint32_t addr, 
        uint32_t size);
private:
    Soc *m_soc;
    NocArch *m_noc_arch;
    uint32_t m_num_dram_banks;
    uint32_t m_num_l1_banks;
};

} // namespace dispatch
} // namespace device
} // namespace metal
} // namespace tt

