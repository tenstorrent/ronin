
#include <cstdint>
#include <cstring>
#include <string>
#include <cassert>
#include <stdexcept>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"

#include "dispatch/noc.hpp"

namespace tt {
namespace metal {
namespace device {
namespace dispatch {

namespace {

uint32_t align(uint32_t addr, uint32_t alignment) { 
    return ((addr - 1) | (alignment - 1)) + 1; 
}

void data_copy(uint8_t *dest, const uint8_t *src, uint32_t len) {
    // Use memmove to support overlapped regions (is overlapping allowed?)
    memmove(dest, src, len);
}

std::string xy_to_string(uint32_t x, uint32_t y) {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

} // namespace

//
//    Noc
//

Noc::Noc(Soc *soc, NocArch *noc_arch):
        m_soc(soc),
        m_noc_arch(noc_arch),
        m_num_dram_banks(m_noc_arch->num_dram_banks()),
        m_num_l1_banks(m_noc_arch->num_l1_banks()) { }

Noc::~Noc() { }

uint64_t Noc::get_noc_addr_interleaved(
        bool is_dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        uint32_t id,
        uint32_t offset) {
    if (is_dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        uint32_t addr = 
            (id / m_num_dram_banks) * align(page_size, 32) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_dram_offset(bank_id);
        // assume noc_index = 0
        uint32_t noc_xy = m_noc_arch->dram_bank_to_noc_xy(0, bank_id);
        return get_noc_addr_helper(noc_xy, addr);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        uint32_t addr = 
            (id / m_num_l1_banks) * align(page_size, 32) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_l1_offset(bank_id);
        // assume noc_index = 0
        uint32_t noc_xy = m_noc_arch->l1_bank_to_noc_xy(0, bank_id);
        return get_noc_addr_helper(noc_xy, addr);
    }
}

uint64_t Noc::get_noc_addr_helper(uint32_t xy, uint32_t addr) {
    // shift by 32 even if (NOC_ADDR_LOCAL_BITS > 32)
    // because 'xy' contais XY value already shifted by (NOC_ADDR_LOCAL_BITS - 32)
    // (by construction of 'bank_to_noc_xy' tables)
    return (uint64_t(xy) << 32) | addr;
}

void Noc::read(
        uint64_t src_noc_addr, 
        uint8_t *dst, 
        uint32_t size) {
    uint8_t *src = map_remote_addr(src_noc_addr, size);
    data_copy(dst, src, size);
}

void Noc::write(
        const uint8_t *src, 
        uint64_t dst_noc_addr, 
        uint32_t size) {
    uint8_t *dst = map_remote_addr(dst_noc_addr, size);
    data_copy(dst, src, size);
}

void Noc::write_multicast(
        const uint8_t *src,
        uint64_t dst_noc_addr_multicast,
        uint32_t size,
        uint32_t num_dests) {
    uint32_t x_start = 0;
    uint32_t y_start = 0;
    uint32_t x_end = 0;
    uint32_t y_end = 0;
    uint32_t addr = 0;
    m_noc_arch->parse_noc_multicast_addr(
        dst_noc_addr_multicast,
        x_start, 
        y_start, 
        x_end, 
        y_end, 
        addr);
    if (x_start > x_end) {
        uint32_t temp = x_start;
        x_start = x_end;
        x_end = temp;
    }
    if (y_start > y_end) {
        uint32_t temp = y_start;
        y_start = y_end;
        y_end = temp;
    }
    for (uint32_t x = x_start; x <= x_end; x++) {
        for (uint32_t y = y_start; y <= y_end; y++) {
            if (m_soc->core_type(int(x), int(y)) != CoreType::WORKER) {
                continue;
            }
            uint8_t *dst = map_remote_addr(x, y, addr, size);
            data_copy(dst, src, size);
        }
    }
}

uint8_t *Noc::map_remote_addr(uint64_t noc_addr, uint32_t size) {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t addr = 0;
    m_noc_arch->parse_noc_addr(noc_addr, x, y, addr);
    return map_remote_addr(x, y, addr, size);
}

uint8_t *Noc::map_remote_addr(
        uint32_t x, 
        uint32_t y, 
        uint32_t addr, 
        uint32_t size) {
    CoreType core_type = m_soc->core_type(int(x), int(y));
    if (core_type == CoreType::DRAM) {
#if 1
        // ACHTUNG: Temporary workaround: addr may be too high on Wormhole B0
        //     (TODO: Investigate how this could happen)
        addr %= m_soc->dram_bank_size();
#endif
        assert(addr + size <= m_soc->dram_bank_size());
        int dram_channel = m_soc->core_dram_channel(int(x), int(y));
        return m_soc->map_dram_addr(dram_channel, addr);
    }
    if (core_type == CoreType::WORKER) {
        assert(addr + size <= m_soc->worker_l1_size());
        return m_soc->map_l1_addr(int(x), int(y), addr);
    }
    throw std::runtime_error("No DRAM or worker core at " + xy_to_string(x, y));
    return nullptr;
}

} // namespace dispatch
} // namespace device
} // namespace metal
} // namespace tt

