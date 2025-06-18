// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>

#include "arch/noc_arch.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

//
//    NocArchWormholeB0
//

class NocArchWormholeB0: public NocArch {
public:
    NocArchWormholeB0();
    ~NocArchWormholeB0();
public:
    uint32_t num_dram_banks() override;
    uint32_t num_l1_banks() override;
    uint32_t noc_size_x() override;
    uint32_t noc_size_y() override;
    uint32_t pcie_noc_x() override;
    uint32_t pcie_noc_y() override;
    // noc_parameters
    uint64_t noc_xy_addr(uint32_t x, uint32_t y, uint32_t addr) override;
    uint64_t noc_multicast_addr(
        uint32_t x_start, 
        uint32_t y_start, 
        uint32_t x_end, 
        uint32_t y_end, 
        uint32_t addr) override;
    uint32_t noc_xy_encoding(uint32_t x, uint32_t y) override;
    uint32_t noc_multicast_encoding(
        uint32_t x_start, 
        uint32_t y_start, 
        uint32_t x_end, 
        uint32_t y_end) override;
    uint64_t noc_xy_addr2(uint32_t xy, uint32_t addr) override;
    // noc address parsing
    void parse_noc_addr(
        uint64_t noc_addr, 
        uint32_t &x, 
        uint32_t &y, 
        uint32_t &addr) override;
    void parse_noc_multicast_addr(
        uint64_t noc_addr,
        uint32_t &x_start, 
        uint32_t &y_start, 
        uint32_t &x_end, 
        uint32_t &y_end, 
        uint32_t &addr) override;
    // bank_to_noc_coord_mapping
    uint32_t dram_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) override;
    uint32_t bank_to_dram_offset(uint32_t bank_id) override;
    uint32_t l1_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) override;
    uint32_t bank_to_l1_offset(uint32_t bank_id) override;
private:
    static constexpr uint32_t NUM_NOCS = 2;
    static constexpr uint32_t NUM_DRAM_BANKS = 12;
    static constexpr uint32_t NUM_L1_BANKS = 64;
    static constexpr uint32_t NOC_SIZE_X = 10;
    static constexpr uint32_t NOC_SIZE_Y = 12;
    static constexpr uint32_t PCIE_NOC_X = 0;
    static constexpr uint32_t PCIE_NOC_Y = 3;
    static constexpr uint32_t NOC_ADDR_LOCAL_BITS = 36;
    static constexpr uint32_t NOC_ADDR_NODE_ID_BITS = 6; 
private:
    static uint32_t m_dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];
    static uint32_t m_bank_to_dram_offset[NUM_DRAM_BANKS];
    static uint32_t m_l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];
    static uint32_t m_bank_to_l1_offset[NUM_L1_BANKS];
};

NocArchWormholeB0::NocArchWormholeB0() { }

NocArchWormholeB0::~NocArchWormholeB0() { }

uint32_t NocArchWormholeB0::num_dram_banks() {
    return NUM_DRAM_BANKS;
}

uint32_t NocArchWormholeB0::num_l1_banks() {
    return NUM_L1_BANKS;
}

uint32_t NocArchWormholeB0::noc_size_x() {
    return NOC_SIZE_X;
}

uint32_t NocArchWormholeB0::noc_size_y() {
    return NOC_SIZE_Y;
}

uint32_t NocArchWormholeB0::pcie_noc_x() {
    return PCIE_NOC_X;
}

uint32_t NocArchWormholeB0::pcie_noc_y() {
    return PCIE_NOC_Y;
}

uint64_t NocArchWormholeB0::noc_xy_addr(uint32_t x, uint32_t y, uint32_t addr) {
    return (uint64_t(y) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
        (uint64_t(x) << NOC_ADDR_LOCAL_BITS) | 
        uint64_t(addr);
}

uint64_t NocArchWormholeB0::noc_multicast_addr(
        uint32_t x_start, 
        uint32_t y_start, 
        uint32_t x_end, 
        uint32_t y_end, 
        uint32_t addr) {
    return (uint64_t(x_start) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) |
        (uint64_t(y_start) << (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) |
        (uint64_t(x_end) << NOC_ADDR_LOCAL_BITS) |
        (uint64_t(y_end) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
        uint64_t(addr);
}

uint32_t NocArchWormholeB0::noc_xy_encoding(uint32_t x, uint32_t y) {
    return (uint32_t(y) << (NOC_ADDR_LOCAL_BITS % 32 + NOC_ADDR_NODE_ID_BITS)) |
        (uint32_t(x) << (NOC_ADDR_LOCAL_BITS % 32)) | 
        (uint32_t(x == PCIE_NOC_X && y == PCIE_NOC_Y) * 0x8);
}

uint32_t NocArchWormholeB0::noc_multicast_encoding(
        uint32_t x_start, 
        uint32_t y_start, 
        uint32_t x_end, 
        uint32_t y_end) {
    return (uint32_t(x_start) << (NOC_ADDR_LOCAL_BITS % 32 + 2 * NOC_ADDR_NODE_ID_BITS)) |
        (uint32_t(y_start) << (NOC_ADDR_LOCAL_BITS % 32 + 3 * NOC_ADDR_NODE_ID_BITS)) |
        (uint32_t(x_end) << (NOC_ADDR_LOCAL_BITS % 32)) |
        (uint32_t(y_end) << (NOC_ADDR_LOCAL_BITS % 32 + NOC_ADDR_NODE_ID_BITS));
}

uint64_t NocArchWormholeB0::noc_xy_addr2(uint32_t xy, uint32_t addr) {
    return (uint64_t(xy) << NOC_ADDR_LOCAL_BITS) | uint64_t(addr);
}

void NocArchWormholeB0::parse_noc_addr(
        uint64_t noc_addr, 
        uint32_t &x, 
        uint32_t &y, 
        uint32_t &addr) {
    // [y x addr]
    uint64_t mask = (uint64_t(1) << NOC_ADDR_LOCAL_BITS) - 1;
    addr = uint32_t(noc_addr & mask);
    noc_addr >>= NOC_ADDR_LOCAL_BITS;
    mask = (uint64_t(1) << NOC_ADDR_NODE_ID_BITS) - 1;
    x = uint32_t(noc_addr & mask);
    noc_addr >>= NOC_ADDR_NODE_ID_BITS;
    y = uint32_t(noc_addr);
}

void NocArchWormholeB0::parse_noc_multicast_addr(
        uint64_t noc_addr,
        uint32_t &x_start, 
        uint32_t &y_start, 
        uint32_t &x_end, 
        uint32_t &y_end, 
        uint32_t &addr) {
    // [y_start x_start y_end x_end addr]
    uint64_t mask = (uint64_t(1) << NOC_ADDR_LOCAL_BITS) - 1;
    addr = uint32_t(noc_addr & mask);
    noc_addr >>= NOC_ADDR_LOCAL_BITS;
    mask = (uint64_t(1) << NOC_ADDR_NODE_ID_BITS) - 1;
    x_end = uint32_t(noc_addr & mask);
    noc_addr >>= NOC_ADDR_NODE_ID_BITS;
    y_end = uint32_t(noc_addr & mask);
    noc_addr >>= NOC_ADDR_NODE_ID_BITS;
    x_start = uint32_t(noc_addr & mask);
    noc_addr >>= NOC_ADDR_NODE_ID_BITS;
    y_start = uint32_t(noc_addr);
}

uint32_t NocArchWormholeB0::dram_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) {
    assert(noc_index < NUM_NOCS);
    assert(bank_id < NUM_DRAM_BANKS);
    return m_dram_bank_to_noc_xy[noc_index][bank_id];
}

uint32_t NocArchWormholeB0::bank_to_dram_offset(uint32_t bank_id) {
    assert(bank_id < NUM_DRAM_BANKS);
    return m_bank_to_dram_offset[bank_id];
}

uint32_t NocArchWormholeB0::l1_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) {
    assert(noc_index < NUM_NOCS);
    assert(bank_id < NUM_L1_BANKS);
    return m_l1_bank_to_noc_xy[noc_index][bank_id];
}

uint32_t NocArchWormholeB0::bank_to_l1_offset(uint32_t bank_id) {
    assert(bank_id < NUM_L1_BANKS);
    return m_bank_to_l1_offset[bank_id];
}

uint32_t NocArchWormholeB0::m_dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] = {
    {
        11264,
        1024,
        5120,
        7168,
        1104,
        11344,
        2128,
        9296,
        8272,
        3152,
        5200,
        7248,
    },
    {
        144,
        10384,
        6288,
        4240,
        10304,
        64,
        9280,
        2112,
        3136,
        8256,
        6208,
        4160,
    },
};

uint32_t NocArchWormholeB0::m_bank_to_dram_offset[NUM_DRAM_BANKS] = {
    0,
    1073741824,
    0,
    1073741824,
    0,
    1073741824,
    0,
    1073741824,
    0,
    1073741824,
    0,
    1073741824,
};

uint32_t NocArchWormholeB0::m_l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] = {
    {
        5184,
        8304,
        4192,
        2144,
        7280,
        8240,
        4128,
        5152,
        4160,
        8288,
        9312,
        7264,
        3120,
        2064,
        7184,
        1152,
        1168,
        1040,
        8208,
        4144,
        4208,
        4224,
        8320,
        3216,
        7216,
        3200,
        2176,
        1056,
        5232,
        5264,
        5216,
        2080,
        2192,
        1088,
        9328,
        7312,
        9344,
        4112,
        7232,
        8336,
        5248,
        8224,
        1120,
        3168,
        5168,
        7296,
        3104,
        3136,
        9280,
        5136,
        2096,
        3088,
        8256,
        7200,
        3184,
        4240,
        9248,
        9360,
        1072,
        9232,
        2160,
        1136,
        9264,
        2112,
    },
    {
        6224,
        3104,
        7216,
        9264,
        4128,
        3168,
        7280,
        6256,
        7248,
        3120,
        2096,
        4144,
        8288,
        9344,
        4224,
        10256,
        10240,
        10368,
        3200,
        7264,
        7200,
        7184,
        3088,
        8192,
        4192,
        8208,
        9232,
        10352,
        6176,
        6144,
        6192,
        9328,
        9216,
        10320,
        2080,
        4096,
        2064,
        7296,
        4176,
        3072,
        6160,
        3184,
        10288,
        8240,
        6240,
        4112,
        8304,
        8272,
        2128,
        6272,
        9312,
        8320,
        3152,
        4208,
        8224,
        7168,
        2160,
        2048,
        10336,
        2176,
        9248,
        10272,
        2144,
        9296,
    },
};

uint32_t NocArchWormholeB0::m_bank_to_l1_offset[NUM_L1_BANKS] = { 0 };

NocArchWormholeB0 g_noc_arch_wormhole_b0;

} // namespace

//
//    Public functions
//

NocArch *get_noc_arch_wormhole_b0() {
    return &g_noc_arch_wormhole_b0;
}

} // namespace device
} // namespace metal
} // namespace tt

