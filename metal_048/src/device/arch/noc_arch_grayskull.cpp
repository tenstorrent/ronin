
#include <cstdint>
#include <cassert>

#include "arch/noc_arch.hpp"

//
//    NOTE: E150 architecture is assumed for L1 bank mapping
//

namespace tt {
namespace metal {
namespace device {

namespace {

//
//    NocArchGrayskull
//

class NocArchGrayskull: public NocArch {
public:
    NocArchGrayskull();
    ~NocArchGrayskull();
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
    static constexpr uint32_t NUM_DRAM_BANKS = 8;
    static constexpr uint32_t NUM_L1_BANKS = 128;
    static constexpr uint32_t NOC_SIZE_X = 13;
    static constexpr uint32_t NOC_SIZE_Y = 12;
    static constexpr uint32_t PCIE_NOC_X = 0;
    static constexpr uint32_t PCIE_NOC_Y = 4;
    static constexpr uint32_t NOC_ADDR_LOCAL_BITS = 32;
    static constexpr uint32_t NOC_ADDR_NODE_ID_BITS = 6; 
private:
    static uint32_t m_dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];
    static uint32_t m_bank_to_dram_offset[NUM_DRAM_BANKS];
    static uint32_t m_l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];
    static uint32_t m_bank_to_l1_offset[NUM_L1_BANKS];
};

NocArchGrayskull::NocArchGrayskull() { }

NocArchGrayskull::~NocArchGrayskull() { }

uint32_t NocArchGrayskull::num_dram_banks() {
    return NUM_DRAM_BANKS;
}

uint32_t NocArchGrayskull::num_l1_banks() {
    return NUM_L1_BANKS;
}

uint32_t NocArchGrayskull::noc_size_x() {
    return NOC_SIZE_X;
}

uint32_t NocArchGrayskull::noc_size_y() {
    return NOC_SIZE_Y;
}

uint32_t NocArchGrayskull::pcie_noc_x() {
    return PCIE_NOC_X;
}

uint32_t NocArchGrayskull::pcie_noc_y() {
    return PCIE_NOC_Y;
}

uint64_t NocArchGrayskull::noc_xy_addr(uint32_t x, uint32_t y, uint32_t addr) {
    return (uint64_t(y) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
        (uint64_t(x) << NOC_ADDR_LOCAL_BITS) | 
        uint64_t(addr);
}

uint64_t NocArchGrayskull::noc_multicast_addr(
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

uint32_t NocArchGrayskull::noc_xy_encoding(uint32_t x, uint32_t y) {
    return (uint32_t(y) << (NOC_ADDR_LOCAL_BITS % 32 + NOC_ADDR_NODE_ID_BITS)) |
        (uint32_t(x) << (NOC_ADDR_LOCAL_BITS % 32)) | 
        (uint32_t(x == PCIE_NOC_X && y == PCIE_NOC_Y) * 0x8);
}

uint32_t NocArchGrayskull::noc_multicast_encoding(
        uint32_t x_start, 
        uint32_t y_start, 
        uint32_t x_end, 
        uint32_t y_end) {
    return (uint32_t(x_start) << (NOC_ADDR_LOCAL_BITS % 32 + 2 * NOC_ADDR_NODE_ID_BITS)) |
        (uint32_t(y_start) << (NOC_ADDR_LOCAL_BITS % 32 + 3 * NOC_ADDR_NODE_ID_BITS)) |
        (uint32_t(x_end) << (NOC_ADDR_LOCAL_BITS % 32)) |
        (uint32_t(y_end) << (NOC_ADDR_LOCAL_BITS % 32 + NOC_ADDR_NODE_ID_BITS));
}

uint64_t NocArchGrayskull::noc_xy_addr2(uint32_t xy, uint32_t addr) {
    return (uint64_t(xy) << NOC_ADDR_LOCAL_BITS) | uint64_t(addr);
}

void NocArchGrayskull::parse_noc_addr(
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

void NocArchGrayskull::parse_noc_multicast_addr(
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

uint32_t NocArchGrayskull::dram_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) {
    assert(noc_index < NUM_NOCS);
    assert(bank_id < NUM_DRAM_BANKS);
    return m_dram_bank_to_noc_xy[noc_index][bank_id];
}

uint32_t NocArchGrayskull::bank_to_dram_offset(uint32_t bank_id) {
    assert(bank_id < NUM_DRAM_BANKS);
    return m_bank_to_dram_offset[bank_id];
}

uint32_t NocArchGrayskull::l1_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) {
    assert(noc_index < NUM_NOCS);
    assert(bank_id < NUM_L1_BANKS);
    return m_l1_bank_to_noc_xy[noc_index][bank_id];
}

uint32_t NocArchGrayskull::bank_to_l1_offset(uint32_t bank_id) {
    assert(bank_id < NUM_L1_BANKS);
    return m_bank_to_l1_offset[bank_id];
}

uint32_t NocArchGrayskull::m_dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] = {
    {
        1,
        385,
        4,
        388,
        7,
        391,
        10,
        394,
    },
    {
        715,
        331,
        712,
        328,
        709,
        325,
        706,
        322,
    },
};

uint32_t NocArchGrayskull::m_bank_to_dram_offset[NUM_DRAM_BANKS] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
};

uint32_t NocArchGrayskull::m_l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] = {
    {
        204,
        581,
        197,
        648,
        578,
        323,
        194,
        706,
        196,
        325,
        513,
        265,
        135,
        73,
        261,
        514,
        72,
        516,
        321,
        195,
        583,
        199,
        459,
        140,
        263,
        139,
        708,
        523,
        585,
        712,
        586,
        644,
        710,
        68,
        643,
        268,
        707,
        193,
        645,
        515,
        458,
        715,
        522,
        137,
        203,
        267,
        134,
        647,
        646,
        715,
        75,
        133,
        324,
        582,
        138,
        200,
        330,
        709,
        67,
        709,
        130,
        70,
        577,
        76,
        453,
        264,
        652,
        707,
        136,
        259,
        327,
        521,
        649,
        71,
        714,
        65,
        716,
        74,
        714,
        262,
        326,
        69,
        588,
        457,
        331,
        708,
        460,
        66,
        455,
        713,
        706,
        651,
        258,
        257,
        519,
        713,
        650,
        587,
        450,
        518,
        454,
        332,
        517,
        129,
        449,
        579,
        580,
        641,
        198,
        642,
        451,
        712,
        266,
        131,
        710,
        329,
        132,
        452,
        260,
        456,
        584,
        520,
        328,
        202,
        201,
        716,
        524,
        322,
    },
    {
        512,
        135,
        519,
        68,
        138,
        393,
        522,
        10,
        520,
        391,
        203,
        451,
        581,
        643,
        455,
        202,
        644,
        200,
        395,
        521,
        133,
        517,
        257,
        576,
        453,
        577,
        8,
        193,
        131,
        4,
        130,
        72,
        6,
        648,
        73,
        448,
        9,
        523,
        71,
        201,
        258,
        1,
        194,
        579,
        513,
        449,
        582,
        69,
        70,
        1,
        641,
        583,
        392,
        134,
        578,
        516,
        386,
        7,
        649,
        7,
        586,
        646,
        139,
        640,
        263,
        452,
        64,
        9,
        580,
        457,
        389,
        195,
        67,
        645,
        2,
        651,
        0,
        642,
        2,
        454,
        390,
        647,
        128,
        259,
        385,
        8,
        256,
        650,
        261,
        3,
        10,
        65,
        458,
        459,
        197,
        3,
        66,
        129,
        266,
        198,
        262,
        384,
        199,
        587,
        267,
        137,
        136,
        75,
        518,
        74,
        265,
        4,
        450,
        585,
        6,
        387,
        584,
        264,
        456,
        260,
        132,
        196,
        388,
        514,
        515,
        0,
        192,
        394,
    },
};

uint32_t NocArchGrayskull::m_bank_to_l1_offset[NUM_L1_BANKS] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    uint32_t(-524288), // -524288
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
};

NocArchGrayskull g_noc_arch_grayskull;

} // namespace

//
//    Public functions
//

NocArch *get_noc_arch_grayskull() {
    return &g_noc_arch_grayskull;
}

} // namespace device
} // namespace metal
} // namespace tt

