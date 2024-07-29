#pragma once

#include <cstdint>

namespace tt {
namespace metal {
namespace device {

//
//    NocArch
//

class NocArch {
public:
    NocArch() { }
    virtual ~NocArch() { }
public:
    virtual uint32_t num_dram_banks() = 0;
    virtual uint32_t num_l1_banks() = 0;
    virtual uint32_t noc_size_x() = 0;
    virtual uint32_t noc_size_y() = 0;
    virtual uint32_t pcie_noc_x() = 0;
    virtual uint32_t pcie_noc_y() = 0;
    // noc_parameters
    virtual uint64_t noc_xy_addr(uint32_t x, uint32_t y, uint32_t addr) = 0;
    virtual uint64_t noc_multicast_addr(
        uint32_t x_start, 
        uint32_t y_start, 
        uint32_t x_end, 
        uint32_t y_end, 
        uint32_t addr) = 0;
    virtual uint32_t noc_xy_encoding(uint32_t x, uint32_t y) = 0;
    virtual uint32_t noc_multicast_encoding(
        uint32_t x_start, 
        uint32_t y_start, 
        uint32_t x_end, 
        uint32_t y_end) = 0;
    virtual uint64_t noc_xy_addr2(uint32_t xy, uint32_t addr) = 0;
    // noc address parsing
    virtual void parse_noc_addr(
        uint64_t noc_addr, 
        uint32_t &x, 
        uint32_t &y, 
        uint32_t &addr) = 0;
    virtual void parse_noc_multicast_addr(
        uint64_t noc_addr,
        uint32_t &x_start, 
        uint32_t &y_start, 
        uint32_t &x_end, 
        uint32_t &y_end, 
        uint32_t &addr) = 0;
    // bank_to_noc_coord_mapping
    virtual uint32_t dram_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) = 0;
    virtual uint32_t bank_to_dram_offset(uint32_t bank_id) = 0;
    virtual uint32_t l1_bank_to_noc_xy(uint32_t noc_index, uint32_t bank_id) = 0;
    virtual uint32_t bank_to_l1_offset(uint32_t bank_id) = 0;
};

//
//    Public functions
//

NocArch *get_noc_arch_grayskull();
NocArch *get_noc_arch_wormhole_b0();

} // namespace device
} // namespace metal
} // namespace tt

