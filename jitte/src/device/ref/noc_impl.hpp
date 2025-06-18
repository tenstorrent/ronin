// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"
#include "core/noc_api.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

class NocImpl: public Noc {
public:
    NocImpl(
        Soc *soc,
        NocArch *noc_arch,
        uint32_t my_x,
        uint32_t my_y);
    ~NocImpl();
public:
    void wait_fast_read_ok(uint32_t noc, uint32_t cmd_buf) override;
    void wait_fast_write_ok(uint32_t noc, uint32_t cmd_buf) override;
    void wait_reads_flushed(uint32_t noc) override;
    void wait_nonposted_writes_flushed(uint32_t noc) override;
    void atomic_increment(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t addr, 
        uint32_t incr, 
        uint32_t wrap, 
        bool linked) override;
    void write_targ_addr_lo(uint32_t noc, uint32_t buf, uint32_t val) override;
    void write_targ_addr_mid(uint32_t noc, uint32_t buf, uint32_t val) override;
    void write_ret_addr_lo(uint32_t noc, uint32_t buf, uint32_t val) override;
    void write_ret_addr_mid(uint32_t noc, uint32_t buf, uint32_t val) override;
    void write_at_len_be(uint32_t noc, uint32_t buf, uint32_t val) override;
    void write_ctrl_cpy_wr(
        uint32_t noc, 
        uint32_t buf, 
        uint32_t vc, 
        bool linked, 
        bool mcast, 
        bool src,
        bool non_posted) override;
    void write_cmd_ctrl_send_req(uint32_t noc, uint32_t buf) override;
    void incr_reads_num_issued(uint32_t noc, uint32_t incr) override;
    void incr_nonposted_writes_num_issued(uint32_t noc, uint32_t incr) override;
    void incr_nonposted_writes_acked(uint32_t noc, uint32_t incr) override;
private:
    void read(
        uint32_t noc,
        uint64_t src_addr, 
        uint32_t dest_addr, 
        uint32_t len_bytes);
    void write(
        uint32_t noc,
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes);
    void write_mcast(
        uint32_t noc,
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        bool loopback_src);
    uint8_t *map_local_addr(uint32_t noc, uint32_t addr, uint32_t size);
    uint8_t *map_remote_addr(uint32_t noc, uint64_t noc_addr, uint32_t size);
    uint8_t *map_remote_addr(
        uint32_t x, 
        uint32_t y, 
        uint32_t addr, 
        uint32_t size);
    void parse_noc_addr(
        uint32_t noc,
        uint64_t noc_addr, 
        uint32_t &x, 
        uint32_t &y, 
        uint32_t &addr);
    void parse_noc_multicast_addr(
        uint32_t noc,
        uint64_t noc_addr,
        uint32_t &x_start, 
        uint32_t &y_start, 
        uint32_t &x_end, 
        uint32_t &y_end, 
        uint32_t &addr);
    uint32_t dec_noc_x(uint32_t noc, uint32_t x);
    uint32_t dec_noc_y(uint32_t noc, uint32_t y);
private:
    struct CmdRegs {
        uint32_t targ_addr_lo;
        uint32_t targ_addr_mid;
        uint32_t ret_addr_lo;
        uint32_t ret_addr_mid;
        uint32_t at_len_be;
        uint32_t ctrl_vc;
        bool ctrl_linked;
        bool ctrl_mcast;
        bool ctrl_src;
        bool ctrl_non_posted;
    };
private:
    Soc *m_soc;
    NocArch *m_noc_arch;
    uint32_t m_my_x;
    uint32_t m_my_y;
    uint32_t m_noc_size_x;
    uint32_t m_noc_size_y;
    CmdRegs m_cmd_regs[NUM_NOCS][NUM_CMD_BUFS];
    uint32_t m_reads_num_issued[NUM_NOCS];
    uint32_t m_nonposted_writes_num_issued[NUM_NOCS];
    uint32_t m_nonposted_writes_acked[NUM_NOCS];
};

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

