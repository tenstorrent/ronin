// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <string>
#include <cassert>
#include <stdexcept>

#include "arch/soc_arch.hpp"

#include "core/noc_api.hpp"

#include "ref/noc_impl.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

namespace {

void data_copy(uint8_t *dest, uint8_t *src, uint32_t len) {
    // Use memmove to support overlapped regions (is overlapping allowed?)
    memmove(dest, src, len);
}

std::string xy_to_string(uint32_t x, uint32_t y) {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

} // namespace

//
//    NocImpl
//

NocImpl::NocImpl(
        Soc *soc,
        NocArch *noc_arch,
        uint32_t my_x,
        uint32_t my_y):
            m_soc(soc),
            m_noc_arch(noc_arch),
            m_my_x(my_x),
            m_my_y(my_y),
            m_noc_size_x(noc_arch->noc_size_x()),
            m_noc_size_y(noc_arch->noc_size_y()) { 
    for (uint32_t i = 0; i < NUM_NOCS; i++) {
        for (uint32_t k = 0; k < NUM_CMD_BUFS; k++) {
            CmdRegs &regs = m_cmd_regs[i][k];
            regs.targ_addr_lo = 0;
            regs.targ_addr_mid = 0;
            regs.ret_addr_lo = 0;
            regs.ret_addr_mid = 0;
            regs.at_len_be = 0;
            regs.ctrl_vc = 0;
            regs.ctrl_linked = false;
            regs.ctrl_mcast = false;
            regs.ctrl_src = false;
            regs.ctrl_non_posted = false;
        }
        m_reads_num_issued[i] = 0;
        m_nonposted_writes_num_issued[i] = 0;
        m_nonposted_writes_acked[i] = 0;
    }
}

NocImpl::~NocImpl() { }

void NocImpl::wait_fast_read_ok(uint32_t noc, uint32_t cmd_buf) {
    // nothing to do
}

void NocImpl::wait_fast_write_ok(uint32_t noc, uint32_t cmd_buf) {
    // nothing to do
}

void NocImpl::wait_reads_flushed(uint32_t noc) {
    // nothing to do
}

void NocImpl::wait_nonposted_writes_flushed(uint32_t noc) {
    // nothing to do
}

void NocImpl::atomic_increment(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t addr, 
        uint32_t incr, 
        uint32_t wrap, 
        bool linked) {
    assert(noc < NUM_NOCS);
    assert(cmd_buf == AT_CMD_BUF);
    uint32_t *ptr = reinterpret_cast<uint32_t *>(map_remote_addr(noc, addr, sizeof(uint32_t)));
    // see 'noc_atomic_increment' in [hw/inc/grayskull/noc/noc.h] 
    if (wrap >= 31) {
        *ptr += incr;
    } else {
        uint32_t mask = (1 << (wrap + 1)) - 1;
        uint32_t val = *ptr;
        uint32_t hi = val & ~mask;
        uint32_t lo = val & mask;
        *ptr = hi + ((lo + incr) & mask);
    }
}

void NocImpl::write_targ_addr_lo(uint32_t noc, uint32_t buf, uint32_t val) {
    assert(noc < NUM_NOCS);
    assert(buf < NUM_CMD_BUFS);
    m_cmd_regs[noc][buf].targ_addr_lo = val;
}

void NocImpl::write_targ_addr_mid(uint32_t noc, uint32_t buf, uint32_t val) {
    assert(noc < NUM_NOCS);
    assert(buf < NUM_CMD_BUFS);
    m_cmd_regs[noc][buf].targ_addr_mid = val;
}

void NocImpl::write_ret_addr_lo(uint32_t noc, uint32_t buf, uint32_t val) {
    assert(noc < NUM_NOCS);
    assert(buf < NUM_CMD_BUFS);
    m_cmd_regs[noc][buf].ret_addr_lo = val;
}

void NocImpl::write_ret_addr_mid(uint32_t noc, uint32_t buf, uint32_t val) {
    assert(noc < NUM_NOCS);
    assert(buf < NUM_CMD_BUFS);
    m_cmd_regs[noc][buf].ret_addr_mid = val;
}

void NocImpl::write_at_len_be(uint32_t noc, uint32_t buf, uint32_t val) {
    assert(noc < NUM_NOCS);
    assert(buf < NUM_CMD_BUFS);
    m_cmd_regs[noc][buf].at_len_be = val;
}

void NocImpl::write_ctrl_cpy_wr(
        uint32_t noc, 
        uint32_t buf, 
        uint32_t vc, 
        bool linked, 
        bool mcast, 
        bool src,
        bool non_posted) {
    assert(noc < NUM_NOCS);
    assert(buf < NUM_CMD_BUFS);
    CmdRegs &regs = m_cmd_regs[noc][buf];
    regs.ctrl_vc = vc;
    regs.ctrl_linked = linked;
    regs.ctrl_mcast = mcast;
    regs.ctrl_src = src;
    regs.ctrl_non_posted = non_posted;
}

void NocImpl::write_cmd_ctrl_send_req(uint32_t noc, uint32_t buf) {
    assert(noc < NUM_NOCS);
    assert(buf < NUM_CMD_BUFS);
    CmdRegs &regs = m_cmd_regs[noc][buf];
    switch (buf) {
    case RD_CMD_BUF:
        {
            uint64_t src_addr = 
                (uint64_t(regs.targ_addr_mid) << 32) | uint64_t(regs.targ_addr_lo);
            uint32_t dest_addr = regs.ret_addr_lo;
            uint32_t len_bytes = regs.at_len_be;
            read(noc, src_addr, dest_addr, len_bytes);
        }
        break;
    case WR_CMD_BUF:
        {
            uint32_t src_addr = regs.targ_addr_lo;
            uint64_t dest_addr =
                (uint64_t(regs.ret_addr_mid) << 32) | uint64_t(regs.ret_addr_lo);
            uint32_t len_bytes = regs.at_len_be;
            write(noc, src_addr, dest_addr, len_bytes);
        }
        break;
    case WR_REG_CMD_BUF:
        {
            uint32_t src_addr = regs.targ_addr_lo;
            uint64_t dest_addr =
                (uint64_t(regs.ret_addr_mid) << 32) | uint64_t(regs.ret_addr_lo);
            uint32_t len_bytes = regs.at_len_be;
            if (!regs.ctrl_mcast) {
                write(noc, src_addr, dest_addr, len_bytes);
            } else {
                write_mcast(noc, src_addr, dest_addr, len_bytes, regs.ctrl_src);
            }
        }
        break;
    case AT_CMD_BUF:
    default:
        assert(false);
        break;
    }
}

void NocImpl::incr_reads_num_issued(uint32_t noc, uint32_t incr) {
    assert(noc < NUM_NOCS);
    m_reads_num_issued[noc] += incr;
}

void NocImpl::incr_nonposted_writes_num_issued(uint32_t noc, uint32_t incr) {
    assert(noc < NUM_NOCS);
    m_nonposted_writes_num_issued[noc] += incr;
}

void NocImpl::incr_nonposted_writes_acked(uint32_t noc, uint32_t incr) {
    assert(noc < NUM_NOCS);
    m_nonposted_writes_acked[noc] += incr;
}

void NocImpl::read(
        uint32_t noc,
        uint64_t src_addr, 
        uint32_t dest_addr, 
        uint32_t len_bytes) {
    uint8_t *src = map_remote_addr(noc, src_addr, len_bytes);
    uint8_t *dest = map_local_addr(noc, dest_addr, len_bytes);
    data_copy(dest, src, len_bytes);
}

void NocImpl::write(
        uint32_t noc,
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes) {
    uint8_t *src = map_local_addr(noc, src_addr, len_bytes);
    uint8_t *dest = map_remote_addr(noc, dest_addr, len_bytes);
    data_copy(dest, src, len_bytes);
}

void NocImpl::write_mcast(
        uint32_t noc,
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        bool loopback_src) {
    uint8_t *src = map_local_addr(noc, src_addr, len_bytes);
    uint32_t x_start = 0;
    uint32_t y_start = 0;
    uint32_t x_end = 0;
    uint32_t y_end = 0;
    uint32_t addr = 0;
    parse_noc_multicast_addr(
        noc,
        dest_addr,
        x_start, 
        y_start, 
        x_end, 
        y_end, 
        addr);
#if 0 // ACHTUNG: Experimental (TODO: Study this)
    if (!loopback_src && 
            m_my_x >= x_start && m_my_x <= x_end && 
            m_my_y >= y_start && m_my_y <= y_end) {
        throw std::runtime_error(
            "Multicast sender cannot be part of mulstcast destinations");
    } 
#endif
    // ACHTUNG: Temporary workaround - is this valid or cheaing?
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
#if 1 // ACHTUNG: Experimental (TODO: Study this)
            if (!loopback_src && x == m_my_x && y == m_my_y) {
                continue;
            }
#endif
            uint8_t *dest = map_remote_addr(x, y, addr, len_bytes);
            data_copy(dest, src, len_bytes);
        }
    }
}

uint8_t *NocImpl::map_local_addr(uint32_t noc, uint32_t addr, uint32_t size) {
    assert(addr + size <= m_soc->worker_l1_size());
    return m_soc->map_l1_addr(int(m_my_x), int(m_my_y), addr);
}

uint8_t *NocImpl::map_remote_addr(uint32_t noc, uint64_t noc_addr, uint32_t size) {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t addr = 0;
    parse_noc_addr(noc, noc_addr, x, y, addr);
    return map_remote_addr(x, y, addr, size);
}

uint8_t *NocImpl::map_remote_addr(
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
#if 1
    // TODO: Revise this
    // Temporary workaround (need proper error handling in coroutines)
    printf("No DRAM or worker core at [%d %d]: core type %d\n", x, y, int(core_type));
#endif
    throw std::runtime_error("No DRAM or worker core at " + xy_to_string(x, y));
    return nullptr;
}

void NocImpl::parse_noc_addr(
        uint32_t noc,
        uint64_t noc_addr, 
        uint32_t &x, 
        uint32_t &y, 
        uint32_t &addr) {
    m_noc_arch->parse_noc_addr(noc_addr, x, y, addr);
    x = dec_noc_x(noc, x);
    y = dec_noc_y(noc, y);
}

void NocImpl::parse_noc_multicast_addr(
        uint32_t noc,
        uint64_t noc_addr,
        uint32_t &x_start, 
        uint32_t &y_start, 
        uint32_t &x_end, 
        uint32_t &y_end, 
        uint32_t &addr) {
    m_noc_arch->parse_noc_multicast_addr(
        noc_addr, 
        x_start, 
        y_start, 
        x_end, 
        y_end, 
        addr);
    x_start = dec_noc_x(noc, x_start);
    y_start = dec_noc_y(noc, y_start);
    x_end = dec_noc_x(noc, x_end);
    y_end = dec_noc_y(noc, y_end);
}

uint32_t NocImpl::dec_noc_x(uint32_t noc, uint32_t x) {
    return (noc == 0) ? x : m_noc_size_x - 1 - x;
}

uint32_t NocImpl::dec_noc_y(uint32_t noc, uint32_t y) {
    return (noc == 0) ? y : m_noc_size_y - 1 - y;
}

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

