
#include <cstdint>

#include "core/noc_api.hpp"

namespace tt {
namespace metal {
namespace device {

//
//    Noc
//

Noc::Noc() { }

Noc::~Noc() { }

void Noc::fast_read_any_len(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t src_addr, 
        uint32_t dest_addr, 
        uint32_t len_bytes) {
    uint32_t size = max_burst_size();
    while (len_bytes > size) {
        wait_fast_read_ok(noc, cmd_buf);
        fast_read(noc, cmd_buf, src_addr, dest_addr, size);
        src_addr += size;
        dest_addr += size;
        len_bytes -= size;
    }
    wait_fast_read_ok(noc, cmd_buf);
    fast_read(noc, cmd_buf, src_addr, dest_addr, len_bytes);
}

void Noc::fast_write_any_len(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests) {
    uint32_t size = max_burst_size();
    while (len_bytes > size) {
        wait_fast_write_ok(noc, cmd_buf);
        fast_write(noc, cmd_buf, src_addr, dest_addr, size, vc, mcast, linked, num_dests);
        src_addr += size;
        dest_addr += size;
        len_bytes -= size;
    }
    wait_fast_write_ok(noc, cmd_buf);
    fast_write(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests);
}

void Noc::fast_write_any_len_loopback_src(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests) {
    uint32_t size = max_burst_size();
    while (len_bytes > size) {
        wait_fast_write_ok(noc, cmd_buf);
        fast_write_loopback_src(
            noc, 
            cmd_buf, 
            src_addr, 
            dest_addr, 
            size, 
            vc, 
            mcast, 
            linked, 
            num_dests);
        src_addr += size;
        dest_addr += size;
        len_bytes -= size;
    }
    wait_fast_write_ok(noc, cmd_buf);
    fast_write_loopback_src(
        noc, 
        cmd_buf, 
        src_addr, 
        dest_addr, 
        len_bytes, 
        vc, 
        mcast, 
        linked, 
        num_dests);
}

void Noc::fast_read(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t src_addr, 
        uint32_t dest_addr, 
        uint32_t len_bytes) {
    if (len_bytes > 0) {
        write_ret_addr_lo(noc, cmd_buf, dest_addr);
        write_targ_addr_lo(noc, cmd_buf, uint32_t(src_addr));
        write_targ_addr_mid(noc, cmd_buf, uint32_t(src_addr >> 32));
        write_at_len_be(noc, cmd_buf, len_bytes);
        write_cmd_ctrl_send_req(noc, cmd_buf);
        incr_reads_num_issued(noc, 1);
    }
}

void Noc::fast_write(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests) {
    if (len_bytes > 0) {
        write_ctrl_cpy_wr(noc, cmd_buf, vc, linked, mcast, false, true);
        write_targ_addr_lo(noc, cmd_buf, src_addr);
        write_ret_addr_lo(noc, cmd_buf, uint32_t(dest_addr));
        write_ret_addr_mid(noc, cmd_buf, uint32_t(dest_addr >> 32));
        write_at_len_be(noc, cmd_buf, len_bytes);
        write_cmd_ctrl_send_req(noc, cmd_buf);
        incr_nonposted_writes_num_issued(noc, 1);
        incr_nonposted_writes_acked(noc, num_dests);
    }
}

void Noc::fast_write_loopback_src(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests) {
    if (len_bytes > 0) {
        write_ctrl_cpy_wr(noc, cmd_buf, vc, linked, mcast, true, true);
        write_targ_addr_lo(noc, cmd_buf, src_addr);
        write_ret_addr_lo(noc, cmd_buf, uint32_t(dest_addr));
        write_ret_addr_mid(noc, cmd_buf, uint32_t(dest_addr >> 32));
        write_at_len_be(noc, cmd_buf, len_bytes);
        write_cmd_ctrl_send_req(noc, cmd_buf);
        incr_nonposted_writes_num_issued(noc, 1);
        incr_nonposted_writes_acked(noc, num_dests);
    }
}

void Noc::fast_atomic_increment(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t addr, 
        uint32_t incr, 
        uint32_t wrap, 
        bool linked) {
    wait_fast_write_ok(noc, cmd_buf);
    atomic_increment(
        noc, 
        cmd_buf, 
        addr, 
        incr, 
        wrap, 
        linked);
}

uint32_t Noc::max_burst_size() {
    // architecture-specific in general
    // this simplified implementation works for Grayskull and Wormhole B0
    return 8192;
}

} // namespace device
} // namespace metal
} // namespace tt

