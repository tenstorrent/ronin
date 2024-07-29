#pragma once

#include <cstdint>

namespace tt {
namespace metal {
namespace device {

class Noc {
public:
    Noc();
    virtual ~Noc();
public:
    void fast_read_any_len(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t src_addr, 
        uint32_t dest_addr, 
        uint32_t len_bytes);
    void fast_write_any_len(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests);
    void fast_write_any_len_loopback_src(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests);
    void fast_read(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t src_addr, 
        uint32_t dest_addr, 
        uint32_t len_bytes);
    void fast_write(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests);
    void fast_write_loopback_src(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint32_t src_addr, 
        uint64_t dest_addr, 
        uint32_t len_bytes, 
        uint32_t vc, 
        bool mcast, 
        bool linked, 
        uint32_t num_dests);
    void fast_atomic_increment(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t addr, 
        uint32_t incr, 
        uint32_t wrap, 
        bool linked);
    uint32_t max_burst_size();
public:
    virtual void wait_fast_read_ok(uint32_t noc, uint32_t cmd_buf) = 0;
    virtual void wait_fast_write_ok(uint32_t noc, uint32_t cmd_buf) = 0;
    virtual void wait_reads_flushed(uint32_t noc) = 0;
    virtual void wait_nonposted_writes_flushed(uint32_t noc) = 0;
    virtual void atomic_increment(
        uint32_t noc, 
        uint32_t cmd_buf, 
        uint64_t addr, 
        uint32_t incr, 
        uint32_t wrap, 
        bool linked) = 0;
    virtual void write_targ_addr_lo(uint32_t noc, uint32_t buf, uint32_t val) = 0;
    virtual void write_targ_addr_mid(uint32_t noc, uint32_t buf, uint32_t val) = 0;
    virtual void write_ret_addr_lo(uint32_t noc, uint32_t buf, uint32_t val) = 0;
    virtual void write_ret_addr_mid(uint32_t noc, uint32_t buf, uint32_t val) = 0;
    virtual void write_at_len_be(uint32_t noc, uint32_t buf, uint32_t val) = 0;
    virtual void write_ctrl_cpy_wr(
        uint32_t noc, 
        uint32_t buf, 
        uint32_t vc, 
        bool linked, 
        bool mcast, 
        bool src,
        bool non_posted) = 0;
    virtual void write_cmd_ctrl_send_req(uint32_t noc, uint32_t buf) = 0;
    virtual void incr_reads_num_issued(uint32_t noc, uint32_t incr) = 0;
    virtual void incr_nonposted_writes_num_issued(uint32_t noc, uint32_t incr) = 0;
    virtual void incr_nonposted_writes_acked(uint32_t noc, uint32_t incr) = 0;
public:
    static constexpr uint32_t NUM_NOCS = 2;
    static constexpr uint32_t NUM_CMD_BUFS = 4;
public:
    static constexpr uint32_t WR_CMD_BUF = 0;
    static constexpr uint32_t RD_CMD_BUF = 1;
    static constexpr uint32_t WR_REG_CMD_BUF = 2;
    static constexpr uint32_t AT_CMD_BUF = 3;
public:
    static constexpr uint32_t UNICAST_WRITE_VC = 1;
    static constexpr uint32_t MULTICAST_WRITE_VC = 4;
};

} // namespace device
} // namespace metal
} // namespace tt

