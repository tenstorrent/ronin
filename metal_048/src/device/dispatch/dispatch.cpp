
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"

#include "dispatch/noc.hpp"
#include "dispatch/dispatch.hpp"
#include "dispatch/cq_commands.hpp"

namespace tt {
namespace metal {
namespace device {
namespace dispatch {

namespace {

constexpr bool DIAG = false;

constexpr uint32_t L1_NOC_ALIGNMENT = 16;

uint32_t round_up_pow2(uint32_t v, uint32_t pow2_size) {
    return (v + (pow2_size - 1)) & ~(pow2_size - 1);
}

} // namespace

//
//    Dispatch
//

Dispatch::Dispatch(Soc *soc, NocArch *noc_arch):
        m_soc(soc),
        m_noc(soc, noc_arch),
        m_max_write_packed_cores(108),
        m_cmd_reg(nullptr),
        m_cmd_reg_end(nullptr),
        m_cmd_ptr(nullptr) { }

Dispatch::~Dispatch() { }

void Dispatch::configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) {
    m_read_buffer_desc.padded_page_size = padded_page_size;
    m_read_buffer_desc.dst = dst;
    m_read_buffer_desc.dst_offset = dst_offset;
    m_read_buffer_desc.num_pages_read = num_pages_read;
}

void Dispatch::run(const uint8_t *cmd_reg, uint32_t cmd_seq_size) {
    m_cmd_reg = cmd_reg;
    m_cmd_reg_end = m_cmd_reg + cmd_seq_size;
    m_cmd_ptr = m_cmd_reg;
    while (m_cmd_ptr < m_cmd_reg_end) {
        process_cmd();
    }
    assert(m_cmd_ptr == m_cmd_reg_end);
}

void Dispatch::process_cmd() {
    check_cmd_reg_limit(sizeof(CQDispatchCmd));

    const CQDispatchCmd *cmd = reinterpret_cast<const CQDispatchCmd *>(m_cmd_ptr);

    switch (cmd->base.cmd_id) {
    case CQ_DISPATCH_CMD_WRITE_LINEAR:
        if (DIAG) {
            printf("cmd_write\n");
        }
        process_write();
        break;

    case CQ_DISPATCH_CMD_WRITE_PAGED:
        if (DIAG) {
            printf("cmd_write_paged is_dram: %d\n", int(cmd->write_paged.is_dram));
        }
        if (cmd->write_paged.is_dram) {
            process_write_paged(true);
        } else {
            process_write_paged(false);
        }
        break;

    case CQ_DISPATCH_CMD_WRITE_PACKED:
        {
            if (DIAG) {
                printf("cmd_write_packed\n");
            }
            uint32_t flags = cmd->write_packed.flags;
            if (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST) {
                process_write_packed(true, flags);
            } else {
                process_write_packed(false, flags);
            }
        }
        break;

    // SKIPPED: CQ_DISPATCH_CMD_WRITE_LINEAR_H (unused)

    case CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST:
        if (DIAG) {
            printf("cmd_write_linear_h_host\n");
        }
        process_write_host();
        break;

    // SKIPPED: CQ_DISPATCH_CMD_GO (unused)
    // SKIPPED: CQ_DISPATCH_CMD_SINK (unused)

    case CQ_DISPATCH_CMD_WAIT:
        if (DIAG) {
            printf("cmd_wait\n");
        }
        process_wait();
        break;

    // SKIPPED: CQ_DISPATCH_CMD_DEBUG (unused)
    // SKIPPED: CQ_DISPATCH_CMD_DELAY (unused)

    case CQ_DISPATCH_CMD_TERMINATE:
        if (DIAG) {
            printf("cmd_terminate\n");
        }
        process_terminate();
        break;

    default:
        throw std::runtime_error(
            "Invalid dispatch command: " +  std::to_string(int(cmd->base.cmd_id)) + 
            " at " + std::to_string(m_cmd_ptr - m_cmd_reg));
    }
}

void Dispatch::process_write() {
    const CQDispatchCmd *cmd = reinterpret_cast<const CQDispatchCmd *>(m_cmd_ptr);
    uint32_t num_mcast_dests = cmd->write_linear.num_mcast_dests;
    if (num_mcast_dests == 0) {
        process_write_linear(false, 0);
    } else {
        process_write_linear(true, num_mcast_dests);
    }
}

void Dispatch::process_write_linear(bool multicast, uint32_t num_mcast_dests) {
    const CQDispatchCmd *cmd = reinterpret_cast<const CQDispatchCmd *>(m_cmd_ptr);

    uint32_t dst_noc = cmd->write_linear.noc_xy_addr;
    uint32_t dst_addr = cmd->write_linear.addr;
    uint32_t length = cmd->write_linear.length;
    const uint8_t *data_ptr = m_cmd_ptr + sizeof(CQDispatchCmd);

    if (DIAG) {
        printf("dispatch_write: %d num_mcast_dests: %d\n", length, num_mcast_dests);
    }

    check_cmd_reg_limit(sizeof(CQDispatchCmd) + length);

    uint64_t dst = m_noc.get_noc_addr_helper(dst_noc, dst_addr);
    if (multicast){
        m_noc.write_multicast(data_ptr, dst, length, num_mcast_dests);
    } else {
        m_noc.write(data_ptr, dst, length);
    }

    m_cmd_ptr = data_ptr + length;
}

void Dispatch::process_write_paged(bool is_dram) {
    const CQDispatchCmd *cmd = reinterpret_cast<const CQDispatchCmd *>(m_cmd_ptr);

    uint32_t page_id = cmd->write_paged.start_page;
    uint32_t base_addr = cmd->write_paged.base_addr;
    uint32_t page_size = cmd->write_paged.page_size;
    uint32_t pages = cmd->write_paged.pages;
    const uint8_t *data_ptr = m_cmd_ptr + sizeof(CQDispatchCmd);
    uint32_t write_length = pages * page_size;

    assert(write_length % page_size == 0);

    if (DIAG) {
        printf(
            "process_write_paged - pages: %d page_size: %d start_page: %d base_addr: %x\n",
                pages, page_size, page_id, base_addr);
    }

    check_cmd_reg_limit(sizeof(CQDispatchCmd) + write_length);

    while (write_length != 0) {
        uint64_t dst = 
            m_noc.get_noc_addr_interleaved(
                is_dram, 
                base_addr, 
                page_size, 
                page_id, 
                0);
        m_noc.write(data_ptr, dst, page_size);
        page_id++;
        write_length -= page_size;
        data_ptr += page_size;
    }

    m_cmd_ptr = data_ptr;
}

//
// Packed write command
// Layout looks like:
//   - CQDispatchCmd struct
//   - count CQDispatchWritePackedSubCmd structs (max 1020)
//   - pad to L1 alignment
//   - count data packets of size size, each L1 aligned
//
// Note that there are multiple size restrictions on this cmd:
//  - all sub_cmds fit in one page
//  - size fits in one page
//

void Dispatch::process_write_packed(bool mcast, uint32_t flags) {
    const CQDispatchCmd *cmd = reinterpret_cast<const CQDispatchCmd *>(m_cmd_ptr);

    uint32_t count = cmd->write_packed.count;
    assert(count <= (mcast ? (m_max_write_packed_cores / 2) : m_max_write_packed_cores));
    uint32_t sub_cmd_size = 
        mcast ?
            sizeof(CQDispatchWritePackedMulticastSubCmd) :
            sizeof(CQDispatchWritePackedUnicastSubCmd);

    uint32_t xfer_size = cmd->write_packed.size;
    uint32_t dst_addr = cmd->write_packed.addr;

    uint32_t data_start = sizeof(CQDispatchCmd) + count * sub_cmd_size;
    data_start = round_up_pow2(data_start, L1_NOC_ALIGNMENT);
    const uint8_t *data_ptr = m_cmd_ptr + data_start;

    uint32_t padded_xfer_size = round_up_pow2(xfer_size, L1_NOC_ALIGNMENT);
    uint32_t stride = 
        (flags & CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE) ? 
            0 : 
            padded_xfer_size;

    if (DIAG) {
        printf(
            "dispatch_write_packed: count %d data_start %d xfer_size %d stride %d\n",
                count, data_start, xfer_size, stride);
    }

    check_cmd_reg_limit(data_start + stride * (count - 1) + padded_xfer_size);

    const uint8_t *sub_cmd_ptr = m_cmd_ptr + sizeof(CQDispatchCmd);
    while (count != 0) {
        uint32_t dst_noc = 0;
        uint32_t num_dests = 0;
        if (mcast) {
            const CQDispatchWritePackedMulticastSubCmd *sub_cmd =
                reinterpret_cast<const CQDispatchWritePackedMulticastSubCmd *>(sub_cmd_ptr);
            dst_noc = sub_cmd->noc_xy_addr;
            num_dests = sub_cmd->num_mcast_dests;
        } else {
            const CQDispatchWritePackedUnicastSubCmd *sub_cmd =
                reinterpret_cast<const CQDispatchWritePackedUnicastSubCmd *>(sub_cmd_ptr);
            dst_noc = sub_cmd->noc_xy_addr;
            num_dests = 1;
        }
        sub_cmd_ptr += sub_cmd_size;
        uint64_t dst = m_noc.get_noc_addr_helper(dst_noc, dst_addr);

        if (mcast) {
            m_noc.write_multicast(data_ptr, dst, xfer_size, num_dests);
        } else {
            m_noc.write(data_ptr, dst, xfer_size);
        }

        count--;
        data_ptr += stride;
    }

    if (stride == 0) {
        data_ptr += padded_xfer_size;
    }

    m_cmd_ptr = data_ptr;
}

void Dispatch::process_write_host() {
    const CQDispatchCmd *cmd = reinterpret_cast<const CQDispatchCmd *>(m_cmd_ptr);

    uint32_t length = cmd->write_linear_host.length;
    if (DIAG) {
        printf("process_write_host_h: %d\n", length);
    }
    const uint8_t *data_ptr = m_cmd_ptr;

    check_cmd_reg_limit(length);

    uint32_t padded_page_size = m_read_buffer_desc.padded_page_size;
    void *dst = m_read_buffer_desc.dst;
    uint32_t dst_offset = m_read_buffer_desc.dst_offset;
    uint32_t num_pages_read = m_read_buffer_desc.num_pages_read;
    uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(dst) + dst_offset;

#if 0 // TODO Revise this
    assert(padded_page_size * num_pages_read == length);
#endif

    // don't copy command struct to "user space"
    // by command construction, "write_linear_host.length" includes command struct
    data_ptr += sizeof(CQDispatchCmd);
    length -= sizeof(CQDispatchCmd);

    assert(padded_page_size * num_pages_read == length);

    memcpy(dst_ptr, data_ptr, length);

    m_cmd_ptr = data_ptr + length;
}

void Dispatch::process_wait() {
    // nothing to do so far
    m_cmd_ptr += sizeof(CQDispatchCmd);
}

void Dispatch::process_terminate() {
    // nothing to do so far
    m_cmd_ptr += sizeof(CQDispatchCmd);
}

void Dispatch::check_cmd_reg_limit(uint32_t length) {
    if (m_cmd_ptr + length > m_cmd_reg_end) {
        throw std::runtime_error(
            "Dispatch command region overflow: got " + 
                std::to_string(m_cmd_reg_end - m_cmd_ptr) +
                " want " + std::to_string(length));
    }
}

} // namespace dispatch
} // namespace device
} // namespace metal
} // namespace tt

