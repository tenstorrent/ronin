// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <stdexcept>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"

#include "dispatch/noc.hpp"
#include "dispatch/prefetch.hpp"
#include "dispatch/cq_commands.hpp"

namespace tt {
namespace metal {
namespace device {
namespace dispatch {

namespace {

constexpr bool DIAG = false;

constexpr uint32_t max_read_packed_cmd =
    CQ_PREFETCH_CMD_RELAY_PAGED_PACKED_MAX_SUB_CMDS *
        sizeof(CQPrefetchRelayPagedPackedSubCmd) / sizeof(uint32_t);
constexpr uint32_t l1_cache_elements = max_read_packed_cmd + 1;  // +1 for sentinel value

} // namespace

//
//    Prefetch
//

Prefetch::Prefetch(
        Soc *soc, 
        NocArch *noc_arch, 
        Dispatch *dispatch):
            m_soc(soc),
            m_noc(soc, noc_arch),
            m_dispatch(dispatch),
            m_cmd_reg(nullptr),
            m_cmd_reg_end(nullptr),
            m_cmd_ptr(nullptr) { 
    m_l1_cache.resize(l1_cache_elements);            
}

Prefetch::~Prefetch() { }

void Prefetch::run(const uint8_t *cmd_reg, uint32_t cmd_seq_size) {
    m_cmd_reg = cmd_reg;
    m_cmd_reg_end = m_cmd_reg + cmd_seq_size;
    m_cmd_ptr = cmd_reg;
    while (m_cmd_ptr < m_cmd_reg_end) {
        uint32_t stride;
        process_cmd(stride);
        m_cmd_ptr += stride;
    }
    assert(m_cmd_ptr == m_cmd_reg_end);
}

void Prefetch::process_cmd(uint32_t &stride) {
    check_cmd_reg_limit(sizeof(CQPrefetchCmd));

    const CQPrefetchCmd *cmd = reinterpret_cast<const CQPrefetchCmd *>(m_cmd_ptr);

    switch (cmd->base.cmd_id) {
    case CQ_PREFETCH_CMD_RELAY_LINEAR:
        if (DIAG) {
            printf("relay linear\n");
        }
        stride = process_relay_linear_cmd();
        break;

    case CQ_PREFETCH_CMD_RELAY_PAGED:
        if (DIAG) {
            printf("relay dram page\n");
        }
        {
            uint32_t packed_page_flags = cmd->relay_paged.packed_page_flags;
            uint32_t is_dram = packed_page_flags & (1 << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT);
            uint32_t start_page =
                (packed_page_flags >> CQ_PREFETCH_RELAY_PAGED_START_PAGE_SHIFT) &
                CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK;
            stride = process_relay_paged_cmd((is_dram != 0), start_page);
        }
        break;

    case CQ_PREFETCH_CMD_RELAY_PAGED_PACKED:
        if (DIAG) {
            printf("relay paged packed\n");
        }
        stride = process_relay_paged_packed_cmd();
        break;

    case CQ_PREFETCH_CMD_RELAY_INLINE:
        if (DIAG) {
            printf("relay inline\n");
        }
        // NOTE: No support for original "exec_buf"
        stride = process_relay_inline_cmd();
        break;

    case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
        if (DIAG) {
            printf("inline no flush\n");
        }
        stride = process_relay_inline_noflush_cmd();
        break;

    // SKIPPED: CQ_PREFETCH_CMD_EXEC_BUF
    // SKIPPED: CQ_PREFETCH_CMD_EXEC_BUF_END

    case CQ_PREFETCH_CMD_STALL:
        if (DIAG) {
            printf("stall\n");
        }
        stride = process_stall();
        break;

    // SKIPPED: CQ_PREFETCH_CMD_DEBUG (unused)

    case CQ_PREFETCH_CMD_TERMINATE:
        if (DIAG) {
            printf("prefetch terminating\n");
        }
        stride = process_terminate();
        break;

    default:
        throw std::runtime_error(
            "Invalid prefetch command: " +  std::to_string(int(cmd->base.cmd_id)) + 
            " at " + std::to_string(m_cmd_ptr - m_cmd_reg));
    }
}

uint32_t Prefetch::process_relay_linear_cmd() {
    const CQPrefetchCmd *cmd = reinterpret_cast<const CQPrefetchCmd *>(m_cmd_ptr);
    uint32_t noc_xy_addr = cmd->relay_linear.noc_xy_addr;
    uint32_t read_addr = cmd->relay_linear.addr;
    uint32_t length = cmd->relay_linear.length;

    if (DIAG) {
        printf(
            "relay_linear: length %d read_addr %d noc_xy_addr %x\n",
                length, read_addr, noc_xy_addr);
    }

    uint64_t noc_addr = m_noc.get_noc_addr_helper(noc_xy_addr, read_addr);

    size_t offset = m_dispatch_data.size();
    m_dispatch_data.resize(offset + length);
    uint8_t *dst = m_dispatch_data.data() + offset;

    m_noc.read(noc_addr, dst, length);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t Prefetch::process_relay_paged_cmd(bool is_dram, uint32_t page_id) {
    const CQPrefetchCmd *cmd = reinterpret_cast<const CQPrefetchCmd *>(m_cmd_ptr);
    uint32_t base_addr = cmd->relay_paged.base_addr;
    uint32_t page_size = cmd->relay_paged.page_size;
    uint32_t pages = cmd->relay_paged.pages;
    uint32_t length_adjust = cmd->relay_paged.length_adjust;

    assert(length_adjust < page_size);

    uint32_t read_length = pages * page_size;
    size_t offset = m_dispatch_data.size();
    m_dispatch_data.resize(offset + read_length);
    uint8_t *dst = m_dispatch_data.data() + offset;

    uint32_t amt_to_read = read_length;
    while (amt_to_read != 0) {
        uint64_t noc_addr = 
            m_noc.get_noc_addr_interleaved(
                is_dram, 
                base_addr, 
                page_size, 
                page_id, 
                0);
        m_noc.read(noc_addr, dst, page_size);
        page_id++;
        amt_to_read -= page_size;
        dst += page_size;
    }


    m_dispatch_data.resize(m_dispatch_data.size() - length_adjust);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t Prefetch::process_relay_paged_packed_cmd() {
    const CQPrefetchCmd *cmd = reinterpret_cast<const CQPrefetchCmd *>(m_cmd_ptr);
    uint32_t total_length = cmd->relay_paged_packed.total_length;
    uint32_t sub_cmds_length = 
        cmd->relay_paged_packed.count * sizeof(CQPrefetchRelayPagedPackedSubCmd);
    uint32_t stride = cmd->relay_paged_packed.stride;
    assert(total_length > 0);

    const uint8_t *data_ptr = m_cmd_ptr + sizeof(CQPrefetchCmd);
    uint32_t amt = sub_cmds_length / sizeof(uint32_t);

    // ACHTUNG: Do we really need "l1_cache" in emulator context?
    copy_to_local_cache(data_ptr, amt);
    // Store a sentinal non 0 value at the end to save a test/branch in read path
    reinterpret_cast<CQPrefetchRelayPagedPackedSubCmd *>(&m_l1_cache[amt])->length = 1;

    process_relay_paged_packed_sub_cmds(total_length);
    return stride;
}

void Prefetch::process_relay_paged_packed_sub_cmds(uint32_t total_length) {
    CQPrefetchRelayPagedPackedSubCmd *sub_cmd = 
        reinterpret_cast<CQPrefetchRelayPagedPackedSubCmd *>(m_l1_cache.data());

    size_t offset = m_dispatch_data.size();
    m_dispatch_data.resize(offset + total_length);
    uint8_t *dst = m_dispatch_data.data() + offset;

    uint32_t read_length = sub_cmd->length;
    uint32_t amt_to_read = total_length;
    uint32_t amt_read = 0;
    // ACHTUNG: Why not (amt_read < amt_to_read) ?
    while (read_length <= amt_to_read) {
        uint32_t page_id = sub_cmd->start_page;
        uint32_t log_page_size = sub_cmd->log_page_size;
        uint32_t base_addr = sub_cmd->base_addr;
        sub_cmd++;

        uint32_t page_size = 1 << log_page_size;

        uint32_t amt_to_read2 = read_length;
        uint32_t amt_read2 = 0;
        while (amt_read2 < amt_to_read2) {
            uint64_t noc_addr = 
                m_noc.get_noc_addr_interleaved(
                    true, 
                    base_addr, 
                    page_size, 
                    page_id, 
                    0);
            uint32_t read_size = 
                (amt_to_read2 - amt_read2 >= page_size) ? 
                    page_size : amt_to_read2 - amt_read2;
            m_noc.read(noc_addr, dst, read_size);
            page_id++;
            amt_read2 += read_size;
            dst += read_size;
        }

        amt_read += amt_read2;
        amt_to_read -= amt_read2;

        // NOTE: below can walk off the end of the sub_cmds
        //     this is ok as we store a sentinel non-zero value
        read_length = sub_cmd->length;
    }
}

uint32_t Prefetch::process_relay_inline_cmd() {
    const CQPrefetchCmd *cmd = reinterpret_cast<const CQPrefetchCmd *>(m_cmd_ptr);

    uint32_t length = cmd->relay_inline.length;
    const uint8_t *data_ptr = m_cmd_ptr + sizeof(CQPrefetchCmd);

    check_cmd_reg_limit(sizeof(CQPrefetchCmd) + length);

    size_t offset = m_dispatch_data.size();
    m_dispatch_data.resize(offset + length);
    uint8_t *dst = m_dispatch_data.data() + offset;
    memcpy(dst, data_ptr, length);

    flush_dispatch_data();

    return cmd->relay_inline.stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page
// to the dispatcher. This is used to assemble dispatcher commands when data comes out of band,
// e.g., reading from DRAM. That means this command is stateful, incorrect use will be...bad.

uint32_t Prefetch::process_relay_inline_noflush_cmd() {
    const CQPrefetchCmd *cmd = reinterpret_cast<const CQPrefetchCmd *>(m_cmd_ptr);

#if 0 // TODO: Revise this
    // ACHTUNG: No payload assumed: is this always correct?
    uint32_t length = sizeof(CQDispatchCmd);
#endif
    uint32_t length = cmd->relay_inline.length;
    const uint8_t *data_ptr = m_cmd_ptr + sizeof(CQPrefetchCmd);

    check_cmd_reg_limit(sizeof(CQPrefetchCmd) + length);

    size_t offset = m_dispatch_data.size();
    m_dispatch_data.resize(offset + length);
    uint8_t *dst = m_dispatch_data.data() + offset;
    memcpy(dst, data_ptr, length);

#if 0 // TODO: Revise this
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
#endif
    return cmd->relay_inline.stride;
}

uint32_t Prefetch::process_stall() {
    // nothing to do so far
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t Prefetch::process_terminate() {
    // nothing to do so far
    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

void Prefetch::flush_dispatch_data() {
    m_dispatch->run(m_dispatch_data.data(), int(m_dispatch_data.size()));
    m_dispatch_data.clear();
}

void Prefetch::check_cmd_reg_limit(uint32_t length) {
    if (m_cmd_ptr + length > m_cmd_reg_end) {
        throw std::runtime_error(
            "Dispatch command region overflow: got " + 
                std::to_string(m_cmd_reg_end - m_cmd_ptr) +
                " want " + std::to_string(length));
    }
}

void Prefetch::copy_to_local_cache(const uint8_t *src, uint32_t count) {
    assert(count < l1_cache_elements);
    memcpy(m_l1_cache.data(), src, count * sizeof(uint32_t));
}

} // namespace dispatch
} // namespace device
} // namespace metal
} // namespace tt

