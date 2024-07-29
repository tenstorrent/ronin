
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
            m_cmd_ptr(nullptr) { }

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
            stride = process_relay_paged_cmd(true, start_page);
        }
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

    // ACHTUNG: No payload assumed: is this always correct?
    uint32_t length = sizeof(CQDispatchCmd);
    const uint8_t *data_ptr = m_cmd_ptr + sizeof(CQPrefetchCmd);

    check_cmd_reg_limit(sizeof(CQPrefetchCmd) + length);

    size_t offset = m_dispatch_data.size();
    m_dispatch_data.resize(offset + length);
    uint8_t *dst = m_dispatch_data.data() + offset;
    memcpy(dst, data_ptr, length);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
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

} // namespace dispatch
} // namespace device
} // namespace metal
} // namespace tt

