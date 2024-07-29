// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "common/env_lib.hpp"
#include "dev_mem_map.h"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"

//
//    NOTE: [RONIN] DeviceCommand defines buffer containing sequence of CQ commands
//        and corresponding data chunks. Template parameter "hugepage_write" is
//        true for device commands (buffer in hugepage device memory) and
//        false for host commands (buffer in host memory). The main difference is that
//        device commands are assembled in host memory and then copied to device while
//        host commands are assembled in host memory directly.
//

template <bool hugepage_write = false>
class DeviceCommand {
   public:
    DeviceCommand() = default;
    DeviceCommand(void *cmd_region, uint32_t cmd_sequence_sizeB) :
        cmd_sequence_sizeB(cmd_sequence_sizeB), cmd_region(cmd_region), cmd_write_offsetB(0) {
        TT_FATAL(
            cmd_sequence_sizeB % sizeof(uint32_t) == 0,
            "Command sequence size B={} is not {}-byte aligned",
            cmd_sequence_sizeB,
            sizeof(uint32_t));
    }
    template <bool hp_w = hugepage_write, typename std::enable_if_t<!hp_w, int> = 0>
    DeviceCommand(uint32_t cmd_sequence_sizeB) : cmd_sequence_sizeB(cmd_sequence_sizeB), cmd_write_offsetB(0) {
        TT_FATAL(
            cmd_sequence_sizeB % sizeof(uint32_t) == 0,
            "Command sequence size B={} is not {}-byte aligned",
            cmd_sequence_sizeB,
            sizeof(uint32_t));
        this->cmd_region_vector.resize(cmd_sequence_sizeB / sizeof(uint32_t), 0);
        this->cmd_region = this->cmd_region_vector.data();
    }

    DeviceCommand &operator=(const DeviceCommand &other) {
        this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
        this->cmd_write_offsetB = other.cmd_write_offsetB;
        this->cmd_region_vector = other.cmd_region_vector;
        this->deepcopy(other);
        return *this;
    }
    DeviceCommand &operator=(DeviceCommand &&other) {
        this->cmd_sequence_sizeB = other.cmd_sequence_sizeB;
        this->cmd_write_offsetB = other.cmd_write_offsetB;
        this->cmd_region_vector = other.cmd_region_vector;
        this->deepcopy(other);
        return *this;
    }
    DeviceCommand(const DeviceCommand &other) :
        cmd_sequence_sizeB(other.cmd_sequence_sizeB),
        cmd_write_offsetB(other.cmd_write_offsetB),
        cmd_region_vector(other.cmd_region_vector) {
        this->deepcopy(other);
    }
    DeviceCommand(DeviceCommand &&other) :
        cmd_sequence_sizeB(other.cmd_sequence_sizeB),
        cmd_write_offsetB(other.cmd_write_offsetB),
        cmd_region_vector(other.cmd_region_vector) {
        this->deepcopy(other);
    }

    // Constants
    static constexpr uint32_t PROGRAM_PAGE_SIZE = 2048;  // TODO: Move this somewhere else

    uint32_t size_bytes() const { return this->cmd_sequence_sizeB; }

    void *data() const { return this->cmd_region; }

    vector_memcpy_aligned<uint32_t> cmd_vector() const { return this->cmd_region_vector; }

    void add_dispatch_wait(
        uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count = 0, bool notify_prefetch = false, bool do_wait = true) {
        auto initialize_wait_cmds = [&](CQPrefetchCmd *relay_wait, CQDispatchCmd *wait_cmd) {
            relay_wait->base.cmd_id = CQ_PREFETCH_CMD_RELAY_INLINE;
            relay_wait->relay_inline.length = sizeof(CQDispatchCmd);
            relay_wait->relay_inline.stride = CQ_PREFETCH_CMD_BARE_MIN_SIZE;

            wait_cmd->base.cmd_id = CQ_DISPATCH_CMD_WAIT;
            wait_cmd->wait.barrier = barrier;
            wait_cmd->wait.notify_prefetch = notify_prefetch;
            wait_cmd->wait.wait = do_wait;
            wait_cmd->wait.addr = address;
            wait_cmd->wait.count = count;
            wait_cmd->wait.clear_count = clear_count;
        };
        CQPrefetchCmd *relay_wait_dst = this->reserve_space<CQPrefetchCmd *>(sizeof(CQPrefetchCmd));
        CQDispatchCmd *wait_cmd_dst = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_wait;
            alignas(MEMCPY_ALIGNMENT) CQDispatchCmd wait_cmd;
            initialize_wait_cmds(&relay_wait, &wait_cmd);
            this->memcpy(relay_wait_dst, &relay_wait, sizeof(CQPrefetchCmd));
            this->memcpy(wait_cmd_dst, &wait_cmd, sizeof(CQDispatchCmd));
        } else {
            initialize_wait_cmds(relay_wait_dst, wait_cmd_dst);
        }
    }

    void add_dispatch_wait_with_prefetch_stall(
        uint8_t barrier, uint32_t address, uint32_t count, uint8_t clear_count = 0, bool do_wait = true) {
        this->add_dispatch_wait(barrier, address, count, clear_count, true, do_wait);
        uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
        auto initialize_stall_cmd = [&](CQPrefetchCmd *stall_cmd) {
            *stall_cmd = {};
            stall_cmd->base.cmd_id = CQ_PREFETCH_CMD_STALL;
        };
        CQPrefetchCmd *stall_cmd_dst = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd stall_cmd;
            initialize_stall_cmd(&stall_cmd);
            this->memcpy(stall_cmd_dst, &stall_cmd, sizeof(CQPrefetchCmd));
        } else {
            initialize_stall_cmd(stall_cmd_dst);
        }
    }

    void add_prefetch_relay_linear(uint32_t noc_xy_addr, uint32_t lengthB, uint32_t addr) {
        uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
        auto initialize_relay_linear_cmd = [&](CQPrefetchCmd *relay_linear_cmd) {
            relay_linear_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_LINEAR;
            relay_linear_cmd->relay_linear.noc_xy_addr = noc_xy_addr;
            relay_linear_cmd->relay_linear.length = lengthB;
            relay_linear_cmd->relay_linear.addr = addr;
        };
        CQPrefetchCmd *relay_linear_cmd_dst = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_linear_cmd;
            initialize_relay_linear_cmd(&relay_linear_cmd);
            this->memcpy(relay_linear_cmd_dst, &relay_linear_cmd, sizeof(CQPrefetchCmd));
        } else {
            initialize_relay_linear_cmd(relay_linear_cmd_dst);
        }
    }

    void add_prefetch_relay_paged(
        uint8_t is_dram,
        uint8_t start_page,
        uint32_t base_addr,
        uint32_t page_size,
        uint32_t pages,
        uint16_t length_adjust = 0) {
        uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
        auto initialize_relay_paged_cmd = [&](CQPrefetchCmd *relay_paged_cmd) {
            relay_paged_cmd->base.cmd_id = CQ_PREFETCH_CMD_RELAY_PAGED;
            relay_paged_cmd->relay_paged.packed_page_flags = (is_dram << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT) |
                                                             (start_page << CQ_PREFETCH_RELAY_PAGED_START_PAGE_SHIFT);
            relay_paged_cmd->relay_paged.length_adjust = length_adjust;
            relay_paged_cmd->relay_paged.base_addr = base_addr;
            relay_paged_cmd->relay_paged.page_size = page_size;
            relay_paged_cmd->relay_paged.pages = pages;
        };
        CQPrefetchCmd *relay_paged_cmd_dst = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_paged_cmd;
            initialize_relay_paged_cmd(&relay_paged_cmd);
            this->memcpy(relay_paged_cmd_dst, &relay_paged_cmd, sizeof(CQPrefetchCmd));
        } else {
            initialize_relay_paged_cmd(relay_paged_cmd_dst);
        }
    }

    template <bool inline_data = false>
    void add_dispatch_write_linear(
        bool flush_prefetch,
        uint8_t num_mcast_dests,
        uint32_t noc_xy_addr,
        uint32_t addr,
        uint32_t data_sizeB,
        const void *data = nullptr) {
        uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        auto initialize_write_cmd = [&](CQDispatchCmd *write_cmd) {
            write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
            write_cmd->write_linear.num_mcast_dests = num_mcast_dests;
            write_cmd->write_linear.noc_xy_addr = noc_xy_addr;
            write_cmd->write_linear.addr = addr;
            write_cmd->write_linear.length = data_sizeB;
        };
        CQDispatchCmd *write_cmd_dst = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_cmd;
            initialize_write_cmd(&write_cmd);
            this->memcpy(write_cmd_dst, &write_cmd, sizeof(CQDispatchCmd));
        } else {
            initialize_write_cmd(write_cmd_dst);
        }

        if (inline_data) {
            TT_ASSERT(data != nullptr);  // compiled out?
            uint32_t increment_sizeB = align(data_sizeB, PCIE_ALIGNMENT);
            this->add_data(data, data_sizeB, increment_sizeB);
        }
    }

    template <bool inline_data = false>
    void add_dispatch_write_paged(
        bool flush_prefetch,
        uint8_t is_dram,
        uint16_t start_page,
        uint32_t base_addr,
        uint32_t page_size,
        uint32_t pages,
        const void *data = nullptr) {
        uint32_t data_sizeB = page_size * pages;
        uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        auto initialize_write_cmd = [&](CQDispatchCmd *write_cmd) {
            write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
            write_cmd->write_paged.is_dram = is_dram;
            write_cmd->write_paged.start_page = start_page;
            write_cmd->write_paged.base_addr = base_addr;
            write_cmd->write_paged.page_size = page_size;
            write_cmd->write_paged.pages = pages;
        };
        CQDispatchCmd *write_cmd_dst = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_cmd;
            initialize_write_cmd(&write_cmd);
            this->memcpy(write_cmd_dst, &write_cmd, sizeof(CQDispatchCmd));
        } else {
            initialize_write_cmd(write_cmd_dst);
        }

        if (inline_data) {
            TT_ASSERT(data != nullptr);  // compiled out?
            uint32_t increment_sizeB = align(data_sizeB, PCIE_ALIGNMENT);
            this->add_data(data, data_sizeB, increment_sizeB);
        }
    }

    template <bool inline_data = false>
    void add_dispatch_write_host(bool flush_prefetch, uint32_t data_sizeB, const void *data = nullptr) {
        uint32_t payload_sizeB = sizeof(CQDispatchCmd) + data_sizeB;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        auto initialize_write_cmd = [&](CQDispatchCmd *write_cmd) {
            write_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
            write_cmd->write_linear_host.length =
                payload_sizeB;  // CQ_DISPATCH_CMD_WRITE_LINEAR_HOST writes dispatch cmd back to completion queue
        };
        CQDispatchCmd *write_cmd_dst = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_cmd;
            initialize_write_cmd(&write_cmd);
            this->memcpy(write_cmd_dst, &write_cmd, sizeof(CQDispatchCmd));
        } else {
            initialize_write_cmd(write_cmd_dst);
        }

        if (inline_data) {
            TT_ASSERT(data != nullptr);  // compiled out?
            uint32_t increment_sizeB = align(data_sizeB, PCIE_ALIGNMENT);
            this->add_data(data, data_sizeB, increment_sizeB);
        }
    }

    void add_prefetch_exec_buf(uint32_t base_addr, uint32_t log_page_size, uint32_t pages) {
        uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
        auto initialize_exec_buf_cmd = [&](CQPrefetchCmd *exec_buf_cmd) {
            exec_buf_cmd->base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF;
            exec_buf_cmd->exec_buf.base_addr = base_addr;
            exec_buf_cmd->exec_buf.log_page_size = log_page_size;
            exec_buf_cmd->exec_buf.pages = pages;
        };
        CQPrefetchCmd *exec_buf_cmd_dst = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd exec_buf_cmd;
            initialize_exec_buf_cmd(&exec_buf_cmd);
            this->memcpy(exec_buf_cmd_dst, &exec_buf_cmd, sizeof(CQPrefetchCmd));
        } else {
            initialize_exec_buf_cmd(exec_buf_cmd_dst);
        }
    }

    void add_dispatch_terminate() {
        this->add_prefetch_relay_inline(true, sizeof(CQDispatchCmd));
        auto initialize_terminate_cmd = [&](CQDispatchCmd *terminate_cmd) {
            *terminate_cmd = {};
            terminate_cmd->base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
        };
        CQDispatchCmd *terminate_cmd_dst = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQDispatchCmd terminate_cmd;
            initialize_terminate_cmd(&terminate_cmd);
            this->memcpy(terminate_cmd_dst, &terminate_cmd, sizeof(CQDispatchCmd));
        } else {
            initialize_terminate_cmd(terminate_cmd_dst);
        }
    }

    void add_prefetch_terminate() {
        uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
        auto initialize_terminate_cmd = [&](CQPrefetchCmd *terminate_cmd) {
            *terminate_cmd = {};
            terminate_cmd->base.cmd_id = CQ_PREFETCH_CMD_TERMINATE;
        };
        CQPrefetchCmd *terminate_cmd_dst = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd terminate_cmd;
            initialize_terminate_cmd(&terminate_cmd);
            this->memcpy(terminate_cmd_dst, &terminate_cmd, sizeof(CQPrefetchCmd));
        } else {
            initialize_terminate_cmd(terminate_cmd_dst);
        }
    }

    void add_prefetch_exec_buf_end() {
        uint32_t increment_sizeB = align(sizeof(CQPrefetchCmd), PCIE_ALIGNMENT);
        auto initialize_exec_buf_end_cmd = [&](CQPrefetchCmd *exec_buf_end_cmd) {
            exec_buf_end_cmd->base.cmd_id = CQ_PREFETCH_CMD_EXEC_BUF_END;
        };
        CQPrefetchCmd *exec_buf_end_cmd_dst = this->reserve_space<CQPrefetchCmd *>(increment_sizeB);

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd exec_buf_end_cmd;
            initialize_exec_buf_end_cmd(&exec_buf_end_cmd);
            this->memcpy(exec_buf_end_cmd_dst, &exec_buf_end_cmd, sizeof(CQPrefetchCmd));
        } else {
            initialize_exec_buf_end_cmd(exec_buf_end_cmd_dst);
        }
    }

    void update_cmd_sequence(uint32_t cmd_offsetB, const void *new_data, uint32_t data_sizeB) {
        this->memcpy((char *)this->cmd_region + cmd_offsetB, new_data, data_sizeB);
    }

    void add_data(const void *data, uint32_t data_size_to_copyB, uint32_t cmd_write_offset_incrementB) {
        this->validate_cmd_write(cmd_write_offset_incrementB);
        this->memcpy((uint8_t *)this->cmd_region + this->cmd_write_offsetB, data, data_size_to_copyB);
        this->cmd_write_offsetB += cmd_write_offset_incrementB;
    }

    template <typename PackedSubCmd>
    void add_dispatch_write_packed(
        uint16_t num_sub_cmds,
        uint32_t common_addr,
        uint16_t packed_data_sizeB,
        uint32_t payload_sizeB,
        const std::vector<PackedSubCmd> &sub_cmds,
        const std::vector<std::pair<const void *, uint32_t>> &data_collection,
        const uint32_t offset_idx = 0,
        const bool no_stride = false) {
        static_assert(
            std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value or
            std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value);
        bool multicast = std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value;

        static constexpr uint32_t max_num_packed_sub_cmds =
            (dispatch_constants::TRANSFER_PAGE_SIZE - sizeof(CQDispatchCmd)) / sizeof(PackedSubCmd);
        TT_ASSERT(
            num_sub_cmds <= max_num_packed_sub_cmds,
            "Max number of packed sub commands are {} but requesting {}",
            max_num_packed_sub_cmds,
            num_sub_cmds);

        bool flush_prefetch = true;
        this->add_prefetch_relay_inline(flush_prefetch, payload_sizeB);

        auto initialize_write_packed_cmd = [&](CQDispatchCmd *write_packed_cmd) {
            write_packed_cmd->base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
            write_packed_cmd->write_packed.flags =
                (multicast ? CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_MCAST : CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE) |
                (no_stride ? CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE : CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE);
            write_packed_cmd->write_packed.count = num_sub_cmds;
            write_packed_cmd->write_packed.addr = common_addr;
            write_packed_cmd->write_packed.size = packed_data_sizeB;
        };
        CQDispatchCmd *write_packed_cmd_dst = this->reserve_space<CQDispatchCmd *>(sizeof(CQDispatchCmd));

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQDispatchCmd write_packed_cmd;
            initialize_write_packed_cmd(&write_packed_cmd);
            this->memcpy(write_packed_cmd_dst, &write_packed_cmd, sizeof(CQDispatchCmd));
        } else {
            initialize_write_packed_cmd(write_packed_cmd_dst);
        }

        static_assert(sizeof(PackedSubCmd) % sizeof(uint32_t) == 0);
        uint32_t sub_cmds_sizeB = num_sub_cmds * sizeof(PackedSubCmd);
        this->memcpy((char *)this->cmd_region + this->cmd_write_offsetB, &sub_cmds[offset_idx], sub_cmds_sizeB);

        uint32_t increment_sizeB =
            align(sub_cmds_sizeB, L1_ALIGNMENT);  // this assumes CQDispatchCmd is L1_ALIGNEMENT aligned
        this->cmd_write_offsetB += increment_sizeB;

        // copy the actual data
        increment_sizeB = align(packed_data_sizeB, L1_ALIGNMENT);
        uint32_t num_data_copies = no_stride ? 1 : num_sub_cmds;
        for (uint32_t i = offset_idx; i < offset_idx + num_data_copies; ++i) {
            this->memcpy(
                (char *)this->cmd_region + this->cmd_write_offsetB,
                data_collection[i].first,
                data_collection[i].second);
            this->cmd_write_offsetB += increment_sizeB;
        }

        this->cmd_write_offsetB = align(this->cmd_write_offsetB, PCIE_ALIGNMENT);
    }

    template <typename CommandPtr, bool data = false>
    CommandPtr reserve_space(uint32_t size_to_writeB) {
        this->validate_cmd_write(size_to_writeB);
        CommandPtr cmd = (CommandPtr)((char *)this->cmd_region + this->cmd_write_offsetB);
        // Only zero out cmds
        if constexpr (!data) {
            if (zero_init_enable)
                DeviceCommand::zero(cmd);
        }
        this->cmd_write_offsetB += size_to_writeB;
        return cmd;
    }

   private:
    static bool zero_init_enable;

    void add_prefetch_relay_inline(bool flush, uint32_t lengthB) {
        auto initialize_relay_write = [&](CQPrefetchCmd *relay_write) {
            relay_write->base.cmd_id = flush ? CQ_PREFETCH_CMD_RELAY_INLINE : CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH;
            relay_write->relay_inline.length = lengthB;
            relay_write->relay_inline.stride = align(sizeof(CQPrefetchCmd) + lengthB, PCIE_ALIGNMENT);
        };
        CQPrefetchCmd *relay_write_dst = this->reserve_space<CQPrefetchCmd *>(sizeof(CQPrefetchCmd));

        if constexpr (hugepage_write) {
            alignas(MEMCPY_ALIGNMENT) CQPrefetchCmd relay_write;
            initialize_relay_write(&relay_write);
            this->memcpy(relay_write_dst, &relay_write, sizeof(CQPrefetchCmd));
        } else {
            initialize_relay_write(relay_write_dst);
        }
    }

    void validate_cmd_write(uint32_t data_sizeB) const {
        uint32_t data_endB = this->cmd_write_offsetB + data_sizeB;
        TT_ASSERT(
            data_endB <= this->cmd_sequence_sizeB,
            "Out of bounds command sequence write: attemping to write {} B but only {} B available",
            data_sizeB,
            this->cmd_sequence_sizeB - this->cmd_write_offsetB);
    }

    template <typename Command>
    void zero(Command *cmd) {
        if constexpr (hugepage_write) {
#if 0 // [RONIN]
            std::vector<char, boost::alignment::aligned_allocator<char, MEMCPY_ALIGNMENT>> 
                zero_cmd(sizeof(Command), 0);
#else
            std::vector<char> zero_cmd(sizeof(Command), 0);
#endif
            this->memcpy(cmd, zero_cmd.data(), sizeof(Command));
        } else {
            std::fill((uint8_t *)cmd, (uint8_t *)cmd + sizeof(Command), 0);
        }
    }

    void deepcopy(const DeviceCommand &other) {
        if (other.cmd_region_vector.empty() and other.cmd_region != nullptr) {
            this->cmd_region = other.cmd_region;
        } else if (not other.cmd_region_vector.empty()) {
            TT_ASSERT(other.cmd_region != nullptr);
            this->cmd_region = this->cmd_region_vector.data();
            this->memcpy(this->cmd_region, other.cmd_region_vector.data(), this->cmd_sequence_sizeB);
        }
    }

#if 0 // TODO: Revise this
    void memcpy(void *__restrict dst, const void *__restrict src, size_t n) {
        if constexpr (hugepage_write) {
            memcpy_to_device(dst, src, n);
        } else {
            std::memcpy(dst, src, n);
        }
    }
#endif

    void memcpy(void *__restrict dst, const void *__restrict src, size_t n) {
        std::memcpy(dst, src, n);
    }

    uint32_t cmd_sequence_sizeB = 0;
    void *cmd_region = nullptr;
    uint32_t cmd_write_offsetB = 0;

    vector_memcpy_aligned<uint32_t> cmd_region_vector;
};

template <bool hugepage_write>
bool DeviceCommand<hugepage_write>::zero_init_enable = tt::parse_env<bool>("TT_METAL_ZERO_INIT_ENABLE", false);

using HugepageDeviceCommand = DeviceCommand<true>;
using HostMemDeviceCommand = DeviceCommand<false>;
