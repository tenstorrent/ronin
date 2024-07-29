// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/command_queue.hpp"

#include <malloc.h>

#include <algorithm>  // for copy() and assign()
#include <iterator>   // for back_inserter
#include <memory>
#include <string>
#include <variant>

#include "allocator/allocator.hpp"
#include "debug_tools.hpp"
#include "dev_msgs.h"
#include "noc/noc_parameters.h"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/program.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
//#include "tt_metal/impl/debug/dprint_server.hpp" // [RONIN]
//#include "tt_metal/impl/debug/watcher_server.hpp" // [RONIN]
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::unique_ptr;

#if 0 // TODO: Revise this
std::mutex finish_mutex;
std::condition_variable finish_cv;
#endif

#ifdef _MSC_VER // [RONIN]
namespace {

inline uint32_t __builtin_ctz(uint32_t x) {
    unsigned long y = 0;
    if (_BitScanForward(&y, (unsigned long)x)) {
        return uint32_t(y);
    } else {
        return 32;
    }
}

} // namespace
#endif

namespace tt::tt_metal {

#if 0 // TODO: Revise this
// TODO: Delete entries when programs are deleted to save memory
thread_local std::unordered_map<uint64_t, EnqueueProgramCommand::CachedProgramCommandSequence>
    EnqueueProgramCommand::cached_program_command_sequences = {};
#endif

std::unordered_map<uint64_t, EnqueueProgramCommand::CachedProgramCommandSequence>
    EnqueueProgramCommand::cached_program_command_sequences = {};

//
//    EnqueueReadBufferCommand
//

EnqueueReadBufferCommand::EnqueueReadBufferCommand(
        uint32_t command_queue_id,
        Device *device,
        NOC noc_index,
        Buffer &buffer,
        void *dst,
#if 0 // TODO: Revise this
        SystemMemoryManager &manager,
#endif
        CQManager *cq_manager,
        uint32_t expected_num_workers_completed,
        uint32_t src_page_index,
        std::optional<uint32_t> pages_to_read):
            command_queue_id(command_queue_id),
            noc_index(noc_index),
            dst(dst),
#if 0 // TODO: Revise this
            manager(manager),
#endif
            cq_manager(cq_manager),
            buffer(buffer),
            expected_num_workers_completed(expected_num_workers_completed),
            src_page_index(src_page_index),
            pages_to_read(pages_to_read.has_value() ? 
                pages_to_read.value() : 
                buffer.num_pages()) {
    TT_ASSERT(
        buffer.buffer_type() == BufferType::DRAM || buffer.buffer_type() == BufferType::L1,
        "Trying to read an invalid buffer");

    this->device = device;
    this->dispatch_core_type = 
        dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void EnqueueReadBufferCommand::process() {
    // accounts for padding
    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_STALL
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_RELAY_LINEAR or CQ_PREFETCH_CMD_RELAY_PAGED

    // [RONIN] Make room for trailing add_dispatch_wait (see comment below)
    cmd_sequence_sizeB +=
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

#if 0 // TODO: Revise this
    void *cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
#endif
    void *cmd_region = this->cq_manager->reserve(cmd_sequence_sizeB);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait_with_prefetch_stall(
        true, 
        DISPATCH_MESSAGE_ADDR, 
        this->expected_num_workers_completed);

    uint32_t padded_page_size = align(this->buffer.page_size(), ADDRESS_ALIGNMENT);
    bool flush_prefetch = false;
    command_sequence.add_dispatch_write_host(flush_prefetch, this->pages_to_read * padded_page_size);

    this->add_prefetch_relay(command_sequence);

    // [RONIN] Added to perform CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST synchronously
    //     (alternative would be enabling recording event after this process command)
    command_sequence.add_dispatch_wait(
        false, 
        DISPATCH_MESSAGE_ADDR, 
        this->expected_num_workers_completed);

#if 0 // TODO: Revise this
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
#endif
    this->cq_manager->push(cmd_sequence_sizeB);
}

//
//    EnqueueReadInterleavedBufferCommand
//

void EnqueueReadInterleavedBufferCommand::add_prefetch_relay(HugepageDeviceCommand &command) {
    uint32_t padded_page_size = align(this->buffer.page_size(), ADDRESS_ALIGNMENT);
    command.add_prefetch_relay_paged(
        (this->buffer.buffer_type() == BufferType::DRAM),
        this->src_page_index,
        this->buffer.address(),
        padded_page_size,
        this->pages_to_read);
}

//
//   EnqueueReadShardedBufferCommand 
//

void EnqueueReadShardedBufferCommand::add_prefetch_relay(HugepageDeviceCommand &command) {
    uint32_t padded_page_size = align(this->buffer.page_size(), ADDRESS_ALIGNMENT);
    const CoreCoord physical_core =
        this->buffer.device()->physical_core_from_logical_core(this->core, this->buffer.core_type());
    command.add_prefetch_relay_linear(
        this->device->get_noc_unicast_encoding(this->noc_index, physical_core),
        padded_page_size * this->pages_to_read,
        this->bank_base_address);
}

//
//    EnqueueWriteBufferCommand
//

EnqueueWriteBufferCommand::EnqueueWriteBufferCommand(
        uint32_t command_queue_id,
        Device *device,
        NOC noc_index,
        const Buffer &buffer,
        const void *src,
#if 0 // TODO: Revise this
        SystemMemoryManager &manager,
#endif
        CQManager *cq_manager,
        bool issue_wait,
        uint32_t expected_num_workers_completed,
        uint32_t bank_base_address,
        uint32_t padded_page_size,
        uint32_t dst_page_index,
        std::optional<uint32_t> pages_to_write):
            command_queue_id(command_queue_id),
            noc_index(noc_index),
#if 0 // TODO: Revise this
            manager(manager),
#endif
            cq_manager(cq_manager),
            issue_wait(issue_wait),
            src(src),
            buffer(buffer),
            expected_num_workers_completed(expected_num_workers_completed),
            bank_base_address(bank_base_address),
            padded_page_size(padded_page_size),
            dst_page_index(dst_page_index),
            pages_to_write(pages_to_write.has_value() ? 
                pages_to_write.value() : 
                buffer.num_pages()) {
    TT_ASSERT(buffer.is_dram() || buffer.is_l1(), "Trying to write to an invalid buffer");
    this->device = device;
    this->dispatch_core_type = 
        dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void EnqueueWriteBufferCommand::process() {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;

    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + (CQ_DISPATCH_CMD_WRITE_PAGED or
                                         // CQ_DISPATCH_CMD_WRITE_LINEAR)
        data_size_bytes;
    if (this->issue_wait) {
        cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
    }

#if 0 // TODO: Revise this
    void *cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
#endif
    void *cmd_region = this->cq_manager->reserve(cmd_sequence_sizeB);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    if (this->issue_wait) {
        command_sequence.add_dispatch_wait(
            false, 
            DISPATCH_MESSAGE_ADDR, 
            this->expected_num_workers_completed);
    }

    this->add_dispatch_write(command_sequence);

    uint32_t full_page_size =
        align(this->buffer.page_size(), ADDRESS_ALIGNMENT);  // this->padded_page_size could be a partial page if buffer
                                                             // page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = this->padded_page_size < full_page_size;

    this->add_buffer_data(command_sequence);

#if 0 // TODO: Revise this
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
#endif
    this->cq_manager->push(cmd_sequence_sizeB);
}

//
//   EnqueueWriteInterleavedBufferCommand 
//

void EnqueueWriteInterleavedBufferCommand::add_dispatch_write(HugepageDeviceCommand &command_sequence) {
    uint8_t is_dram = uint8_t(this->buffer.buffer_type() == BufferType::DRAM);
    TT_ASSERT(
        this->dst_page_index <= 0xFFFF,
        "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    uint16_t start_page = uint16_t(this->dst_page_index & 0xFFFF);
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_paged(
        flush_prefetch, 
        is_dram, 
        start_page, 
        this->bank_base_address, 
        this->padded_page_size, 
        this->pages_to_write);
}

//
//    EnqueueWriteInterleavedBufferCommand
//

void EnqueueWriteInterleavedBufferCommand::add_buffer_data(HugepageDeviceCommand &command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;

    uint32_t full_page_size =
        align(this->buffer.page_size(), ADDRESS_ALIGNMENT);  // this->padded_page_size could be a partial page if buffer
                                                             // page size > MAX_PREFETCH_CMD_SIZE
    bool write_partial_pages = (this->padded_page_size < full_page_size);

    uint32_t buffer_addr_offset = this->bank_base_address - this->buffer.address();
    uint32_t num_banks = this->device->num_banks(this->buffer.buffer_type());

    // TODO: Consolidate
    if (write_partial_pages) {
        uint32_t padding = full_page_size - this->buffer.page_size();
        uint32_t unpadded_src_offset = buffer_addr_offset;
        uint32_t src_address_offset = unpadded_src_offset;
        for (uint32_t sysmem_address_offset = 0; 
                sysmem_address_offset < data_size_bytes;
                sysmem_address_offset += this->padded_page_size) {
            uint32_t page_size_to_copy = this->padded_page_size;
            if (src_address_offset + this->padded_page_size > buffer.page_size()) {
                // last partial page being copied from unpadded src buffer
                page_size_to_copy -= padding;
            }
            command_sequence.add_data(
                (char *)this->src + src_address_offset, 
                page_size_to_copy, 
                this->padded_page_size);
            src_address_offset += page_size_to_copy;
        }
    } else {
        uint32_t unpadded_src_offset =
            (((buffer_addr_offset / this->padded_page_size) * num_banks) + this->dst_page_index) *
            this->buffer.page_size();
        if (this->buffer.page_size() % ADDRESS_ALIGNMENT != 0 && 
                this->buffer.page_size() != this->buffer.size()) {
            // If page size is not 32B-aligned, we cannot do a contiguous write
            uint32_t src_address_offset = unpadded_src_offset;
            for (uint32_t sysmem_address_offset = 0; 
                    sysmem_address_offset < data_size_bytes;
                    sysmem_address_offset += this->padded_page_size) {
                command_sequence.add_data(
                    (char *)this->src + src_address_offset, 
                    this->buffer.page_size(), 
                    this->padded_page_size);
                src_address_offset += this->buffer.page_size();
            }
        } else {
            command_sequence.add_data(
                (char *)this->src + unpadded_src_offset, 
                data_size_bytes, 
                data_size_bytes);
        }
    }
}

//
//    EnqueueWriteShardedBufferCommand
//

void EnqueueWriteShardedBufferCommand::add_dispatch_write(HugepageDeviceCommand &command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    const CoreCoord physical_core =
        this->buffer.device()->physical_core_from_logical_core(this->core, this->buffer.core_type());
    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_linear(
        flush_prefetch,
        0,
        this->device->get_noc_unicast_encoding(this->noc_index, physical_core),
        this->bank_base_address,
        data_size_bytes);
}

void EnqueueWriteShardedBufferCommand::add_buffer_data(HugepageDeviceCommand &command_sequence) {
    uint32_t data_size_bytes = this->pages_to_write * this->padded_page_size;
    if (this->buffer_page_mapping.has_value()) {
        const auto &page_mapping = this->buffer_page_mapping.value();
        uint8_t *dst = command_sequence.reserve_space<uint8_t *, true>(data_size_bytes);
        // TODO: Expose getter for cmd_write_offsetB?
        uint32_t dst_offset = dst - (uint8_t *)command_sequence.data();
        for (uint32_t dev_page = this->dst_page_index; 
                dev_page < this->dst_page_index + this->pages_to_write;
                ++dev_page) {
            auto &host_page = page_mapping.dev_page_to_host_page_mapping_[dev_page];
            if (host_page.has_value()) {
                command_sequence.update_cmd_sequence(
                    dst_offset,
                    (char *)this->src + host_page.value() * this->buffer.page_size(),
                    this->buffer.page_size());
            }
            dst_offset += this->padded_page_size;
        }
    } else {
        if (this->buffer.page_size() != this->padded_page_size && 
                this->buffer.page_size() != this->buffer.size()) {
            uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
            for (uint32_t i = 0; i < this->pages_to_write; ++i) {
                command_sequence.add_data(
                    (char *)this->src + unpadded_src_offset, 
                    this->buffer.page_size(), 
                    this->padded_page_size);
                unpadded_src_offset += this->buffer.page_size();
            }
        } else {
            uint32_t unpadded_src_offset = this->dst_page_index * this->buffer.page_size();
            command_sequence.add_data(
                (char *)this->src + unpadded_src_offset, 
                data_size_bytes, 
                data_size_bytes);
        }
    }
}

//
//    EnqueueProgramCommand
//

EnqueueProgramCommand::EnqueueProgramCommand(
        uint32_t command_queue_id,
        Device *device,
        NOC noc_index,
        Program &program,
#if 0 // TODO: Revise this
        SystemMemoryManager &manager,
#endif
        CQManager *cq_manager,
        uint32_t expected_num_workers_completed):
            command_queue_id(command_queue_id),
            noc_index(noc_index),
#if 0 // TODO: Revise this
            manager(manager),
#endif
            cq_manager(cq_manager),
            expected_num_workers_completed(expected_num_workers_completed),
            program(program) {
    this->device = device;
    this->dispatch_core_type = 
        dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

void EnqueueProgramCommand::assemble_preamble_commands(bool prefetch_stall) {
    if (prefetch_stall) {
        // Wait command so previous program finishes
        // Wait command with barrier for binaries to commit to DRAM
        // Prefetch stall to prevent prefetcher picking up incomplete binaries from DRAM
        constexpr uint32_t uncached_cmd_sequence_sizeB =
            CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_STALL

        this->cached_program_command_sequences[program.id].preamble_command_sequence =
            HostMemDeviceCommand(uncached_cmd_sequence_sizeB);

        // Wait for Noc Write Barrier
        // wait for binaries to commit to dram, also wait for previous program to be done
        // Wait Noc Write Barrier, wait for binaries to be written to worker cores
        // Stall to allow binaries to commit to DRAM first
        // TODO: this can be removed for all but the first program run
        this->cached_program_command_sequences[program.id].preamble_command_sequence
            .add_dispatch_wait_with_prefetch_stall(
                true, 
                DISPATCH_MESSAGE_ADDR, 
                this->expected_num_workers_completed);
    } else {
        // Wait command so previous program finishes
        constexpr uint32_t cached_cmd_sequence_sizeB =
            CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        this->cached_program_command_sequences[program.id].preamble_command_sequence =
            HostMemDeviceCommand(cached_cmd_sequence_sizeB);
        this->cached_program_command_sequences[program.id].preamble_command_sequence
            .add_dispatch_wait(
                false, 
                DISPATCH_MESSAGE_ADDR, 
                this->expected_num_workers_completed);
    }
}

template <typename PackedSubCmd>
void generate_dispatch_write_packed(
        std::vector<HostMemDeviceCommand> &runtime_args_command_sequences,
        const uint32_t &l1_arg_base_addr,
        const std::vector<PackedSubCmd> &sub_cmds,
        const std::vector<std::pair<const void *, uint32_t>> &rt_data_and_sizes,
        const uint32_t &max_runtime_args_len,
        std::vector<std::reference_wrapper<RuntimeArgsData>> &rt_args_data,
        const uint32_t max_prefetch_command_size,
        const uint32_t id,
        bool no_stride = false) {
    static_assert(
        std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value ||
        std::is_same<PackedSubCmd, CQDispatchWritePackedMulticastSubCmd>::value);

#if 0 // TODO: Revise this
    thread_local static auto get_runtime_payload_sizeB =
#endif
    static auto get_runtime_payload_sizeB =
        [](uint32_t num_packed_cmds, uint32_t runtime_args_len, bool is_unicast, bool no_stride) {
            uint32_t sub_cmd_sizeB =
                is_unicast ? 
                    sizeof(CQDispatchWritePackedUnicastSubCmd) : 
                    sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB = sizeof(CQDispatchCmd) + align(num_packed_cmds * sub_cmd_sizeB, L1_ALIGNMENT);
            uint32_t aligned_runtime_data_sizeB =
                (no_stride ? 1 : num_packed_cmds) * 
                align(runtime_args_len * sizeof(uint32_t), L1_ALIGNMENT);
            return dispatch_cmd_sizeB + aligned_runtime_data_sizeB;
        };
#if 0 // TODO: Revise this
    thread_local static auto get_max_num_packed_cmds =
#endif
    static auto get_max_num_packed_cmds =
        [](uint32_t runtime_args_len, uint32_t max_size, bool is_unicast, bool no_stride) {
            uint32_t sub_cmd_sizeB =
                is_unicast ? 
                    sizeof(CQDispatchWritePackedUnicastSubCmd) : 
                    sizeof(CQDispatchWritePackedMulticastSubCmd);
            // Approximate calculation due to alignment
            max_size = max_size - sizeof(CQPrefetchCmd) - PCIE_ALIGNMENT - sizeof(CQDispatchCmd) - L1_ALIGNMENT;
            uint32_t max_num_packed_cmds =
                no_stride ? 
                    (max_size - align(runtime_args_len * sizeof(uint32_t), L1_ALIGNMENT)) / sub_cmd_sizeB : 
                    max_size / (align(runtime_args_len * sizeof(uint32_t), L1_ALIGNMENT) + sub_cmd_sizeB);
            return max_num_packed_cmds;
        };
#if 0 // TODO: Revise this
    thread_local static auto get_runtime_args_data_offset =
#endif
    static auto get_runtime_args_data_offset =
        [](uint32_t num_packed_cmds, uint32_t runtime_args_len, bool is_unicast) {
            uint32_t sub_cmd_sizeB =
                is_unicast ? 
                    sizeof(CQDispatchWritePackedUnicastSubCmd) : 
                    sizeof(CQDispatchWritePackedMulticastSubCmd);
            uint32_t dispatch_cmd_sizeB = sizeof(CQDispatchCmd) + align(num_packed_cmds * sub_cmd_sizeB, L1_ALIGNMENT);
            return sizeof(CQPrefetchCmd) + dispatch_cmd_sizeB;
        };

    constexpr bool unicast = std::is_same<PackedSubCmd, CQDispatchWritePackedUnicastSubCmd>::value;

    uint32_t num_packed_cmds_in_seq = sub_cmds.size();
    uint32_t max_packed_cmds =
        get_max_num_packed_cmds(max_runtime_args_len, max_prefetch_command_size, unicast, no_stride);
    uint32_t offset_idx = 0;
    if (no_stride) {
        TT_FATAL(max_packed_cmds >= num_packed_cmds_in_seq);
    }
    while (num_packed_cmds_in_seq != 0) {
        uint32_t num_packed_cmds = std::min(num_packed_cmds_in_seq, max_packed_cmds);
        uint32_t rt_payload_sizeB =
            get_runtime_payload_sizeB(num_packed_cmds, max_runtime_args_len, unicast, no_stride);
        uint32_t cmd_sequence_sizeB = align(sizeof(CQPrefetchCmd) + rt_payload_sizeB, PCIE_ALIGNMENT);
        runtime_args_command_sequences.emplace_back(cmd_sequence_sizeB);
        runtime_args_command_sequences.back().add_dispatch_write_packed<PackedSubCmd>(
            num_packed_cmds,
            l1_arg_base_addr,
            max_runtime_args_len * sizeof(uint32_t),
            rt_payload_sizeB,
            sub_cmds,
            rt_data_and_sizes,
            offset_idx,
            no_stride);
        uint32_t data_offset = (uint32_t)get_runtime_args_data_offset(num_packed_cmds, max_runtime_args_len, unicast);
        const uint32_t data_inc = align(max_runtime_args_len * sizeof(uint32_t), L1_ALIGNMENT);
        uint32_t num_data_copies = no_stride ? 1 : num_packed_cmds;
        for (uint32_t i = offset_idx; i < offset_idx + num_data_copies; ++i) {
            rt_args_data[i].get().rt_args_data =
                (uint32_t *)((char *)runtime_args_command_sequences.back().data() + data_offset);
            data_offset += data_inc;
        }
        num_packed_cmds_in_seq -= num_packed_cmds;
        offset_idx += num_packed_cmds;
    }
}

// Generate command sequence for unique (unicast) and common (multicast) runtime args
void EnqueueProgramCommand::assemble_runtime_args_commands() {
    // Maps to enum class RISCV, tt_backend_api_types.h
#if 0 // TODO: Revise this
    thread_local static const std::vector<uint32_t> unique_processor_to_l1_arg_base_addr = {
#endif
    static const std::vector<uint32_t> unique_processor_to_l1_arg_base_addr = {
        BRISC_L1_ARG_BASE,
        NCRISC_L1_ARG_BASE,
        0,
        0,
        0,
        eth_l1_mem::address_map::ERISC_L1_ARG_BASE,
        TRISC_L1_ARG_BASE,
    };
    CoreType dispatch_core_type =
        dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();

    uint32_t num_processors = unique_processor_to_l1_arg_base_addr.size();
    std::vector<std::vector<CQDispatchWritePackedUnicastSubCmd>> unique_sub_cmds(num_processors);
    std::vector<std::vector<std::pair<const void *, uint32_t>>> unique_rt_data_and_sizes(num_processors);
    std::vector<std::vector<std::reference_wrapper<RuntimeArgsData>>> unique_rt_args_data(num_processors);
    std::vector<uint32_t> unique_max_runtime_args_len(num_processors, 0);

    uint32_t num_kernels = program.num_kernels();
    std::vector<std::variant<
        std::vector<CQDispatchWritePackedMulticastSubCmd>,
        std::vector<CQDispatchWritePackedUnicastSubCmd>>>
        common_sub_cmds(num_kernels);
    std::vector<std::vector<std::pair<const void *, uint32_t>>> common_rt_data_and_sizes(num_kernels);
    std::vector<std::vector<std::reference_wrapper<RuntimeArgsData>>> common_rt_args_data(num_kernels);
    std::vector<uint32_t> common_max_runtime_args_len(num_kernels, 0);
    std::vector<uint32_t> common_processor_to_l1_arg_base_addr(num_kernels);

    std::set<uint32_t> unique_processors;
    std::set<uint32_t> common_kernels;

    // Unique Runtime Args (Unicast)
    // TODO: If we need to break apart cmds if they exceed the max prefetch cmd size
    // would potentially be more optimal to sort the data by size and have a max rt arg size
    // to pad to per split. Currently we take the max of all rt args before splitting
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);

        uint32_t processor_idx = static_cast<typename std::underlying_type<tt::RISCV>::type>(kernel->processor());

        if (!kernel->cores_with_runtime_args().empty()) {
            unique_processors.insert(processor_idx);
            unique_sub_cmds[processor_idx].reserve(kernel->cores_with_runtime_args().size());
            unique_rt_data_and_sizes[processor_idx].reserve(kernel->cores_with_runtime_args().size());
            unique_rt_args_data[processor_idx].reserve(kernel->cores_with_runtime_args().size());
            for (const auto &core_coord: kernel->cores_with_runtime_args()) {
                // can make a vector of unicast encodings here
                CoreCoord physical_core =
                    device->physical_core_from_logical_core(core_coord, kernel->get_kernel_core_type());
                const auto &runtime_args_data = kernel->runtime_args(core_coord);
                unique_rt_args_data[processor_idx].emplace_back(kernel->runtime_args_data(core_coord));
                // 2, 17, could be differnet len here

                unique_sub_cmds[processor_idx].emplace_back(CQDispatchWritePackedUnicastSubCmd{
                    .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, physical_core)});
                unique_rt_data_and_sizes[processor_idx].emplace_back(
                    runtime_args_data.data(), runtime_args_data.size() * sizeof(uint32_t));
                unique_max_runtime_args_len[processor_idx] =
                    std::max(unique_max_runtime_args_len[processor_idx], (uint32_t)runtime_args_data.size());
            }
        }
        // Common Runtime Args (Multicast)
        const auto &common_rt_args = kernel->common_runtime_args();

        if (common_rt_args.size() > 0) {
            common_kernels.insert(kernel_id);
            uint32_t common_args_addr =
                unique_processor_to_l1_arg_base_addr[processor_idx] + kernel->get_common_runtime_args_offset();
            common_processor_to_l1_arg_base_addr[kernel_id] = common_args_addr;
            common_rt_data_and_sizes[kernel_id].emplace_back(
                common_rt_args.data(), common_rt_args.size() * sizeof(uint32_t));
            common_rt_args_data[kernel_id].emplace_back(kernel->common_runtime_args_data());
            common_max_runtime_args_len[kernel_id] = (uint32_t)common_rt_args.size();
            if (kernel->get_kernel_core_type() == CoreType::ETH) {
                common_sub_cmds[kernel_id].emplace<std::vector<CQDispatchWritePackedUnicastSubCmd>>(
                    std::vector<CQDispatchWritePackedUnicastSubCmd>());
                auto &unicast_sub_cmd =
                    std::get<std::vector<CQDispatchWritePackedUnicastSubCmd>>(common_sub_cmds[kernel_id]);
                unicast_sub_cmd.reserve(kernel->logical_cores().size());
                for (auto &core_coord: kernel->logical_cores()) {
                    // can make a vector of unicast encodings here
                    CoreCoord physical_core = device->ethernet_core_from_logical_core(core_coord);
                    unicast_sub_cmd.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                        .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, physical_core)});
                }
            } else {
                vector<pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info =
                    extract_dst_noc_multicast_info<std::vector<CoreRange>>(
                        device, kernel->logical_coreranges(), kernel->get_kernel_core_type());
                common_sub_cmds[kernel_id].emplace<std::vector<CQDispatchWritePackedMulticastSubCmd>>(
                    std::vector<CQDispatchWritePackedMulticastSubCmd>());
                auto &multicast_sub_cmd =
                    std::get<std::vector<CQDispatchWritePackedMulticastSubCmd>>(common_sub_cmds[kernel_id]);
                multicast_sub_cmd.reserve(dst_noc_multicast_info.size());
                for (const auto &mcast_dests: dst_noc_multicast_info) {
                    multicast_sub_cmd.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                        .noc_xy_addr = this->device->get_noc_multicast_encoding(
                            this->noc_index, std::get<CoreRange>(mcast_dests.first)),
                        .num_mcast_dests = mcast_dests.second});
                }
            }
        }
    }
    // Reserve 2x for unique rtas as we pontentially split the cmds due to not fitting in one prefetch cmd
    // Common rtas are always expected to fit in one prefetch cmd
    this->cached_program_command_sequences[program.id].runtime_args_command_sequences = {};
    this->cached_program_command_sequences[program.id].runtime_args_command_sequences.reserve(
        2 * unique_processors.size() + common_kernels.size());
    std::vector<std::pair<uint32_t, uint32_t>> runtime_args_data_index;
    runtime_args_data_index.reserve(2 * (unique_processors.size() + common_kernels.size()));
    // Array of cmd idx, # sub cmds, rt arg offset, rt arg len
    // Currently we only really need the base cmd idx since they are sequential, and the rt arg len is currently the
    // same for all splits
    for (const uint32_t &processor_idx: unique_processors) {
        generate_dispatch_write_packed(
            this->cached_program_command_sequences[program.id].runtime_args_command_sequences,
            unique_processor_to_l1_arg_base_addr[processor_idx],
            unique_sub_cmds[processor_idx],
            unique_rt_data_and_sizes[processor_idx],
            unique_max_runtime_args_len[processor_idx],
            unique_rt_args_data[processor_idx],
            max_prefetch_command_size,
            processor_idx,
            false);
    }
    for (const uint32_t &kernel_id: common_kernels) {
        std::visit(
            [&](auto &&sub_cmds) {
                generate_dispatch_write_packed(
                    this->cached_program_command_sequences[program.id].runtime_args_command_sequences,
                    common_processor_to_l1_arg_base_addr[kernel_id],
                    sub_cmds,
                    common_rt_data_and_sizes[kernel_id],
                    common_max_runtime_args_len[kernel_id],
                    common_rt_args_data[kernel_id],
                    max_prefetch_command_size,
                    kernel_id,
                    true);
            },
            common_sub_cmds[kernel_id]);
    }
    uint32_t runtime_args_fetch_size_bytes = 0;
    for (const auto &cmds: this->cached_program_command_sequences[program.id].runtime_args_command_sequences) {
        // BRISC, NCRISC, TRISC...
        runtime_args_fetch_size_bytes += cmds.size_bytes();
    }
    this->cached_program_command_sequences[program.id].runtime_args_fetch_size_bytes = runtime_args_fetch_size_bytes;
}

void EnqueueProgramCommand::assemble_device_commands() {
    auto &cached_program_command_sequence = this->cached_program_command_sequences[this->program.id];
    if (!program.loaded_onto_device) {
        // Calculate size of command and fill program indices of data to update
        // TODO: Would be nice if we could pull this out of program
        uint32_t cmd_sequence_sizeB = 0;

        for (const auto &[dst, transfer_info_vec]: program.program_transfer_info.multicast_semaphores) {
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            for (const auto &transfer_info: transfer_info_vec) {
                for (const auto &dst_noc_info: transfer_info.dst_noc_info) {
                    TT_ASSERT(
                        transfer_info.data.size() == write_packed_len,
                        "Not all data vectors in write packed semaphore cmd equal in len");
                    num_packed_cmds += 1;
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd), L1_ALIGNMENT);
            uint32_t mcast_payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;
            cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + mcast_payload_sizeB, PCIE_ALIGNMENT);
        }

        for (const auto &[dst, transfer_info_vec]: program.program_transfer_info.unicast_semaphores) {
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            for (const auto &transfer_info: transfer_info_vec) {
                for (const auto &dst_noc_info: transfer_info.dst_noc_info) {
                    TT_ASSERT(
                        transfer_info.data.size() == write_packed_len,
                        "Not all data vectors in write packed semaphore cmd equal in len");
                    num_packed_cmds += 1;
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT);
            uint32_t ucast_payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;
            cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + ucast_payload_sizeB, PCIE_ALIGNMENT);
        }

        const auto &circular_buffers_unique_coreranges = program.circular_buffers_unique_coreranges();
        const uint16_t num_multicast_cb_sub_cmds = circular_buffers_unique_coreranges.size();
        uint32_t cb_configs_payload_start =
            (cmd_sequence_sizeB + CQ_PREFETCH_CMD_BARE_MIN_SIZE +
             align(num_multicast_cb_sub_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd), L1_ALIGNMENT)) /
            sizeof(uint32_t);
        uint32_t mcast_cb_payload_sizeB = 0;
        uint16_t cb_config_size_bytes = 0;
        uint32_t aligned_cb_config_size_bytes = 0;
        std::vector<std::vector<uint32_t>> cb_config_payloads(
            num_multicast_cb_sub_cmds,
            std::vector<uint32_t>(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * NUM_CIRCULAR_BUFFERS, 0));
        std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_cb_config_sub_cmds;
        std::vector<std::pair<const void *, uint32_t>> multicast_cb_config_data;
        if (num_multicast_cb_sub_cmds > 0) {
            multicast_cb_config_sub_cmds.reserve(num_multicast_cb_sub_cmds);
            multicast_cb_config_data.reserve(num_multicast_cb_sub_cmds);
            cached_program_command_sequence.circular_buffers_on_core_ranges.resize(num_multicast_cb_sub_cmds);
            uint32_t i = 0;
            uint32_t max_overall_base_index = 0;
            for (const CoreRange &core_range: circular_buffers_unique_coreranges) {
                const CoreCoord physical_start = device->worker_core_from_logical_core(core_range.start);
                const CoreCoord physical_end = device->worker_core_from_logical_core(core_range.end);

                const uint32_t num_receivers = core_range.size();
                auto &cb_config_payload = cb_config_payloads[i];
                uint32_t max_base_index = 0;
                const auto &circular_buffers_on_corerange = program.circular_buffers_on_corerange(core_range);
                cached_program_command_sequence.circular_buffers_on_core_ranges[i].reserve(
                    circular_buffers_on_corerange.size());
                for (const shared_ptr<CircularBuffer> &cb: circular_buffers_on_corerange) {
                    cached_program_command_sequence.circular_buffers_on_core_ranges[i].emplace_back(cb);
                    const uint32_t cb_address = cb->address() >> 4;
                    const uint32_t cb_size = cb->size() >> 4;
                    for (const auto &buffer_index: cb->buffer_indices()) {
                        // 1 cmd for all 32 buffer indices, populate with real data for specified indices

                        // cb config payload
                        const uint32_t base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * (uint32_t)buffer_index;
                        cb_config_payload[base_index] = cb_address;
                        cb_config_payload[base_index + 1] = cb_size;
                        cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                        cb_config_payload[base_index + 3] = cb->page_size(buffer_index) >> 4;
                        max_base_index = max(max_base_index, base_index);
                    }
                }
                multicast_cb_config_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                    .noc_xy_addr = this->device->get_noc_multicast_encoding(
                        this->noc_index, CoreRange(physical_start, physical_end)),
                    .num_mcast_dests = (uint32_t)core_range.size()});
                multicast_cb_config_data.emplace_back(
                    cb_config_payload.data(),
                    (max_base_index + UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG) * sizeof(uint32_t));
                max_overall_base_index = max(max_overall_base_index, max_base_index);
                i++;
            }
            cb_config_size_bytes =
                (max_overall_base_index + UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG) * sizeof(uint32_t);
            aligned_cb_config_size_bytes = align(cb_config_size_bytes, L1_ALIGNMENT);
            const uint32_t aligned_cb_config_data_sizeB = aligned_cb_config_size_bytes * num_multicast_cb_sub_cmds;
            const uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_multicast_cb_sub_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd),
                L1_ALIGNMENT);
            mcast_cb_payload_sizeB = dispatch_cmd_sizeB + aligned_cb_config_data_sizeB;
            cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + mcast_cb_payload_sizeB, PCIE_ALIGNMENT);
        }

        // Program Binaries and Go Signals
        // Get launch msg data while getting size of cmds
        // TODO: move program binaries to here as well
        for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
            const auto &kg_transfer_info = program.program_transfer_info.kernel_bins[buffer_idx];
            for (int kernel_idx = 0; kernel_idx < kg_transfer_info.dst_base_addrs.size(); kernel_idx++) {
                for (const pair<transfer_info_cores, uint32_t> &dst_noc_info: kg_transfer_info.dst_noc_info) {
                    cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
                    cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
                }
            }
        }

        // Wait Cmd
        if (program.program_transfer_info.num_active_cores > 0) {
            cmd_sequence_sizeB += CQ_PREFETCH_CMD_BARE_MIN_SIZE;
        }

        std::vector<std::pair<const void *, uint32_t>> multicast_go_signal_data;
        std::vector<std::pair<const void *, uint32_t>> unicast_go_signal_data;
        std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_go_signal_sub_cmds;
        std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_go_signal_sub_cmds;
        const uint32_t go_signal_sizeB = sizeof(launch_msg_t);
        for (KernelGroup &kernel_group: program.get_kernel_groups(CoreType::WORKER)) {
            kernel_group.launch_msg.mode = DISPATCH_MODE_DEV;
            const void *launch_message_data = (const void *)(&kernel_group.launch_msg);
            for (const CoreRange &core_range: kernel_group.core_ranges.ranges()) {
                CoreCoord physical_start =
                    device->physical_core_from_logical_core(core_range.start, kernel_group.get_core_type());
                CoreCoord physical_end =
                    device->physical_core_from_logical_core(core_range.end, kernel_group.get_core_type());

                multicast_go_signal_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                    .noc_xy_addr = this->device->get_noc_multicast_encoding(
                        this->noc_index, CoreRange(physical_start, physical_end)),
                    .num_mcast_dests = (uint32_t)core_range.size()});
                multicast_go_signal_data.emplace_back(launch_message_data, go_signal_sizeB);
            }
        }
        if (multicast_go_signal_sub_cmds.size() > 0) {
            uint32_t num_multicast_sub_cmds = multicast_go_signal_sub_cmds.size();
            uint32_t aligned_go_signal_data_sizeB = align(sizeof(launch_msg_t), L1_ALIGNMENT) * num_multicast_sub_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_multicast_sub_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd),
                L1_ALIGNMENT);
            uint32_t mcast_payload_sizeB = dispatch_cmd_sizeB + aligned_go_signal_data_sizeB;
            cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + mcast_payload_sizeB, PCIE_ALIGNMENT);
        }

        for (KernelGroup &kernel_group: program.get_kernel_groups(CoreType::ETH)) {
            kernel_group.launch_msg.mode = DISPATCH_MODE_DEV;
            const void *launch_message_data = (const void *)(&kernel_group.launch_msg);
            for (const CoreRange &core_range: kernel_group.core_ranges.ranges()) {
                for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                    for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                        CoreCoord physical_coord =
                            device->physical_core_from_logical_core(CoreCoord({x, y}), kernel_group.get_core_type());
                        unicast_go_signal_sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                            .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, physical_coord)});
                        unicast_go_signal_data.emplace_back(launch_message_data, go_signal_sizeB);
                    }
                }
            }
        }
        if (unicast_go_signal_sub_cmds.size() > 0) {
            uint32_t num_unicast_sub_cmds = unicast_go_signal_sub_cmds.size();
            uint32_t aligned_go_signal_data_sizeB = align(sizeof(launch_msg_t), L1_ALIGNMENT) * num_unicast_sub_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_unicast_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd),
                L1_ALIGNMENT);
            uint32_t ucast_payload_sizeB = dispatch_cmd_sizeB + aligned_go_signal_data_sizeB;
            cmd_sequence_sizeB += align(sizeof(CQPrefetchCmd) + ucast_payload_sizeB, PCIE_ALIGNMENT);
        }

        cached_program_command_sequence.program_command_sequence = HostMemDeviceCommand(cmd_sequence_sizeB);

        auto &program_command_sequence = cached_program_command_sequence.program_command_sequence;

        // Semaphores
        // Multicast Semaphore Cmd
        for (const auto &[dst, transfer_info_vec]: program.program_transfer_info.multicast_semaphores) {
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            std::vector<CQDispatchWritePackedMulticastSubCmd> multicast_sub_cmds;
            std::vector<std::pair<const void *, uint32_t>> sem_data;

            for (const auto &transfer_info: transfer_info_vec) {
                for (const auto &dst_noc_info: transfer_info.dst_noc_info) {
                    num_packed_cmds += 1;
                    multicast_sub_cmds.emplace_back(CQDispatchWritePackedMulticastSubCmd{
                        .noc_xy_addr = this->device->get_noc_multicast_encoding(
                            this->noc_index, std::get<CoreRange>(dst_noc_info.first)),
                        .num_mcast_dests = dst_noc_info.second});
                    sem_data.emplace_back(transfer_info.data.data(), transfer_info.data.size() * sizeof(uint32_t));
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd), L1_ALIGNMENT);
            uint32_t payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;

            program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                num_packed_cmds, 
                dst, 
                write_packed_len * sizeof(uint32_t), 
                payload_sizeB, 
                multicast_sub_cmds, 
                sem_data);
        }

        // Unicast Semaphore Cmd
        for (const auto &[dst, transfer_info_vec]: program.program_transfer_info.unicast_semaphores) {
            // TODO: loop over things inside transfer_info[i]
            uint32_t num_packed_cmds = 0;
            uint32_t write_packed_len = transfer_info_vec[0].data.size();

            std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds;
            std::vector<std::pair<const void *, uint32_t>> sem_data;

            for (const auto &transfer_info: transfer_info_vec) {
                for (const auto &dst_noc_info: transfer_info.dst_noc_info) {
                    num_packed_cmds += 1;
                    unicast_sub_cmds.emplace_back(CQDispatchWritePackedUnicastSubCmd{
                        .noc_xy_addr = this->device->get_noc_unicast_encoding(
                            this->noc_index, std::get<CoreCoord>(dst_noc_info.first))});
                    sem_data.emplace_back(transfer_info.data.data(), transfer_info.data.size() * sizeof(uint32_t));
                }
            }

            uint32_t aligned_semaphore_data_sizeB =
                align(write_packed_len * sizeof(uint32_t), L1_ALIGNMENT) * num_packed_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_packed_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT);
            uint32_t payload_sizeB = dispatch_cmd_sizeB + aligned_semaphore_data_sizeB;

            program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_packed_cmds, 
                dst, 
                write_packed_len * sizeof(uint32_t), 
                payload_sizeB, 
                unicast_sub_cmds, 
                sem_data);
        }

        // CB Configs commands
        cached_program_command_sequence.cb_configs_payload_start = cb_configs_payload_start;
        cached_program_command_sequence.aligned_cb_config_size_bytes = aligned_cb_config_size_bytes;
        if (num_multicast_cb_sub_cmds > 0) {
            program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                num_multicast_cb_sub_cmds,
                CIRCULAR_BUFFER_CONFIG_BASE,
                cb_config_size_bytes,
                mcast_cb_payload_sizeB,
                multicast_cb_config_sub_cmds,
                multicast_cb_config_data);
        }

        // Program Binaries
        for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
            const auto &kg_transfer_info = program.program_transfer_info.kernel_bins[buffer_idx];
            for (int kernel_idx = 0; kernel_idx < kg_transfer_info.dst_base_addrs.size(); kernel_idx++) {
                for (const pair<transfer_info_cores, uint32_t> &dst_noc_info: kg_transfer_info.dst_noc_info) {
                    uint32_t noc_encoding;
                    std::visit(
                        [&](auto &&cores) {
                            using T = std::decay_t<decltype(cores)>;
                            if constexpr (std::is_same_v<T, CoreRange>) {
                                noc_encoding = this->device->get_noc_multicast_encoding(this->noc_index, cores);
                            } else {
                                noc_encoding = this->device->get_noc_unicast_encoding(this->noc_index, cores);
                            }
                        },
                        dst_noc_info.first);
                    program_command_sequence.add_dispatch_write_linear(
                        false,                // flush_prefetch
                        dst_noc_info.second,  // num_mcast_dests
                        noc_encoding,         // noc_xy_addr
                        kg_transfer_info.dst_base_addrs[kernel_idx],
                        align(kg_transfer_info.lengths[kernel_idx], NOC_DRAM_ALIGNMENT_BYTES));
                    // Difference between prefetch total relayed pages and dispatch write linear
                    uint32_t relayed_bytes =
                        align(kg_transfer_info.lengths[kernel_idx], HostMemDeviceCommand::PROGRAM_PAGE_SIZE);
                    // length_adjust needs to be aligned to NOC_DRAM_ALIGNMENT
                    uint16_t length_adjust =
                        uint16_t(relayed_bytes - align(kg_transfer_info.lengths[kernel_idx], NOC_DRAM_ALIGNMENT_BYTES));

                    uint32_t base_address, page_offset;
                    if (kg_transfer_info.page_offsets[kernel_idx] > CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK) {
                        const uint32_t num_banks =
                            this->device->num_banks(this->program.kg_buffers[buffer_idx]->buffer_type());
                        page_offset = kg_transfer_info.page_offsets[kernel_idx] % num_banks;
                        uint32_t num_full_pages_written_per_bank =
                            kg_transfer_info.page_offsets[kernel_idx] / num_banks;
                        base_address =
                            this->program.kg_buffers[buffer_idx]->address() +
                            num_full_pages_written_per_bank * this->program.kg_buffers[buffer_idx]->page_size();
                    } else {
                        base_address = this->program.kg_buffers[buffer_idx]->address();
                        page_offset = kg_transfer_info.page_offsets[kernel_idx];
                    }

                    program_command_sequence.add_prefetch_relay_paged(
                        true,  // is_dram
                        page_offset,
                        base_address,
                        this->program.kg_buffers[buffer_idx]->page_size(),
                        relayed_bytes / this->program.kg_buffers[buffer_idx]->page_size(),
                        length_adjust);
                }
            }
        }

        // Wait Noc Write Barrier, wait for binaries/configs to be written to worker cores
        if (program.program_transfer_info.num_active_cores > 0) {
            program_command_sequence.add_dispatch_wait(
                true, 
                DISPATCH_MESSAGE_ADDR, 
                0, 
                0, 
                false, 
                false);
        }

        // Go Signals
        if (multicast_go_signal_sub_cmds.size() > 0) {
            uint32_t num_multicast_sub_cmds = multicast_go_signal_sub_cmds.size();
            uint32_t aligned_go_signal_data_sizeB = align(sizeof(launch_msg_t), L1_ALIGNMENT) * num_multicast_sub_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_multicast_sub_cmds * sizeof(CQDispatchWritePackedMulticastSubCmd),
                L1_ALIGNMENT);
            uint32_t mcast_payload_sizeB = dispatch_cmd_sizeB + aligned_go_signal_data_sizeB;
            program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedMulticastSubCmd>(
                num_multicast_sub_cmds,
                GET_MAILBOX_ADDRESS_HOST(launch),
                go_signal_sizeB,
                mcast_payload_sizeB,
                multicast_go_signal_sub_cmds,
                multicast_go_signal_data);
        }

        if (unicast_go_signal_sub_cmds.size() > 0) {
            uint32_t num_unicast_sub_cmds = unicast_go_signal_sub_cmds.size();
            uint32_t aligned_go_signal_data_sizeB = align(sizeof(launch_msg_t), L1_ALIGNMENT) * num_unicast_sub_cmds;
            uint32_t dispatch_cmd_sizeB = align(
                sizeof(CQDispatchCmd) + num_unicast_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd),
                L1_ALIGNMENT);
            uint32_t ucast_payload_sizeB = dispatch_cmd_sizeB + aligned_go_signal_data_sizeB;
            program_command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
                num_unicast_sub_cmds,
                GET_ETH_MAILBOX_ADDRESS_HOST(launch),
                go_signal_sizeB,
                ucast_payload_sizeB,
                unicast_go_signal_sub_cmds,
                unicast_go_signal_data);
        }
    } else {
        auto &program_command_sequence = cached_program_command_sequence.program_command_sequence;
        uint32_t *cb_config_payload =
            (uint32_t *)program_command_sequence.data() + cached_program_command_sequence.cb_configs_payload_start;
        uint32_t aligned_cb_config_size_bytes = cached_program_command_sequence.aligned_cb_config_size_bytes;
        for (const auto &cbs_on_core_range: cached_program_command_sequence.circular_buffers_on_core_ranges) {
            for (const shared_ptr<CircularBuffer> &cb: cbs_on_core_range) {
                const uint32_t cb_address = cb->address() >> 4;
                const uint32_t cb_size = cb->size() >> 4;
                for (const auto &buffer_index: cb->buffer_indices()) {
                    // 1 cmd for all 32 buffer indices, populate with real data for specified indices

                    // cb config payload
                    uint32_t base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * (uint32_t)buffer_index;
                    cb_config_payload[base_index] = cb_address;
                    cb_config_payload[base_index + 1] = cb_size;
                    cb_config_payload[base_index + 2] = cb->num_pages(buffer_index);
                    cb_config_payload[base_index + 3] = cb->page_size(buffer_index) >> 4;
                }
            }
            cb_config_payload += aligned_cb_config_size_bytes / sizeof(uint32_t);
        }
    }
}

#if 0 // TODO: Revise this
void EnqueueProgramCommand::process() {
    // Calculate all commands size and determine how many fetch q entries to use

    // Preamble, some waits and stalls
    // can be written directly to the issue queue
    if (!program.loaded_onto_device) {
        this->assemble_preamble_commands(true);
        // Runtime Args Command Sequence
        this->assemble_runtime_args_commands();
        // Main Command Sequence
        this->assemble_device_commands();
    } else {
        static constexpr uint32_t count_offset = (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, wait.count));
        this->cached_program_command_sequences[program.id].preamble_command_sequence.update_cmd_sequence(
            count_offset, &this->expected_num_workers_completed, sizeof(uint32_t));
        this->assemble_device_commands();
    }

    const auto &cached_program_command_sequence = this->cached_program_command_sequences[program.id];
    uint32_t preamble_fetch_size_bytes = cached_program_command_sequence.preamble_command_sequence.size_bytes();
    uint32_t runtime_args_fetch_size_bytes = cached_program_command_sequence.runtime_args_fetch_size_bytes;
    uint32_t program_fetch_size_bytes = cached_program_command_sequence.program_command_sequence.size_bytes();
    uint32_t total_fetch_size_bytes =
        preamble_fetch_size_bytes + runtime_args_fetch_size_bytes + program_fetch_size_bytes;

    CoreType dispatch_core_type =
        dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    if (total_fetch_size_bytes <= dispatch_constants::get(dispatch_core_type).max_prefetch_command_size()) {
        this->manager.issue_queue_reserve(total_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);

        this->manager.cq_write(
            cached_program_command_sequence.preamble_command_sequence.data(), 
            preamble_fetch_size_bytes, 
            write_ptr);
        write_ptr += preamble_fetch_size_bytes;

        for (const auto &cmds: cached_program_command_sequence.runtime_args_command_sequences) {
            this->manager.cq_write(cmds.data(), cmds.size_bytes(), write_ptr);
            write_ptr += cmds.size_bytes();
        }

        this->manager.cq_write(
            cached_program_command_sequence.program_command_sequence.data(), 
            program_fetch_size_bytes, 
            write_ptr);

        this->manager.issue_queue_push_back(total_fetch_size_bytes, this->command_queue_id);

        // One fetch queue entry for entire program
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(total_fetch_size_bytes, this->command_queue_id);
    } else {
        this->manager.issue_queue_reserve(preamble_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(
            cached_program_command_sequence.preamble_command_sequence.data(), 
            preamble_fetch_size_bytes, 
            write_ptr);
        this->manager.issue_queue_push_back(preamble_fetch_size_bytes, this->command_queue_id);
        // One fetch queue entry for just the wait and stall, very inefficient
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(preamble_fetch_size_bytes, this->command_queue_id);

        // TODO: We can pack multiple RT args into one fetch q entry
        for (const auto &cmds: cached_program_command_sequence.runtime_args_command_sequences) {
            uint32_t fetch_size_bytes = cmds.size_bytes();
            this->manager.issue_queue_reserve(fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(cmds.data(), fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for each runtime args write location, e.g. BRISC/NCRISC/TRISC/ERISC
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
        }

        this->manager.issue_queue_reserve(program_fetch_size_bytes, this->command_queue_id);
        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(
            cached_program_command_sequence.program_command_sequence.data(), 
            program_fetch_size_bytes, 
            write_ptr);
        this->manager.issue_queue_push_back(program_fetch_size_bytes, this->command_queue_id);
        // One fetch queue entry for rest of program commands
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(program_fetch_size_bytes, this->command_queue_id);
    }

    // Front load generating and caching preamble without stall during program loading stage
    if (!program.loaded_onto_device) {
        this->assemble_preamble_commands(false);
        program.loaded_onto_device = true;
    }
}
#endif

void EnqueueProgramCommand::process() {
    // Calculate all commands size and determine how many fetch q entries to use

    // Preamble, some waits and stalls
    // can be written directly to the issue queue
    if (!program.loaded_onto_device) {
        this->assemble_preamble_commands(true);
        // Runtime Args Command Sequence
        this->assemble_runtime_args_commands();
        // Main Command Sequence
        this->assemble_device_commands();
    } else {
        static constexpr uint32_t count_offset = (sizeof(CQPrefetchCmd) + offsetof(CQDispatchCmd, wait.count));
        this->cached_program_command_sequences[program.id].preamble_command_sequence.update_cmd_sequence(
            count_offset, &this->expected_num_workers_completed, sizeof(uint32_t));
        this->assemble_device_commands();
    }

    const auto &cached_program_command_sequence = this->cached_program_command_sequences[program.id];
    uint32_t preamble_fetch_size_bytes = cached_program_command_sequence.preamble_command_sequence.size_bytes();
    uint32_t runtime_args_fetch_size_bytes = cached_program_command_sequence.runtime_args_fetch_size_bytes;
    uint32_t program_fetch_size_bytes = cached_program_command_sequence.program_command_sequence.size_bytes();
    uint32_t total_fetch_size_bytes =
        preamble_fetch_size_bytes + runtime_args_fetch_size_bytes + program_fetch_size_bytes;

#if 0
    CoreType dispatch_core_type =
        dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    if (total_fetch_size_bytes <= dispatch_constants::get(dispatch_core_type).max_prefetch_command_size()) {
        this->manager.issue_queue_reserve(total_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);

        this->manager.cq_write(
            cached_program_command_sequence.preamble_command_sequence.data(), 
            preamble_fetch_size_bytes, 
            write_ptr);
        write_ptr += preamble_fetch_size_bytes;

        for (const auto &cmds: cached_program_command_sequence.runtime_args_command_sequences) {
            this->manager.cq_write(cmds.data(), cmds.size_bytes(), write_ptr);
            write_ptr += cmds.size_bytes();
        }

        this->manager.cq_write(
            cached_program_command_sequence.program_command_sequence.data(), 
            program_fetch_size_bytes, 
            write_ptr);

        this->manager.issue_queue_push_back(total_fetch_size_bytes, this->command_queue_id);

        // One fetch queue entry for entire program
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(total_fetch_size_bytes, this->command_queue_id);
    } else {
        this->manager.issue_queue_reserve(preamble_fetch_size_bytes, this->command_queue_id);
        uint32_t write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(
            cached_program_command_sequence.preamble_command_sequence.data(), 
            preamble_fetch_size_bytes, 
            write_ptr);
        this->manager.issue_queue_push_back(preamble_fetch_size_bytes, this->command_queue_id);
        // One fetch queue entry for just the wait and stall, very inefficient
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(preamble_fetch_size_bytes, this->command_queue_id);

        // TODO: We can pack multiple RT args into one fetch q entry
        for (const auto &cmds: cached_program_command_sequence.runtime_args_command_sequences) {
            uint32_t fetch_size_bytes = cmds.size_bytes();
            this->manager.issue_queue_reserve(fetch_size_bytes, this->command_queue_id);
            write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
            this->manager.cq_write(cmds.data(), fetch_size_bytes, write_ptr);
            this->manager.issue_queue_push_back(fetch_size_bytes, this->command_queue_id);
            // One fetch queue entry for each runtime args write location, e.g. BRISC/NCRISC/TRISC/ERISC
            this->manager.fetch_queue_reserve_back(this->command_queue_id);
            this->manager.fetch_queue_write(fetch_size_bytes, this->command_queue_id);
        }

        this->manager.issue_queue_reserve(program_fetch_size_bytes, this->command_queue_id);
        write_ptr = this->manager.get_issue_queue_write_ptr(this->command_queue_id);
        this->manager.cq_write(
            cached_program_command_sequence.program_command_sequence.data(), 
            program_fetch_size_bytes, 
            write_ptr);
        this->manager.issue_queue_push_back(program_fetch_size_bytes, this->command_queue_id);
        // One fetch queue entry for rest of program commands
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(program_fetch_size_bytes, this->command_queue_id);
    }
#endif
    this->cq_manager->reserve(preamble_fetch_size_bytes);
    this->cq_manager->write(
        cached_program_command_sequence.preamble_command_sequence.data(), 
        preamble_fetch_size_bytes, 
        0);
    this->cq_manager->push(preamble_fetch_size_bytes);

    for (const auto &cmds: cached_program_command_sequence.runtime_args_command_sequences) {
        uint32_t fetch_size_bytes = cmds.size_bytes();
        this->cq_manager->reserve(fetch_size_bytes);
        this->cq_manager->write(cmds.data(), fetch_size_bytes, 0);
        this->cq_manager->push(fetch_size_bytes);
    }

    this->cq_manager->reserve(program_fetch_size_bytes);
    this->cq_manager->write(
        cached_program_command_sequence.program_command_sequence.data(), 
        program_fetch_size_bytes, 
        0);
    this->cq_manager->push(program_fetch_size_bytes);

    // Front load generating and caching preamble without stall during program loading stage
    if (!program.loaded_onto_device) {
        this->assemble_preamble_commands(false);
        program.loaded_onto_device = true;
    }

    // [RONIN]
    this->cq_manager->launch_kernels();
}

#if 0 // TODO: Revise this
//
//    EnqueueRecordEventCommand
//

EnqueueRecordEventCommand::EnqueueRecordEventCommand(
        uint32_t command_queue_id,
        Device *device,
        NOC noc_index,
        SystemMemoryManager &manager,
        uint32_t event_id,
        uint32_t expected_num_workers_completed,
        bool clear_count):
            command_queue_id(command_queue_id),
            device(device),
            noc_index(noc_index),
            manager(manager),
            event_id(event_id),
            expected_num_workers_completed(expected_num_workers_completed),
            clear_count(clear_count) { }

void EnqueueRecordEventCommand::process() {
    std::vector<uint32_t> event_payload(dispatch_constants::EVENT_PADDED_SIZE / sizeof(uint32_t), 0);
    event_payload[0] = this->event_id;

    uint8_t num_hw_cqs =
        this->device->num_hw_cqs();  // Device initialize asserts that there can only be a maximum of 2 HW CQs
    uint32_t packed_event_payload_sizeB =
        align(sizeof(CQDispatchCmd) + num_hw_cqs * sizeof(CQDispatchWritePackedUnicastSubCmd), L1_ALIGNMENT) +
        (align(dispatch_constants::EVENT_PADDED_SIZE, L1_ALIGNMENT) * num_hw_cqs);
    uint32_t packed_write_sizeB = align(sizeof(CQPrefetchCmd) + packed_event_payload_sizeB, PCIE_ALIGNMENT);

    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        packed_write_sizeB +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PACKED + unicast subcmds + event
                              // payload
        align(
            CQ_PREFETCH_CMD_BARE_MIN_SIZE + dispatch_constants::EVENT_PADDED_SIZE,
            PCIE_ALIGNMENT);  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_LINEAR_HOST + event ID

    void *cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait(
        false, 
        DISPATCH_MESSAGE_ADDR, 
        this->expected_num_workers_completed, 
        this->clear_count);

    CoreType core_type = dispatch_core_manager::get(num_hw_cqs).get_dispatch_core_type(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    std::vector<CQDispatchWritePackedUnicastSubCmd> unicast_sub_cmds(num_hw_cqs);
    std::vector<std::pair<const void *, uint32_t>> event_payloads(num_hw_cqs);

    for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
        tt_cxy_pair dispatch_location;
        if (device->is_mmio_capable()) {
            dispatch_location =
                dispatch_core_manager::get(num_hw_cqs).dispatcher_core(this->device->id(), channel, cq_id);
        } else {
            dispatch_location =
                dispatch_core_manager::get(num_hw_cqs).dispatcher_d_core(this->device->id(), channel, cq_id);
        }

        CoreCoord dispatch_physical_core = get_physical_core_coordinate(dispatch_location, core_type);
        unicast_sub_cmds[cq_id] = CQDispatchWritePackedUnicastSubCmd{
            .noc_xy_addr = this->device->get_noc_unicast_encoding(this->noc_index, dispatch_physical_core)};
        event_payloads[cq_id] = {event_payload.data(), event_payload.size() * sizeof(uint32_t)};
    }

    uint32_t address = 
        (this->command_queue_id == 0) ? 
            CQ0_COMPLETION_LAST_EVENT : 
            CQ1_COMPLETION_LAST_EVENT;
    command_sequence.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_hw_cqs,
        address,
        dispatch_constants::EVENT_PADDED_SIZE,
        packed_event_payload_sizeB,
        unicast_sub_cmds,
        event_payloads);

    bool flush_prefetch = true;
    command_sequence.add_dispatch_write_host<true>(
        flush_prefetch, 
        dispatch_constants::EVENT_PADDED_SIZE, 
        event_payload.data());

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

//
//    EnqueueWaitForEventCommand
//

EnqueueWaitForEventCommand::EnqueueWaitForEventCommand(
        uint32_t command_queue_id,
        Device *device,
        SystemMemoryManager &manager,
        const Event &sync_event,
        bool clear_count):
            command_queue_id(command_queue_id),
            device(device),
            manager(manager),
            sync_event(sync_event),
            clear_count(clear_count) {
    this->dispatch_core_type = dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
    // Should not be encountered under normal circumstances (record, wait) unless user is modifying sync event ID.
    // TT_ASSERT(command_queue_id != sync_event.cq_id || event != sync_event.event_id,
    //     "EnqueueWaitForEventCommand cannot wait on it's own event id on the same CQ. Event ID: {} CQ ID: {}",
    //     event, command_queue_id);
}

void EnqueueWaitForEventCommand::process() {
    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE;  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT

    void *cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    uint32_t last_completed_event_address =
        (sync_event.cq_id == 0) ? 
            CQ0_COMPLETION_LAST_EVENT : 
            CQ1_COMPLETION_LAST_EVENT;
    command_sequence.add_dispatch_wait(
        false, 
        last_completed_event_address, 
        sync_event.event_id, 
        this->clear_count);

    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}
#endif

// [RONIN] SKIPPED: EnqueueRecordEventCommand
// [RONIN] SKIPPED: EnqueueWaitForEventCommand

//
//    EnqueueTraceCommand
//

EnqueueTraceCommand::EnqueueTraceCommand(
        uint32_t command_queue_id,
        Device *device,
#if 0 // TODO: Revise this
        SystemMemoryManager &manager,
#endif
        CQManager *cq_manager,
        Buffer &buffer,
        uint32_t &expected_num_workers_completed):
            command_queue_id(command_queue_id),
            buffer(buffer),
            device(device),
#if 0 // TODO: Revise this
            manager(manager),
#endif
            cq_manager(cq_manager),
            expected_num_workers_completed(expected_num_workers_completed),
            clear_count(true) { }

void EnqueueTraceCommand::process() {
    uint32_t cmd_sequence_sizeB =
        CQ_PREFETCH_CMD_BARE_MIN_SIZE +  // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        CQ_PREFETCH_CMD_BARE_MIN_SIZE;   // CQ_PREFETCH_CMD_EXEC_BUF

#if 0 // TODO: Revise this
    void *cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
#endif
    void *cmd_region = this->cq_manager->reserve(cmd_sequence_sizeB);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    command_sequence.add_dispatch_wait(
        false, 
        DISPATCH_MESSAGE_ADDR, 
        this->expected_num_workers_completed, 
        this->clear_count);

    if (this->clear_count) {
        this->expected_num_workers_completed = 0;
    }

    uint32_t page_size = buffer.page_size();
    uint32_t page_size_log2 = __builtin_ctz(page_size);
    TT_ASSERT((page_size & (page_size - 1)) == 0, "Page size must be a power of 2");

    command_sequence.add_prefetch_exec_buf(
        buffer.address(), 
        page_size_log2, 
        buffer.num_pages());

#if 0 // TODO: Revise this
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    const bool stall_prefetcher = true;
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id, stall_prefetcher);
#endif
    this->cq_manager->push(cmd_sequence_sizeB);
}

//
//    EnqueueTerminateCommand
//

#if 0 // TODO: Revise this
EnqueueTerminateCommand::EnqueueTerminateCommand(
        uint32_t command_queue_id, 
        Device *device, 
        SystemMemoryManager &manager) :
            command_queue_id(command_queue_id), 
            device(device), 
            manager(manager) { }
#endif

EnqueueTerminateCommand::EnqueueTerminateCommand(
        uint32_t command_queue_id, 
        Device *device, 
        CQManager *cq_manager):
            command_queue_id(command_queue_id), 
            device(device), 
            cq_manager(cq_manager) { }

#if 0 // TODO: Revise this
void EnqueueTerminateCommand::process() {
    // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
    // CQ_PREFETCH_CMD_TERMINATE
    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE;

    // dispatch and prefetch terminate commands each needs to be a separate fetch queue entry
    void *cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand dispatch_command_sequence(cmd_region, cmd_sequence_sizeB);
    dispatch_command_sequence.add_dispatch_terminate();
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);

    cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand prefetch_command_sequence(cmd_region, cmd_sequence_sizeB);
    prefetch_command_sequence.add_prefetch_terminate();
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}
#endif

void EnqueueTerminateCommand::process() {
    // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
    // CQ_PREFETCH_CMD_TERMINATE
    uint32_t cmd_sequence_sizeB = CQ_PREFETCH_CMD_BARE_MIN_SIZE;

    // dispatch and prefetch terminate commands each needs to be a separate fetch queue entry
    void *cmd_region = this->cq_manager->reserve(cmd_sequence_sizeB);
    HugepageDeviceCommand dispatch_command_sequence(cmd_region, cmd_sequence_sizeB);
    dispatch_command_sequence.add_dispatch_terminate();
    this->cq_manager->push(cmd_sequence_sizeB);

    cmd_region = this->cq_manager->reserve(cmd_sequence_sizeB);
    HugepageDeviceCommand prefetch_command_sequence(cmd_region, cmd_sequence_sizeB);
    prefetch_command_sequence.add_prefetch_terminate();
    this->cq_manager->push(cmd_sequence_sizeB);
}

//
//    HWCommandQueue
//

#if 0 // TODO: Revuse this
HWCommandQueue::HWCommandQueue(Device *device, uint32_t id, NOC noc_index):
        manager(device->sysmem_manager()), 
        completion_queue_thread{} {
#endif
HWCommandQueue::HWCommandQueue(Device *device, uint32_t id, NOC noc_index):
        cq_manager(device->cq_manager()) {
    ZoneScopedN("CommandQueue_constructor");
    this->device = device;
    this->id = id;
    this->noc_index = noc_index;
#if 0 // TODO: Revise this
    this->num_entries_in_completion_q = 0;
    this->num_completed_completion_q_reads = 0;
#endif

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
    this->size_B = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device->num_hw_cqs();

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::get(device->num_hw_cqs()).completion_queue_writer_core(device->id(), channel, this->id);

#if 0 // TODO: Revise this
    this->completion_queue_writer_core = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);
#endif

#if 0 // TODO: Revise this
    this->exit_condition = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread = std::move(completion_queue_thread);
#endif
#if 0 // [RONIN]
    // Set the affinity of the completion queue reader.
    set_device_thread_affinity(this->completion_queue_thread, device->worker_thread_core);
#endif
    this->expected_num_workers_completed = 0;
}

#if 0 // TODO: Revise this
HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition) {
        this->completion_queue_thread.join();  // We errored out already prior
    } else {
        TT_ASSERT(
            this->issued_completion_q_reads.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q == this->num_completed_completion_q_reads,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted "
            "commands: {}",
            this->num_entries_in_completion_q - this->num_completed_completion_q_reads);
        this->set_exit_condition();
        this->completion_queue_thread.join();
    }
}
#endif

HWCommandQueue::~HWCommandQueue() { }

#if 0 // TODO: Revise this
void HWCommandQueue::increment_num_entries_in_completion_q() {
    // Increment num_entries_in_completion_q and inform reader thread
    // that there is work in the completion queue to process
    this->num_entries_in_completion_q++;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}

void HWCommandQueue::set_exit_condition() {
    this->exit_condition = true;
    {
        std::lock_guard lock(this->reader_thread_cv_mutex);
        this->reader_thread_cv.notify_one();
    }
}
#endif

template <typename T>
void HWCommandQueue::enqueue_command(T &command, bool blocking) {
    command.process();
    if (blocking) {
        this->finish();
    }
}

void HWCommandQueue::enqueue_read_buffer(std::shared_ptr<Buffer> buffer, void *dst, bool blocking) {
    this->enqueue_read_buffer(*buffer, dst, blocking);
}

// Read buffer command is enqueued in the issue region and
// device writes requested buffer data into the completion region
void HWCommandQueue::enqueue_read_buffer(Buffer &buffer, void *dst, bool blocking) {
    ZoneScopedN("HWCommandQueue_read_buffer");
#if 0 // TODO: Revise this
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Read Buffer cannot be used with tracing");
#endif
    TT_FATAL(
        !this->cq_manager->get_bypass_mode(), 
        "Enqueue Read Buffer cannot be used with tracing");

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    CoreType dispatch_core_type =
        dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());

    uint32_t padded_page_size = align(buffer.page_size(), ADDRESS_ALIGNMENT);
    uint32_t pages_to_read = buffer.num_pages();
    uint32_t unpadded_dst_offset = 0;
    uint32_t src_page_index = 0;

    if (is_sharded(buffer.buffer_layout())) {
        bool width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
        std::optional<BufferPageMapping> buffer_page_mapping = std::nullopt;
        if (width_split) {
            buffer_page_mapping = generate_buffer_page_mapping(buffer);
        }
        // Note that the src_page_index is the device page idx, not the host page idx
        // Since we read core by core we are reading the device pages sequentially
        const auto &cores = 
            width_split ? 
                buffer_page_mapping.value().all_cores_ : 
                corerange_to_cores(
                    buffer.shard_spec().grid(),
                    buffer.num_cores(),
                    (buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR));
        uint32_t num_total_pages = buffer.num_pages();
        uint32_t max_pages_per_shard = buffer.shard_spec().size();
        bool linear_page_copy = true;
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            uint32_t num_pages_to_read;
            if (width_split) {
                num_pages_to_read =
                    buffer_page_mapping.value().core_shard_shape_[core_id][0] * 
                    buffer.shard_spec().shape_in_pages()[1];
            } else {
                num_pages_to_read = min(num_total_pages, max_pages_per_shard);
                num_total_pages -= num_pages_to_read;
            }
            uint32_t bank_base_address = buffer.address();
            if (buffer.buffer_type() == BufferType::DRAM) {
                bank_base_address += buffer.device()->bank_offset(
                    BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
            }
            if (num_pages_to_read > 0) {
                if (width_split) {
                    uint32_t host_page = buffer_page_mapping.value().core_host_page_indices_[core_id][0];
                    src_page_index = buffer_page_mapping.value().host_page_to_dev_page_mapping_[host_page];
                    unpadded_dst_offset = host_page * buffer.page_size();
                } else {
                    unpadded_dst_offset = src_page_index * buffer.page_size();
                }

                auto command = EnqueueReadShardedBufferCommand(
                    this->id,
                    this->device,
                    this->noc_index,
                    buffer,
                    dst,
#if 0 // TODO: Revise this
                    this->manager,
#endif
                    this->cq_manager,
                    this->expected_num_workers_completed,
                    cores[core_id],
                    bank_base_address,
                    src_page_index,
                    num_pages_to_read);

#if 0 // TODO: Revise this
                this->issued_completion_q_reads.push(detail::ReadBufferDescriptor(
                    buffer.buffer_layout(),
                    buffer.page_size(),
                    padded_page_size,
                    dst,
                    unpadded_dst_offset,
                    num_pages_to_read,
                    src_page_index,
                    width_split ? 
                        (*buffer_page_mapping).dev_page_to_host_page_mapping_ : 
                        vector<std::optional<uint32_t>>()));
#endif
                this->cq_manager->config_read_buffer(
                    padded_page_size,
                    dst,
                    unpadded_dst_offset,
                    num_pages_to_read);

                src_page_index += num_pages_to_read;
                this->enqueue_command(command, false);
#if 0 // TODO: Revise this
                this->increment_num_entries_in_completion_q();
#endif
            }
        }
        if (blocking) {
            this->finish();
#if 0 // TODO: Revise this
        } else {
            std::shared_ptr<Event> event = std::make_shared<Event>();
            this->enqueue_record_event(event);
#endif
        }
    } else {
        // this is a streaming command so we don't need to break down to multiple
        auto command = EnqueueReadInterleavedBufferCommand(
            this->id,
            this->device,
            this->noc_index,
            buffer,
            dst,
#if 0 // TODO: Revise this
            this->manager,
#endif
            this->cq_manager,
            this->expected_num_workers_completed,
            src_page_index,
            pages_to_read);

#if 0 // TODO: Revise this
        this->issued_completion_q_reads.push(detail::ReadBufferDescriptor(
            buffer.buffer_layout(),
            buffer.page_size(),
            padded_page_size,
            dst,
            unpadded_dst_offset,
            pages_to_read,
            src_page_index));
#endif
        this->cq_manager->config_read_buffer(
            padded_page_size,
            dst,
            unpadded_dst_offset,
            pages_to_read);

        this->enqueue_command(command, blocking);
#if 0 // TODO: Revise this
        this->increment_num_entries_in_completion_q();
#endif
        if (!blocking) {  // should this be unconditional?
            std::shared_ptr<Event> event = std::make_shared<Event>();
            this->enqueue_record_event(event);
        }
    }
}

void HWCommandQueue::enqueue_write_buffer(
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<const Buffer>
        > buffer,
        HostDataType src,
        bool blocking) {
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    std::visit(
        [this, &buffer, &blocking](auto &&data) {
            using T = std::decay_t<decltype(data)>;
            std::visit(
                [this, &buffer, &blocking, &data](auto &&b) {
                    using type_buf = std::decay_t<decltype(b)>;
                    if constexpr (std::is_same_v<T, const void *>) {
                        if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                            this->enqueue_write_buffer(*b, data, blocking);
                        } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                            this->enqueue_write_buffer(b.get(), data, blocking);
                        }
                    } else {
                        if constexpr (std::is_same_v<type_buf, std::shared_ptr<const Buffer>>) {
                            this->enqueue_write_buffer(*b, data.get()->data(), blocking);
                        } else if constexpr (std::is_same_v<type_buf, std::reference_wrapper<Buffer>>) {
                            this->enqueue_write_buffer(b.get(), data.get()->data(), blocking);
                        }
                    }
                },
                buffer);
        },
        src);
}

CoreType HWCommandQueue::get_dispatch_core_type() {
    return dispatch_core_manager::get(device->num_hw_cqs()).get_dispatch_core_type(device->id());
}

#if 0 // TODO: Revise this
void HWCommandQueue::enqueue_write_buffer(const Buffer &buffer, const void *src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer");
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Write Buffer cannot be used with tracing");

    uint32_t padded_page_size = align(buffer.page_size(), ADDRESS_ALIGNMENT);

    const uint32_t command_issue_limit = this->manager.get_issue_queue_limit(this->id);
    CoreType dispatch_core_type =
        dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    uint32_t max_data_sizeB =
        max_prefetch_command_size - ((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) * 2);  // * 2 to account for issue

    uint32_t dst_page_index = 0;

    if (is_sharded(buffer.buffer_layout())) {
        const bool width_split = buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1];
        std::optional<BufferPageMapping> buffer_page_mapping = std::nullopt;
        if (width_split) {
            buffer_page_mapping = generate_buffer_page_mapping(buffer);
        }
        const auto &cores = 
            width_split ? 
                buffer_page_mapping.value().all_cores_ : 
                corerange_to_cores(
                    buffer.shard_spec().grid(),
                    buffer.num_cores(),
                    (buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR));
        TT_FATAL(
            max_data_sizeB >= padded_page_size,
            "Writing padded page size > {} is currently unsupported for sharded tensors.",
            max_data_sizeB);
        uint32_t num_total_pages = buffer.num_pages();
        uint32_t max_pages_per_shard = buffer.shard_spec().size();

        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            // Skip writing the padded pages along the bottom
            // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
            // Alternative write each page row into separate commands, or have a strided linear write
            uint32_t num_pages;
            if (width_split) {
                num_pages =
                    buffer_page_mapping.value().core_shard_shape_[core_id][0] * 
                    buffer.shard_spec().shape_in_pages()[1];
                if (num_pages == 0) {
                    continue;
                }
                dst_page_index = 
                    buffer_page_mapping.value().host_page_to_dev_page_mapping_[
                        buffer_page_mapping.value().core_host_page_indices_[core_id][0]];
            } else {
                num_pages = min(num_total_pages, max_pages_per_shard);
                num_total_pages -= num_pages;
            }
            uint32_t curr_page_idx_in_shard = 0;
            uint32_t bank_base_address = buffer.address();
            if (buffer.buffer_type() == BufferType::DRAM) {
                bank_base_address += buffer.device()->bank_offset(
                    BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
            }
            while (num_pages != 0) {
                // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
                uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
                bool issue_wait = dst_page_index == 0;  // only stall for the first write of the buffer
                if (issue_wait) {
                    // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
                    data_offset_bytes *= 2;
                }
                uint32_t space_available_bytes = std::min(
                    command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
                int32_t num_pages_available =
                    (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(padded_page_size);

                uint32_t pages_to_write = std::min(num_pages, (uint32_t)num_pages_available);
                if (pages_to_write > 0) {
                    uint32_t address = bank_base_address + curr_page_idx_in_shard * padded_page_size;

                    tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);

                    auto command = EnqueueWriteShardedBufferCommand(
                        this->id,
                        this->device,
                        this->noc_index,
                        buffer,
                        src,
                        this->manager,
                        issue_wait,
                        this->expected_num_workers_completed,
                        address,
                        buffer_page_mapping,
                        cores[core_id],
                        padded_page_size,
                        dst_page_index,
                        pages_to_write);

                    this->enqueue_command(command, false);
                    curr_page_idx_in_shard += pages_to_write;
                    num_pages -= pages_to_write;
                    dst_page_index += pages_to_write;
                } else {
                    this->manager.wrap_issue_queue_wr_ptr(this->id);
                }
            }
        }
    } else {
        uint32_t total_pages_to_write = buffer.num_pages();
        bool write_partial_pages = padded_page_size > max_data_sizeB;
        uint32_t page_size_to_write = padded_page_size;
        uint32_t padded_buffer_size = buffer.num_pages() * padded_page_size;
        if (write_partial_pages) {
            TT_FATAL(buffer.num_pages() == 1, "TODO: add support for multi-paged buffer with page size > 64KB");
            uint32_t partial_size = dispatch_constants::BASE_PARTIAL_PAGE_SIZE;
            while (padded_buffer_size % partial_size != 0) {
                partial_size += PCIE_ALIGNMENT;
            }
            page_size_to_write = partial_size;
            total_pages_to_write = padded_buffer_size / page_size_to_write;
        }

        const uint32_t num_banks = this->device->num_banks(buffer.buffer_type());
        uint32_t num_pages_round_robined = buffer.num_pages() / num_banks;
        uint32_t num_banks_with_residual_pages = buffer.num_pages() % num_banks;
        uint32_t num_partial_pages_per_page = padded_page_size / page_size_to_write;
        uint32_t num_partials_round_robined = num_partial_pages_per_page * num_pages_round_robined;

        uint32_t max_num_pages_to_write = 
            write_partial_pages ? 
                ((num_pages_round_robined > 0) ? 
                    num_banks * num_partials_round_robined : 
                    num_banks_with_residual_pages) : 
                total_pages_to_write;

        uint32_t bank_base_address = buffer.address();

        uint32_t num_full_pages_written = 0;
        while (total_pages_to_write > 0) {
            // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
            uint32_t data_offsetB = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);
            // only stall for the first write of the buffer
            bool issue_wait = (dst_page_index == 0 && bank_base_address == buffer.address());
            if (issue_wait) {
                data_offsetB *= 2;  // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
            }

            uint32_t space_availableB = std::min(
                command_issue_limit - this->manager.get_issue_queue_write_ptr(this->id), max_prefetch_command_size);
            int32_t num_pages_available =
                (int32_t(space_availableB) - int32_t(data_offsetB)) / int32_t(page_size_to_write);

            if (num_pages_available <= 0) {
                this->manager.wrap_issue_queue_wr_ptr(this->id);
                continue;
            }

            uint32_t num_pages_to_write =
                std::min(std::min((uint32_t)num_pages_available, max_num_pages_to_write), total_pages_to_write);

            // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
            // To handle larger page offsets move bank base address up and update page offset
            // to be relative to the new bank address
            if (dst_page_index > 0xFFFF || 
                    (num_pages_to_write == max_num_pages_to_write && write_partial_pages)) {
                uint32_t num_banks_to_use = write_partial_pages ? max_num_pages_to_write : num_banks;
                uint32_t residual = dst_page_index % num_banks_to_use;
                uint32_t num_pages_written_per_bank = dst_page_index / num_banks_to_use;
                bank_base_address += num_pages_written_per_bank * page_size_to_write;
                dst_page_index = residual;
            }

            tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", this->id);

            auto command = EnqueueWriteInterleavedBufferCommand(
                this->id,
                this->device,
                this->noc_index,
                buffer,
                src,
                this->manager,
                issue_wait,
                this->expected_num_workers_completed,
                bank_base_address,
                page_size_to_write,
                dst_page_index,
                num_pages_to_write);
            this->enqueue_command(
                command, false);  // don't block until the entire src data is enqueued in the issue queue

            total_pages_to_write -= num_pages_to_write;
            dst_page_index += num_pages_to_write;
        }
    }

    if (blocking) {
        this->finish();
    } else {
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
    }
}
#endif

void HWCommandQueue::enqueue_write_buffer(const Buffer &buffer, const void *src, bool blocking) {
    ZoneScopedN("HWCommandQueue_write_buffer\n");
#if 0 // TODO: Revise this
    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Write Buffer cannot be used with tracing");
#endif
    TT_FATAL(
        !this->cq_manager->get_bypass_mode(), 
        "Enqueue Write Buffer cannot be used with tracing");

    uint32_t padded_page_size = align(buffer.page_size(), ADDRESS_ALIGNMENT);

    // [RONIN] No use for "command_issue_limit" (aka "issue_queue_limit")
    CoreType dispatch_core_type =
        dispatch_core_manager::get(this->device->num_hw_cqs()).get_dispatch_core_type(this->device->id());
    const uint32_t max_prefetch_command_size = 
        dispatch_constants::get(dispatch_core_type).max_prefetch_command_size();
    // * 2 to account for issue
    uint32_t max_data_sizeB =
        max_prefetch_command_size - ((sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd)) * 2);

    uint32_t dst_page_index = 0;

    if (is_sharded(buffer.buffer_layout())) {
        const bool width_split = 
            (buffer.shard_spec().shape_in_pages()[1] != buffer.shard_spec().tensor2d_shape[1]);
        std::optional<BufferPageMapping> buffer_page_mapping = std::nullopt;
        if (width_split) {
            buffer_page_mapping = generate_buffer_page_mapping(buffer);
        }
        const auto &cores = 
            width_split ? 
                buffer_page_mapping.value().all_cores_ : 
                corerange_to_cores(
                    buffer.shard_spec().grid(),
                    buffer.num_cores(),
                    (buffer.shard_spec().orientation() == ShardOrientation::ROW_MAJOR));
        TT_FATAL(
            max_data_sizeB >= padded_page_size,
            "Writing padded page size > {} is currently unsupported for sharded tensors.",
            max_data_sizeB);
        uint32_t num_total_pages = buffer.num_pages();
        uint32_t max_pages_per_shard = buffer.shard_spec().size();

        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            // Skip writing the padded pages along the bottom
            // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
            // Alternative write each page row into separate commands, or have a strided linear write
            uint32_t num_pages;
            if (width_split) {
                num_pages =
                    buffer_page_mapping.value().core_shard_shape_[core_id][0] * 
                    buffer.shard_spec().shape_in_pages()[1];
                if (num_pages == 0) {
                    continue;
                }
                dst_page_index = 
                    buffer_page_mapping.value().host_page_to_dev_page_mapping_[
                        buffer_page_mapping.value().core_host_page_indices_[core_id][0]];
            } else {
                num_pages = min(num_total_pages, max_pages_per_shard);
                num_total_pages -= num_pages;
            }
            uint32_t curr_page_idx_in_shard = 0;
            uint32_t bank_base_address = buffer.address();
            if (buffer.buffer_type() == BufferType::DRAM) {
                bank_base_address += buffer.device()->bank_offset(
                    BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
            }
            while (num_pages != 0) {
                // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
                uint32_t data_offset_bytes = (sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd));
                // only stall for the first write of the buffer
                bool issue_wait = (dst_page_index == 0);
                if (issue_wait) {
                    // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
                    data_offset_bytes *= 2;
                }
                uint32_t space_available_bytes = max_prefetch_command_size;
                int32_t num_pages_available =
                    (int32_t(space_available_bytes) - int32_t(data_offset_bytes)) / int32_t(padded_page_size);

                uint32_t pages_to_write = std::min(num_pages, (uint32_t)num_pages_available);
                uint32_t address = bank_base_address + curr_page_idx_in_shard * padded_page_size;

                tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for channel {}", this->id);

                auto command = EnqueueWriteShardedBufferCommand(
                    this->id,
                    this->device,
                    this->noc_index,
                    buffer,
                    src,
#if 0 // TODO: Revise this
                    this->manager,
#endif
                    this->cq_manager,
                    issue_wait,
                    this->expected_num_workers_completed,
                    address,
                    buffer_page_mapping,
                    cores[core_id],
                    padded_page_size,
                    dst_page_index,
                    pages_to_write);

                this->enqueue_command(command, false);
                curr_page_idx_in_shard += pages_to_write;
                num_pages -= pages_to_write;
                dst_page_index += pages_to_write;
            }
        }
    } else {
        uint32_t total_pages_to_write = buffer.num_pages();
        bool write_partial_pages = (padded_page_size > max_data_sizeB);
        uint32_t page_size_to_write = padded_page_size;
        uint32_t padded_buffer_size = buffer.num_pages() * padded_page_size;
        if (write_partial_pages) {
            TT_FATAL(
                buffer.num_pages() == 1, 
                "TODO: add support for multi-paged buffer with page size > 64KB");
            uint32_t partial_size = dispatch_constants::BASE_PARTIAL_PAGE_SIZE;
            while (padded_buffer_size % partial_size != 0) {
                partial_size += PCIE_ALIGNMENT;
            }
            page_size_to_write = partial_size;
            total_pages_to_write = padded_buffer_size / page_size_to_write;
        }

        const uint32_t num_banks = this->device->num_banks(buffer.buffer_type());
        uint32_t num_pages_round_robined = buffer.num_pages() / num_banks;
        uint32_t num_banks_with_residual_pages = buffer.num_pages() % num_banks;
        uint32_t num_partial_pages_per_page = padded_page_size / page_size_to_write;
        uint32_t num_partials_round_robined = num_partial_pages_per_page * num_pages_round_robined;

        uint32_t max_num_pages_to_write = 
            write_partial_pages ? 
                ((num_pages_round_robined > 0) ? 
                    num_banks * num_partials_round_robined : 
                    num_banks_with_residual_pages) : 
                total_pages_to_write;

        uint32_t bank_base_address = buffer.address();

        uint32_t num_full_pages_written = 0;
        while (total_pages_to_write > 0) {
            // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
            uint32_t data_offsetB = sizeof(CQPrefetchCmd) + sizeof(CQDispatchCmd);
            // only stall for the first write of the buffer
            bool issue_wait = (dst_page_index == 0 && bank_base_address == buffer.address());
            if (issue_wait) {
                // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
                data_offsetB *= 2;  
            }

            uint32_t space_availableB = max_prefetch_command_size;
            int32_t num_pages_available =
                (int32_t(space_availableB) - int32_t(data_offsetB)) / int32_t(page_size_to_write);

            uint32_t num_pages_to_write =
                std::min(std::min((uint32_t)num_pages_available, max_num_pages_to_write), total_pages_to_write);

            // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
            // To handle larger page offsets move bank base address up and update page offset
            // to be relative to the new bank address
            if (dst_page_index > 0xFFFF || 
                    (num_pages_to_write == max_num_pages_to_write && write_partial_pages)) {
                uint32_t num_banks_to_use = write_partial_pages ? max_num_pages_to_write : num_banks;
                uint32_t residual = dst_page_index % num_banks_to_use;
                uint32_t num_pages_written_per_bank = dst_page_index / num_banks_to_use;
                bank_base_address += num_pages_written_per_bank * page_size_to_write;
                dst_page_index = residual;
            }

            tt::log_debug(tt::LogDispatch, "EnqueueWriteBuffer for command queue {}", this->id);

            auto command = EnqueueWriteInterleavedBufferCommand(
                this->id,
                this->device,
                this->noc_index,
                buffer,
                src,
#if 0 // TODO: Revise this
                this->manager,
#endif
                this->cq_manager,
                issue_wait,
                this->expected_num_workers_completed,
                bank_base_address,
                page_size_to_write,
                dst_page_index,
                num_pages_to_write);
            // don't block until the entire src data is enqueued in the issue queue
            this->enqueue_command(command, false);  

            total_pages_to_write -= num_pages_to_write;
            dst_page_index += num_pages_to_write;
        }
    }

    if (blocking) {
        this->finish();
#if 0 // TODO: Revise this
    } else {
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
#endif
    }
}

void HWCommandQueue::enqueue_program(Program &program, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");
    if (!program.loaded_onto_device) {
#if 0 // TODO: Revise this
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing should only be used when programs have been cached");
#endif
        TT_FATAL(
            !this->cq_manager->get_bypass_mode(), 
            "Tracing should only be used when programs have been cached");
        TT_ASSERT(program.program_transfer_info.kernel_bins.size() == program.kg_buffers.size());
        for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
            this->enqueue_write_buffer(
                *program.kg_buffers[buffer_idx],
                program.program_transfer_info.kernel_bins[buffer_idx].data.data(),
                false);
        }
    }
#ifdef DEBUG
    if (tt::llrt::OptionsG.get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
            const auto &buffer = program.kg_buffers[buffer_idx];
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            this->enqueue_read_buffer(*buffer, read_data.data(), true);
            TT_FATAL(
                program.program_transfer_info.kernel_bins[buffer_idx].data == read_data,
                "Binary for program to be executed is corrupted. Another program likely corrupted this binary");
        }
    }
#endif

#if 0 // TODO: Revise this
    // Snapshot of expected workers from previous programs, used for dispatch_wait cmd generation.
    uint32_t expected_workers_completed = 
        this->manager.get_bypass_mode() ? 
            this->trace_ctx->num_completion_worker_cores : 
            this->expected_num_workers_completed;
    if (this->manager.get_bypass_mode()) {
        this->trace_ctx->num_completion_worker_cores += program.program_transfer_info.num_active_cores;
    } else {
        this->expected_num_workers_completed += program.program_transfer_info.num_active_cores;
    }
#endif
    // Snapshot of expected workers from previous programs, used for dispatch_wait cmd generation.
    uint32_t expected_workers_completed = 
        this->cq_manager->get_bypass_mode() ? 
            this->trace_ctx->num_completion_worker_cores : 
            this->expected_num_workers_completed;
    if (this->cq_manager->get_bypass_mode()) {
        this->trace_ctx->num_completion_worker_cores += program.program_transfer_info.num_active_cores;
    } else {
        this->expected_num_workers_completed += program.program_transfer_info.num_active_cores;
    }

    auto command = EnqueueProgramCommand(
        this->id, 
        this->device, 
        this->noc_index, 
        program, 
#if 0 // TODO: Revise this
        this->manager, 
#endif
        this->cq_manager, 
        expected_workers_completed);
    this->enqueue_command(command, blocking);

#ifdef DEBUG
    if (tt::llrt::OptionsG.get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        for (int buffer_idx = 0; buffer_idx < program.program_transfer_info.kernel_bins.size(); buffer_idx++) {
            const auto &buffer = program.kg_buffers[buffer_idx];
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            this->enqueue_read_buffer(*buffer, read_data.data(), true);
            TT_FATAL(
                program.program_transfer_info.kernel_bins[buffer_idx].data == read_data,
                "Binary for program that executed is corrupted. This program likely corrupted its own binary.");
        }
    }
#endif

    log_trace(
        tt::LogMetal,
        "Created EnqueueProgramCommand (active_cores: {} bypass_mode: {} expected_workers_completed: {})",
        program.program_transfer_info.num_active_cores,
#if 0 // TODO: Revise this
        this->manager.get_bypass_mode(),
#endif
        this->cq_manager->get_bypass_mode(),
        expected_workers_completed);
}

#if 0 // TODO: Revise this
void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    TT_FATAL(!this->manager.get_bypass_mode(), "Enqueue Record Event cannot be used with tracing");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id;
    event->event_id = this->manager.get_next_event(this->id);
    event->device = this->device;
    event->ready = true;  // what does this mean???

    auto command = EnqueueRecordEventCommand(
        this->id,
        this->device,
        this->noc_index,
        this->manager,
        event->event_id,
        this->expected_num_workers_completed,
        clear_count);
    this->enqueue_command(command, false);

    if (clear_count) {
        this->expected_num_workers_completed = 0;
    }
    this->issued_completion_q_reads.push(detail::ReadEventDescriptor(event->event_id));
#if 0 // TODO: Revise this
    this->increment_num_entries_in_completion_q();
#endif
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> sync_event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    auto command = EnqueueWaitForEventCommand(
        this->id, 
        this->device, 
        this->manager, 
        *sync_event, 
        clear_count);
    this->enqueue_command(command, false);

    if (clear_count) {
        this->manager.reset_event_id(this->id);
    }
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event);
}
#endif

void HWCommandQueue::enqueue_record_event(std::shared_ptr<Event> event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    // [RONIN] Don't enqueue any commands

    TT_FATAL(
        !this->cq_manager->get_bypass_mode(), 
        "Enqueue Record Event cannot be used with tracing");

    // Populate event struct for caller.
    event->cq_id = this->id;
    event->event_id = this->cq_manager->get_next_event(this->id);
    event->device = this->device;
    event->ready = true;  // what does this mean???

    if (clear_count) {
        this->expected_num_workers_completed = 0;
    }
}

void HWCommandQueue::enqueue_wait_for_event(std::shared_ptr<Event> sync_event, bool clear_count) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");

    // [RONIN] Don't enqueue any commands

    if (clear_count) {
        this->cq_manager->reset_event_id(this->id);
    }
}

void HWCommandQueue::enqueue_trace(const uint32_t trace_id, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_trace");

    auto trace_inst = this->device->get_trace(trace_id);
    auto command = EnqueueTraceCommand(
        this->id, 
        this->device, 
#if 0 // TODO: Revise this
        this->manager, 
#endif
        this->cq_manager,
        *trace_inst->buffer, 
        this->expected_num_workers_completed);

    this->enqueue_command(command, false);

    // Increment the exepected worker cores counter due to trace programs completions
    this->expected_num_workers_completed += trace_inst->desc->num_completion_worker_cores;

    if (blocking) {
        this->finish();
#if 0 // TODO: Revise this
    } else {
        std::shared_ptr<Event> event = std::make_shared<Event>();
        this->enqueue_record_event(event);
#endif
    }
}

// [RONIN] SKIPPED: 
//     HWCommandQueue::copy_into_user_space
//     HWCommandQueue::read_completion_queue

#if 0 // TODO: Revise this
void HWCommandQueue::copy_into_user_space(
        const detail::ReadBufferDescriptor &read_buffer_descriptor, 
        chip_id_t mmio_device_id, 
        uint16_t channel) {
    const auto &[
        buffer_layout, 
        page_size, 
        padded_page_size, 
        dev_page_to_host_page_mapping, 
        dst, 
        dst_offset, 
        num_pages_read, 
        cur_dev_page_id] =
            read_buffer_descriptor;

    uint32_t padded_num_bytes = (num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint32_t contig_dst_offset = dst_offset;
    uint32_t remaining_bytes_to_read = padded_num_bytes;
    uint32_t dev_page_id = cur_dev_page_id;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;
    std::optional<uint32_t> host_page_id = std::nullopt;
    uint32_t offset_in_completion_q_data = sizeof(CQDispatchCmd);

    uint32_t pad_size_bytes = padded_page_size - page_size;

    while (remaining_bytes_to_read != 0) {
        this->manager.completion_queue_wait_front(this->id, this->exit_condition);

        if (this->exit_condition) {
            break;
        }

        uint32_t completion_queue_write_ptr_and_toggle =
            get_cq_completion_wr_ptr<true>(this->device->id(), this->id, this->manager.get_cq_size());
        uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        uint32_t completion_q_read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
        uint32_t completion_q_read_toggle = this->manager.get_completion_queue_read_toggle(this->id);

        uint32_t bytes_avail_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr && completion_q_write_toggle == completion_q_read_toggle) {
            bytes_avail_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            bytes_avail_in_completion_queue =
                this->manager.get_completion_queue_limit(this->id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered = std::min(remaining_bytes_to_read, bytes_avail_in_completion_queue);
        uint32_t num_pages_xfered =
            (bytes_xfered + dispatch_constants::TRANSFER_PAGE_SIZE - 1) / dispatch_constants::TRANSFER_PAGE_SIZE;

        remaining_bytes_to_read -= bytes_xfered;

        if (dev_page_to_host_page_mapping.empty()) {
            void *contiguous_dst = (void *)(uint64_t(dst) + contig_dst_offset);
            if ((page_size % ADDRESS_ALIGNMENT) == 0) {
                uint32_t data_bytes_xfered = bytes_xfered - offset_in_completion_q_data;
                tt::Cluster::instance().read_sysmem(
                    contiguous_dst,
                    data_bytes_xfered,
                    completion_q_read_ptr + offset_in_completion_q_data,
                    mmio_device_id,
                    channel);
                contig_dst_offset += data_bytes_xfered;
                offset_in_completion_q_data = 0;
            } else {
                uint32_t src_offset_bytes = offset_in_completion_q_data;
                offset_in_completion_q_data = 0;
                uint32_t dst_offset_bytes = 0;

                while (src_offset_bytes < bytes_xfered) {
                    uint32_t src_offset_increment = padded_page_size;
                    uint32_t num_bytes_to_copy;
                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                        remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                        src_offset_increment = num_bytes_to_copy;
                        // We finished copying the page
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                            // There is more data after padding
                            if (rem_bytes_in_cq >= pad_size_bytes) {
                                src_offset_increment += pad_size_bytes;
                                // Only pad data left in queue
                            } else {
                                offset_in_completion_q_data = pad_size_bytes - rem_bytes_in_cq;
                            }
                        }
                    } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                        // Case 2: Last page of data that was popped off the completion queue
                        // Don't need to compute src_offset_increment since this is end of loop
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                        remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                        // We've copied needed data, start of next read is offset due to remaining pad bytes
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        }
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    tt::Cluster::instance().read_sysmem(
                        (char *)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        num_bytes_to_copy,
                        completion_q_read_ptr + src_offset_bytes,
                        mmio_device_id,
                        channel);

                    src_offset_bytes += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else {
            uint32_t src_offset_bytes = offset_in_completion_q_data;
            offset_in_completion_q_data = 0;
            uint32_t dst_offset_bytes = contig_dst_offset;
            uint32_t num_bytes_to_copy = 0;

            while (src_offset_bytes < bytes_xfered) {
                uint32_t src_offset_increment = padded_page_size;
                if (remaining_bytes_of_nonaligned_page > 0) {
                    // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                    remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                    src_offset_increment = num_bytes_to_copy;
                    // We finished copying the page
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        dev_page_id++;
                        uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                        // There is more data after padding
                        if (rem_bytes_in_cq >= pad_size_bytes) {
                            src_offset_increment += pad_size_bytes;
                            offset_in_completion_q_data = 0;
                            // Only pad data left in queue
                        } else {
                            offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq);
                        }
                    }
                    if (!host_page_id.has_value()) {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                    // Case 2: Last page of data that was popped off the completion queue
                    // Don't need to compute src_offset_increment since this is end of loop
                    host_page_id = dev_page_to_host_page_mapping[dev_page_id];
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                    remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                    // We've copied needed data, start of next read is offset due to remaining pad bytes
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        dev_page_id++;
                    }
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = host_page_id.value() * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else {
                    num_bytes_to_copy = page_size;
                    host_page_id = dev_page_to_host_page_mapping[dev_page_id];
                    dev_page_id++;
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = host_page_id.value() * page_size;
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                }

                tt::Cluster::instance().read_sysmem(
                    (char *)(uint64_t(dst) + dst_offset_bytes),
                    num_bytes_to_copy,
                    completion_q_read_ptr + src_offset_bytes,
                    mmio_device_id,
                    channel);

                src_offset_bytes += src_offset_increment;
            }
            dst_offset_bytes += num_bytes_to_copy;
            contig_dst_offset = dst_offset_bytes;
        }
        this->manager.completion_queue_pop_front(num_pages_xfered, this->id);
    }
}

void HWCommandQueue::read_completion_queue() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    while (true) {
        {
            std::unique_lock<std::mutex> lock(this->reader_thread_cv_mutex);
            this->reader_thread_cv.wait(lock, [this] {
                return (this->num_entries_in_completion_q > this->num_completed_completion_q_reads ||
                    this->exit_condition);
            });
        }
        if (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            ZoneScopedN("CompletionQueueReader");
            uint32_t num_events_to_read = this->num_entries_in_completion_q - this->num_completed_completion_q_reads;
            for (uint32_t i = 0; i < num_events_to_read; i++) {
                ZoneScopedN("CompletionQueuePopulated");
                std::variant<detail::ReadBufferDescriptor, detail::ReadEventDescriptor> read_descriptor =
                    *(this->issued_completion_q_reads.pop());
                {
                    ZoneScopedN("CompletionQueueWait");
                    this->manager.completion_queue_wait_front(
                        this->id, this->exit_condition);  // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN
                }
                if (this->exit_condition) {  // Early exit
                    return;
                }

                std::visit(
                    [&](auto &&read_descriptor) {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, detail::ReadBufferDescriptor>) {
                            ZoneScopedN("CompletionQueueReadData");
                            this->copy_into_user_space(read_descriptor, mmio_device_id, channel);
                        } else if constexpr (std::is_same_v<T, detail::ReadEventDescriptor>) {
                            ZoneScopedN("CompletionQueueReadEvent");
                            uint32_t read_ptr = this->manager.get_completion_queue_read_ptr(this->id);
                            thread_local static std::vector<uint32_t> dispatch_cmd_and_event(
                                (sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE) / sizeof(uint32_t));
                            tt::Cluster::instance().read_sysmem(
                                dispatch_cmd_and_event.data(),
                                sizeof(CQDispatchCmd) + dispatch_constants::EVENT_PADDED_SIZE,
                                read_ptr,
                                mmio_device_id,
                                channel);
                            uint32_t event_completed =
                                dispatch_cmd_and_event.at(sizeof(CQDispatchCmd) / sizeof(uint32_t));
                            TT_ASSERT(
                                event_completed == read_descriptor.event_id,
                                "Event Order Issue: expected to read back completion signal for event {} but got {}!",
                                read_descriptor.event_id,
                                event_completed);
                            this->manager.completion_queue_pop_front(1, this->id);
                            this->manager.set_last_completed_event(this->id, read_descriptor.get_global_event_id());
                            log_trace(
                                LogAlways,
                                "Completion queue popped event {} (global: {})",
                                event_completed,
                                read_descriptor.get_global_event_id());
                        }
                    },
                    read_descriptor);
            }
            this->num_completed_completion_q_reads += num_events_to_read;
        } else if (this->exit_condition) {
            return;
        }
    }
}
#endif

void HWCommandQueue::finish() {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id);
    // [RONIN] Nothing to do
#if 0 // TODO: Revise this
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event);
    if (tt::llrt::OptionsG.get_test_mode_enabled()) {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang. Mark state and early exit. Assert in main thread.
                this->dprint_server_hang = true;
                this->set_exit_condition();
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher. Mark state and early exit. Assert in main thread.
                this->illegal_noc_txn_hang = true;
                this->set_exit_condition();
                return;
            }
        }
    } else {
        while (this->num_entries_in_completion_q > this->num_completed_completion_q_reads);
    }
#endif
}

#if 0 // TODO: Revise this
volatile bool HWCommandQueue::is_dprint_server_hung() { 
    return dprint_server_hang; 
}

volatile bool HWCommandQueue::is_noc_hung() { 
    return illegal_noc_txn_hang; 
}
#endif

void HWCommandQueue::record_begin(const uint32_t tid, std::shared_ptr<detail::TraceDescriptor> ctx) {
    // Issue event as a barrier and a counter reset
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event, true);
    // Record commands using bypass mode
    this->tid = tid;
    this->trace_ctx = ctx;
#if 0 // TODO: Revise this
    this->manager.set_bypass_mode(true, true);  // start
#endif
    this->cq_manager->set_bypass_mode(true);  // start
}

void HWCommandQueue::record_end() {
    this->tid = std::nullopt;
    this->trace_ctx = nullptr;
#if 0 // TODO: Revise this
    this->manager.set_bypass_mode(false, false);  // stop
#endif
    this->cq_manager->set_bypass_mode(false);  // stop
}

#if 0 // TODO: Revise this
void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->manager.get_bypass_mode(), "Terminate cannot be used with tracing");
    tt::log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id);
    auto command = EnqueueTerminateCommand(this->id, this->device, this->manager);
    this->enqueue_command(command, false);
}
#endif

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->cq_manager->get_bypass_mode(), "Terminate cannot be used with tracing");
    tt::log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id);
    auto command = EnqueueTerminateCommand(this->id, this->device, this->cq_manager);
    this->enqueue_command(command, false);
}

//
//    Host API functions mixed with respective Impl functions.
//
//    Impl functions are invoked by "CommandQueue::run_command_impl"
//

void EnqueueAddBufferToProgramImpl(
        const std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        std::variant<
            std::reference_wrapper<Program>, 
            std::shared_ptr<Program>
        > program) {
    std::visit(
        [program](auto &&b) {
            using buffer_type = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<buffer_type, std::shared_ptr<Buffer>>) {
                std::visit(
                    [&b](auto &&p) {
                        using program_type = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<program_type, std::reference_wrapper<Program>>) {
                            p.get().add_buffer(b);
                        } else {
                            p->add_buffer(b);
                        }
                    },
                    program);
            }
        },
        buffer);
}

void EnqueueAddBufferToProgram(
        CommandQueue &cq,
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        std::variant<
            std::reference_wrapper<Program>, 
            std::shared_ptr<Program>
        > program,
        bool blocking) {
    EnqueueAddBufferToProgramImpl(buffer, program);
    // cq.run_command(CommandInterface{
    //     .type = EnqueueCommandType::ADD_BUFFER_TO_PROGRAM,
    //     .blocking = blocking,
    //     .buffer = buffer,
    //     .program = program
    // });
}

void EnqueueSetRuntimeArgsImpl(const RuntimeArgsMetadata &runtime_args_md) {
    std::vector<uint32_t> resolved_runtime_args = {};
    resolved_runtime_args.reserve((*runtime_args_md.runtime_args_ptr).size());

    for (const auto &arg: *(runtime_args_md.runtime_args_ptr)) {
        std::visit(
            [&resolved_runtime_args](auto &&a) {
                using T = std::decay_t<decltype(a)>;
                if constexpr (std::is_same_v<T, Buffer*>) {
                    resolved_runtime_args.push_back(a->address());
                } else {
                    resolved_runtime_args.push_back(a);
                }
            },
            arg);
    }
    runtime_args_md.kernel->set_runtime_args(runtime_args_md.core_coord, resolved_runtime_args);
}

void EnqueueSetRuntimeArgs(
        CommandQueue &cq,
        const std::shared_ptr<Kernel> kernel,
        const CoreCoord &core_coord,
        std::shared_ptr<RuntimeArgs> runtime_args_ptr,
        bool blocking) {
    auto runtime_args_md = RuntimeArgsMetadata{
        .core_coord = core_coord,
        .runtime_args_ptr = runtime_args_ptr,
        .kernel = kernel
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::SET_RUNTIME_ARGS,
        .blocking = blocking,
        .runtime_args_md = runtime_args_md
    });
}

void EnqueueGetBufferAddrImpl(void *dst_buf_addr, const Buffer *buffer) {
    *(static_cast<uint32_t *>(dst_buf_addr)) = buffer->address();
}

void EnqueueGetBufferAddr(
        CommandQueue &cq, 
        uint32_t *dst_buf_addr, 
        const Buffer *buffer, 
        bool blocking) {
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::GET_BUF_ADDR, 
        .blocking = blocking, 
        .shadow_buffer = buffer, 
        .dst = dst_buf_addr
    });
}

void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md) {
    Buffer *buffer = alloc_md.buffer;
    uint32_t allocated_addr;
    if (is_sharded(buffer->buffer_layout())) {
        allocated_addr = allocator::allocate_buffer(
            *(buffer->device()->allocator_),
            buffer->shard_spec().size() * buffer->num_cores() * buffer->page_size(),
            buffer->page_size(),
            buffer->buffer_type(),
            alloc_md.bottom_up,
            buffer->num_cores());
    } else {
        allocated_addr = allocator::allocate_buffer(
            *(buffer->device()->allocator_),
            buffer->size(),
            buffer->page_size(),
            buffer->buffer_type(),
            alloc_md.bottom_up,
            std::nullopt);
    }
    buffer->set_address(static_cast<uint64_t>(allocated_addr));
}

void EnqueueAllocateBuffer(
        CommandQueue &cq, 
        Buffer *buffer, 
        bool bottom_up, 
        bool blocking) {
    auto alloc_md = AllocBufferMetadata{
        .buffer = buffer,
        .allocator = *(buffer->device()->allocator_),
        .bottom_up = bottom_up
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md
    });
}

void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md) {
    allocator::deallocate_buffer(alloc_md.allocator, alloc_md.device_address, alloc_md.buffer_type);
}

void EnqueueDeallocateBuffer(
        CommandQueue &cq, 
        Allocator &allocator, 
        uint32_t device_address, 
        BufferType buffer_type, 
        bool blocking) {
    // Need to explictly pass in relevant buffer attributes here, since the Buffer *ptr can be deallocated a this point
    auto alloc_md = AllocBufferMetadata{
        .allocator = allocator,
        .buffer_type = buffer_type,
        .device_address = device_address
    };
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::DEALLOCATE_BUFFER,
        .blocking = blocking,
        .alloc_md = alloc_md
    });
}

void EnqueueReadBuffer(
        CommandQueue &cq,
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        vector<uint32_t> &dst,
        bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    ZoneScoped;
    tt_metal::detail::DispatchStateCheck(true);
    Buffer &b = 
        std::holds_alternative<std::shared_ptr<Buffer>>(buffer) ? 
            *(std::get<std::shared_ptr<Buffer>>(buffer)) : 
            std::get<std::reference_wrapper<Buffer>>(buffer).get();
    // Only resizing here to keep with the original implementation. Notice how in the void *
    // version of this API, I assume the user mallocs themselves
    std::visit(
        [&dst](auto &&b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>>) {
                dst.resize(b.get().page_size() * b.get().num_pages() / sizeof(uint32_t));
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
                dst.resize(b->page_size() * b->num_pages() / sizeof(uint32_t));
            }
        },
        buffer);

    // TODO(agrebenisan): Move to deprecated
    EnqueueReadBuffer(cq, buffer, dst.data(), blocking);
}

void EnqueueWriteBuffer(
        CommandQueue &cq,
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        vector<uint32_t> &src,
        bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, buffer, src.data(), blocking);
}

void EnqueueReadBuffer(
        CommandQueue &cq,
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        void *dst,
        bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_READ_BUFFER, 
        .blocking = blocking, 
        .buffer = buffer, 
        .dst = dst
    });
}

void EnqueueReadBufferImpl(
        CommandQueue &cq,
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        void *dst,
        bool blocking) {
    std::visit(
        [&cq, dst, blocking](auto &&b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || 
                    std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.hw_command_queue().enqueue_read_buffer(b, dst, blocking);
            }
        },
        buffer);
}

void EnqueueWriteBuffer(
        CommandQueue &cq,
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        HostDataType src,
        bool blocking) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WRITE_BUFFER, 
        .blocking = blocking, 
        .buffer = buffer, 
        .src = src
    });
}

void EnqueueWriteBufferImpl(
        CommandQueue &cq,
        std::variant<
            std::reference_wrapper<Buffer>, 
            std::shared_ptr<Buffer>
        > buffer,
        HostDataType src,
        bool blocking) {
    std::visit(
        [&cq, src, blocking](auto &&b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, std::reference_wrapper<Buffer>> || 
                    std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.hw_command_queue().enqueue_write_buffer(b, src, blocking);
            }
        },
        buffer);
}

void EnqueueProgram(
        CommandQueue &cq, 
        std::variant<
            std::reference_wrapper<Program>, 
            std::shared_ptr<Program>
        > program, 
        bool blocking) {
    detail::DispatchStateCheck(true);
    if (cq.get_mode() != CommandQueue::CommandQueueMode::TRACE) {
        TT_FATAL(cq.id() == 0, "EnqueueProgram only supported on first command queue on device for time being.");
    }
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_PROGRAM, 
        .blocking = blocking, 
        .program = program
    });
}

void EnqueueProgramImpl(
        CommandQueue &cq, 
        std::variant<
            std::reference_wrapper<Program>, 
            std::shared_ptr<Program>
        > program, 
        bool blocking) {
    ZoneScoped;
    std::visit(
        [&cq, blocking](auto &&program) {
            ZoneScoped;
            using T = std::decay_t<decltype(program)>;
            Device *device = cq.device();
            if constexpr (std::is_same_v<T, std::reference_wrapper<Program>>) {
                detail::CompileProgram(device, program);
                program.get().allocate_circular_buffers();
                detail::ValidateCircularBufferRegion(program, device);
                cq.hw_command_queue().enqueue_program(program, blocking);
                // Program relinquishes ownership of all global buffers its using,
                // once its been enqueued. Avoid mem leaks on device.
                program.get().release_buffers();
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Program>>) {
                detail::CompileProgram(device, *program);
                program->allocate_circular_buffers();
                detail::ValidateCircularBufferRegion(*program, device);
                cq.hw_command_queue().enqueue_program(*program, blocking);
                // Program relinquishes ownership of all global buffers its using,
                // once its been enqueued. Avoid mem leaks on device.
                program->release_buffers();
            }
        },
        program);
}

// NOTE: [RONIN] Simplify event handling (all events are completed immediately)

void EnqueueRecordEvent(CommandQueue &cq, std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_RECORD_EVENT,
        .blocking = false,
        .event = event
    });
}

void EnqueueRecordEventImpl(CommandQueue &cq, std::shared_ptr<Event> event) {
    cq.hw_command_queue().enqueue_record_event(event);
}

void EnqueueWaitForEvent(CommandQueue &cq, std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT,
        .blocking = false,
        .event = event
    });
}

void EnqueueWaitForEventImpl(CommandQueue &cq, std::shared_ptr<Event> event) {
    event->wait_until_ready();  // Block until event populated.
    log_trace(
        tt::LogMetal,
        "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(),
        event->cq_id,
        event->event_id,
        cq.device()->id(),
        cq.id());
    cq.hw_command_queue().enqueue_wait_for_event(event);
}

void EventSynchronize(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated.
    log_trace(
        tt::LogMetal,
        "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})",
        event->device->id(),
        event->cq_id,
        event->event_id);

    // [RONIN] All events are completed instantly

#if 0 // TODO: Revise this
    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::llrt::OptionsG.get_test_mode_enabled() && tt::watcher_server_killed_due_to_error()) {
            TT_FATAL(
                false,
                "Command Queue could not complete EventSynchronize. See {} for details.",
                tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
#endif
}

bool EventQuery(std::shared_ptr<Event> event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated.
#if 0 // TODO: Revise this
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(
        tt::LogMetal,
        "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed,
        event->device->id(),
        event->cq_id,
        event->event_id);
    return event_completed;
#endif
    // [RONIN] All events are completed instantly
    return true;
}

void Finish(CommandQueue &cq) {
    detail::DispatchStateCheck(true);
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::FINISH, 
        .blocking = true
    });
#if 0 // TODO: Revise this
    TT_ASSERT(
        !(cq.device()->hw_command_queue(cq.id()).is_dprint_server_hung()),
        "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
    TT_ASSERT(
        !(cq.device()->hw_command_queue(cq.id()).is_noc_hung()),
        "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
        tt::watcher_get_log_file_name());
#endif
}

void FinishImpl(CommandQueue &cq) { 
    cq.hw_command_queue().finish(); 
}

void EnqueueTrace(CommandQueue &cq, uint32_t trace_id, bool blocking) {
    detail::DispatchStateCheck(true);
    TT_FATAL(
        cq.device()->get_trace(trace_id) != nullptr,
        "Trace instance " + std::to_string(trace_id) + " must exist on device");
    cq.run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE, 
        .blocking = blocking, 
        .trace_id = trace_id
    });
}

void EnqueueTraceImpl(CommandQueue &cq, uint32_t trace_id, bool blocking) {
    cq.hw_command_queue().enqueue_trace(trace_id, blocking);
}

//
//    CommandQueue
//

//
//    NOTE: [RONIN] Only passthrough_mode will be supported.
//        This means no worker.
//        Also, fields related to other modes must be identified and removed.
//

#if 0 // TODO: Revise this
CommandQueue::CommandQueue(Device *device, uint32_t id, CommandQueueMode mode):
        device_ptr(device), 
        cq_id(id), 
        mode(mode), 
        worker_state(CommandQueueState::IDLE) {
    // NOTE: [RONIN] Support only passthrough mode
    if (this->async_mode()) {
        num_async_cqs++;
        // The main program thread launches the Command Queue
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        this->start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
    }
}

CommandQueue::CommandQueue(Trace &trace):
        device_ptr(nullptr),
        parent_thread_id(0),
        cq_id(-1),
        mode(CommandQueueMode::TRACE),
        worker_state(CommandQueueState::IDLE) { }

CommandQueue::~CommandQueue() {
    // NOTE: [RONIN] This must be void function (mode is fixed to passthrough)
    if (this->async_mode()) {
        this->stop_worker();
    }
    if (!this->trace_mode()) {
        TT_FATAL(this->worker_queue.empty(), "{} worker queue must be empty on destruction", this->name());
    }
}
#endif

#if 0 // TODO: Revise this
CommandQueue::CommandQueue(Device *device, uint32_t id, CommandQueueMode mode):
        device_ptr(device), 
        cq_id(id), 
        mode(mode), 
        worker_state(CommandQueueState::IDLE) { }
#endif

CommandQueue::CommandQueue(Device *device, uint32_t id, CommandQueueMode mode):
        device_ptr(device), 
        cq_id(id), 
        mode(mode) { }

CommandQueue::~CommandQueue() { }

HWCommandQueue &CommandQueue::hw_command_queue() { 
    return this->device()->hw_command_queue(this->cq_id); 
}

#if 0 // TODO: Revise this
void CommandQueue::dump() {
    int cid = 0;
    log_info(LogMetalTrace, "Dumping {}, mode={}", this->name(), this->get_mode());
    for (const auto &cmd: this->worker_queue) {
        log_info(LogMetalTrace, "[{}]: {}", cid, cmd.type);
        cid++;
    }
}
#endif

std::string CommandQueue::name() {
#if 0 // TODO: Revise this
    if (this->mode == CommandQueueMode::TRACE) {
        return "TraceQueue";
    }
#endif
    return "CQ" + std::to_string(this->cq_id);
}

// [RONIN] SKIP:
//     CommandQueue::wait_until_empty
//     CommandQueue::start_worker
//     CommandQueue::stop_worker
//     CommandQueue::run_worker

#if 0 // TODO: Revise this
void CommandQueue::wait_until_empty() {
    // NOTE: [RONIN] This must be void function (worker queue always empty and not used at all?)
    log_trace(LogDispatch, "{} WFI start", this->name());
    if (this->async_mode()) {
        // Insert a flush token to push all prior commands to completion
        // Necessary to avoid implementing a peek and pop on the lock-free queue
        this->worker_queue.push(CommandInterface{.type = EnqueueCommandType::FLUSH});
    }
    while (true) {
        if (this->worker_queue.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    log_trace(LogDispatch, "{} WFI complete", this->name());
}

void CommandQueue::set_mode(const CommandQueueMode &mode) {
    // NOTE [RONIN]: This must be void function (mode is fixed to passthrough)
    TT_ASSERT(
        !this->trace_mode(),
        "Cannot change mode of a trace command queue, copy to a non-trace command queue instead!");
    if (this->mode == mode) {
        // Do nothing if requested mode matches current CQ mode.
        return;
    }
    this->mode = mode;
    if (this->async_mode()) {
        num_async_cqs++;
        num_passthrough_cqs--;
        // Record parent thread-id and start worker.
        parent_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        start_worker();
    } else if (this->passthrough_mode()) {
        num_passthrough_cqs++;
        num_async_cqs--;
        // Wait for all cmds sent in async mode to complete and stop worker.
        this->wait_until_empty();
        this->stop_worker();
    }
}

void CommandQueue::start_worker() {
    // NOTE: [RONIN] Delete this function (no worker in passthrough mode)
    if (this->worker_state == CommandQueueState::RUNNING) {
        return;  // worker already running, exit
    }
    this->worker_state = CommandQueueState::RUNNING;
    this->worker_thread = std::make_unique<std::thread>(std::thread(&CommandQueue::run_worker, this));
    tt::log_debug(tt::LogDispatch, "{} started worker thread", this->name());
}

void CommandQueue::stop_worker() {
    // NOTE: [RONIN] Delete this function (no worker in passthrough mode)
    if (this->worker_state == CommandQueueState::IDLE) {
        return;  // worker already stopped, exit
    }
    this->worker_state = CommandQueueState::TERMINATE;
    this->worker_thread->join();
    this->worker_state = CommandQueueState::IDLE;
    tt::log_debug(tt::LogDispatch, "{} stopped worker thread", this->name());
}

void CommandQueue::run_worker() {
    // NOTE: [RONIN] Delete this function (no worker in passthrough mode)
    // forever loop checking for commands in the worker queue
    // Track the worker thread id, for cases where a command calls a sub command.
    // This is to detect cases where commands may be nested.
    worker_thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    while (true) {
        if (this->worker_queue.empty()) {
            if (this->worker_state == CommandQueueState::TERMINATE) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else {
            std::shared_ptr<CommandInterface> command(this->worker_queue.pop());
            run_command_impl(*command);
        }
    }
}

void CommandQueue::run_command(const CommandInterface &command) {
    // NOTE: [RONIN] Support only passthrough mode; remove anything else
    log_trace(LogDispatch, "{} received {} in {} mode", this->name(), command.type, this->mode);
    if (this->async_mode()) {
        if (std::hash<std::thread::id>{}(std::this_thread::get_id()) == parent_thread_id) {
            // In async mode when parent pushes cmd, feed worker through queue.
            this->worker_queue.push(command);
            bool blocking = (command.blocking.has_value() && *command.blocking);
            if (blocking) {
                TT_ASSERT(!this->trace_mode(), "Blocking commands cannot be traced!");
                this->wait_until_empty();
            }
        } else {
            // Handle case where worker pushes command to itself (passthrough)
            TT_ASSERT(
                std::hash<std::thread::id>{}(std::this_thread::get_id()) == worker_thread_id,
                "Only main thread or worker thread can run commands through the SW command queue");
            run_command_impl(command);
        }
    } else if (this->trace_mode()) {
        // In trace mode push to the trace queue
        this->worker_queue.push(command);
    } else if (this->passthrough_mode()) {
        this->run_command_impl(command);
    } else {
        TT_THROW("Unsupported CommandQueue mode!");
    }
}
#endif

void CommandQueue::set_mode(const CommandQueueMode &mode) {
    // [RONIN] Ignored
}

void CommandQueue::run_command(const CommandInterface &command) {
    // [RONIN] Support only passthrough mode
    log_trace(LogDispatch, "{} received {} in {} mode", this->name(), command.type, this->mode);
    this->run_command_impl(command);
}

void CommandQueue::run_command_impl(const CommandInterface &command) {
    log_trace(LogDispatch, "{} running {}", this->name(), command.type);
    switch (command.type) {
    case EnqueueCommandType::ENQUEUE_READ_BUFFER:
        TT_ASSERT(command.dst.has_value(), "Must provide a dst!");
        TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
        TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
        EnqueueReadBufferImpl(*this, command.buffer.value(), command.dst.value(), command.blocking.value());
        break;
    case EnqueueCommandType::ENQUEUE_WRITE_BUFFER:
        TT_ASSERT(command.src.has_value(), "Must provide a src!");
        TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
        TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
        EnqueueWriteBufferImpl(*this, command.buffer.value(), command.src.value(), command.blocking.value());
        break;
    case EnqueueCommandType::ALLOCATE_BUFFER:
        TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
        EnqueueAllocateBufferImpl(command.alloc_md.value());
        break;
    case EnqueueCommandType::DEALLOCATE_BUFFER:
        TT_ASSERT(command.alloc_md.has_value(), "Must provide buffer allocation metdata!");
        EnqueueDeallocateBufferImpl(command.alloc_md.value());
        break;
    case EnqueueCommandType::GET_BUF_ADDR:
        TT_ASSERT(command.dst.has_value(), "Must provide a dst address!");
        TT_ASSERT(command.shadow_buffer.has_value(), "Must provide a shadow buffer!");
        EnqueueGetBufferAddrImpl(command.dst.value(), command.shadow_buffer.value());
        break;
    case EnqueueCommandType::SET_RUNTIME_ARGS:
        TT_ASSERT(command.runtime_args_md.has_value(), "Must provide RuntimeArgs Metdata!");
        EnqueueSetRuntimeArgsImpl(command.runtime_args_md.value());
        break;
    case EnqueueCommandType::ADD_BUFFER_TO_PROGRAM:
        TT_ASSERT(command.buffer.has_value(), "Must provide a buffer!");
        TT_ASSERT(command.program.has_value(), "Must provide a program!");
        EnqueueAddBufferToProgramImpl(command.buffer.value(), command.program.value());
        break;
    case EnqueueCommandType::ENQUEUE_PROGRAM:
        TT_ASSERT(command.program.has_value(), "Must provide a program!");
        TT_ASSERT(command.blocking.has_value(), "Must specify blocking value!");
        EnqueueProgramImpl(*this, command.program.value(), command.blocking.value());
        break;
    case EnqueueCommandType::ENQUEUE_TRACE:
        EnqueueTraceImpl(*this, command.trace_id.value(), command.blocking.value());
        break;
    case EnqueueCommandType::ENQUEUE_RECORD_EVENT:
        TT_ASSERT(command.event.has_value(), "Must provide an event!");
        EnqueueRecordEventImpl(*this, command.event.value());
        break;
    case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT:
        TT_ASSERT(command.event.has_value(), "Must provide an event!");
        EnqueueWaitForEventImpl(*this, command.event.value());
        break;
    case EnqueueCommandType::FINISH: 
        FinishImpl(*this); 
        break;
    case EnqueueCommandType::FLUSH:
        // Used by CQ to push prior commands
        break;
    default: 
        TT_THROW("Invalid command type");
    }
    log_trace(LogDispatch, "{} running {} complete", this->name(), command.type);
}

}  // namespace tt::tt_metal

//
//    I/O operators
//

std::ostream &operator<<(std::ostream &os, EnqueueCommandType const &type) {
    switch (type) {
    case EnqueueCommandType::ENQUEUE_READ_BUFFER: 
        os << "ENQUEUE_READ_BUFFER"; 
        break;
    case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: 
        os << "ENQUEUE_WRITE_BUFFER"; 
        break;
    case EnqueueCommandType::ENQUEUE_PROGRAM: 
        os << "ENQUEUE_PROGRAM"; 
        break;
    case EnqueueCommandType::ENQUEUE_TRACE: 
        os << "ENQUEUE_TRACE"; 
        break;
    case EnqueueCommandType::ENQUEUE_RECORD_EVENT: 
        os << "ENQUEUE_RECORD_EVENT"; 
        break;
    case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: 
        os << "ENQUEUE_WAIT_FOR_EVENT"; 
        break;
    case EnqueueCommandType::FINISH: 
        os << "FINISH"; 
        break;
    case EnqueueCommandType::FLUSH:
        os << "FLUSH"; 
        break;
    default: 
        TT_THROW("Invalid command type!");
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, CommandQueue::CommandQueueMode const &type) {
    switch (type) {
    case CommandQueue::CommandQueueMode::PASSTHROUGH: 
        os << "PASSTHROUGH"; 
        break;
    case CommandQueue::CommandQueueMode::ASYNC: 
        os << "ASYNC"; 
        break;
    case CommandQueue::CommandQueueMode::TRACE: 
        os << "TRACE"; 
        break;
    default: 
        TT_THROW("Invalid CommandQueueMode type!");
    }
    return os;
}

