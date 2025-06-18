// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <utility>

#include "common/env_lib.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/dispatch/command_queue_manager.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/trace/trace_buffer.hpp"

namespace tt::tt_metal {

class Event;
class Trace;
using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType {
    ENQUEUE_READ_BUFFER,
    ENQUEUE_WRITE_BUFFER,
    ALLOCATE_BUFFER,
    DEALLOCATE_BUFFER,
    GET_BUF_ADDR,
    ADD_BUFFER_TO_PROGRAM,
    SET_RUNTIME_ARGS,
    ENQUEUE_PROGRAM,
    ENQUEUE_TRACE,
    ENQUEUE_RECORD_EVENT,
    ENQUEUE_WAIT_FOR_EVENT,
    FINISH,
    FLUSH,
    TERMINATE,
    INVALID
};

class CommandQueue;
class CommandInterface;

class Command {
   public:
    Command() {}
    virtual void process() {};
    virtual EnqueueCommandType type() = 0;
};

class EnqueueReadBufferCommand : public Command {
private:
    CQManager *cq_manager;
    void* dst;
    CoreType dispatch_core_type;

    virtual void add_prefetch_relay(HugepageDeviceCommand& command) = 0;

protected:
    Device* device;
    uint32_t command_queue_id;
    NOC noc_index;
    uint32_t expected_num_workers_completed;
    uint32_t src_page_index;
    uint32_t pages_to_read;

public:
    Buffer& buffer;
    EnqueueReadBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        NOC noc_index,
        Buffer& buffer,
        void* dst,
        CQManager *cq_manager,
        uint32_t expected_num_workers_completed,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_READ_BUFFER; };

    constexpr bool has_side_effects() { return false; }
};

class EnqueueReadInterleavedBufferCommand : public EnqueueReadBufferCommand {
private:
    void add_prefetch_relay(HugepageDeviceCommand& command) override;

public:
    EnqueueReadInterleavedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        NOC noc_index,
        Buffer& buffer,
        void* dst,
        CQManager *cq_manager,
        uint32_t expected_num_workers_completed,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt) :
        EnqueueReadBufferCommand(
            command_queue_id,
            device,
            noc_index,
            buffer,
            dst,
            cq_manager,
            expected_num_workers_completed,
            src_page_index,
            pages_to_read) {}
};

class EnqueueReadShardedBufferCommand : public EnqueueReadBufferCommand {
private:
    void add_prefetch_relay(HugepageDeviceCommand& command) override;
    const CoreCoord core;
    const uint32_t bank_base_address;

public:
    EnqueueReadShardedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        NOC noc_index,
        Buffer& buffer,
        void* dst,
        CQManager *cq_manager,
        uint32_t expected_num_workers_completed,
        const CoreCoord& core,
        uint32_t bank_base_address,
        uint32_t src_page_index = 0,
        std::optional<uint32_t> pages_to_read = std::nullopt) :
        EnqueueReadBufferCommand(
            command_queue_id,
            device,
            noc_index,
            buffer,
            dst,
            cq_manager,
            expected_num_workers_completed,
            src_page_index,
            pages_to_read),
        core(core),
        bank_base_address(bank_base_address) {}
};

class EnqueueWriteShardedBufferCommand;
class EnqueueWriteInterleavedBufferCommand;

class EnqueueWriteBufferCommand : public Command {
private:
    CQManager *cq_manager;
    CoreType dispatch_core_type;

    virtual void add_dispatch_write(HugepageDeviceCommand& command) = 0;
    virtual void add_buffer_data(HugepageDeviceCommand& command) = 0;

protected:
    Device* device;
    uint32_t command_queue_id;
    NOC noc_index;
    const void* src;
    const Buffer& buffer;
    uint32_t expected_num_workers_completed;
    uint32_t bank_base_address;
    uint32_t padded_page_size;
    uint32_t dst_page_index;
    uint32_t pages_to_write;
    bool issue_wait;

public:
    EnqueueWriteBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        NOC noc_index,
        const Buffer& buffer,
        const void* src,
        CQManager *cq_manager,
        bool issue_wait,
        uint32_t expected_num_workers_completed,
        uint32_t bank_base_address,
        uint32_t padded_page_size,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_WRITE_BUFFER; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueWriteInterleavedBufferCommand : public EnqueueWriteBufferCommand {
private:
    void add_dispatch_write(HugepageDeviceCommand& command) override;
    void add_buffer_data(HugepageDeviceCommand& command) override;

public:
    EnqueueWriteInterleavedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        NOC noc_index,
        const Buffer& buffer,
        const void* src,
        CQManager *cq_manager,
        bool issue_wait,
        uint32_t expected_num_workers_completed,
        uint32_t bank_base_address,
        uint32_t padded_page_size,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt) :
        EnqueueWriteBufferCommand(
            command_queue_id,
            device,
            noc_index,
            buffer,
            src,
            cq_manager,
            issue_wait,
            expected_num_workers_completed,
            bank_base_address,
            padded_page_size,
            dst_page_index,
            pages_to_write) {
        ;
    }
};

class EnqueueWriteShardedBufferCommand : public EnqueueWriteBufferCommand {
private:
    void add_dispatch_write(HugepageDeviceCommand& command) override;
    void add_buffer_data(HugepageDeviceCommand& command) override;

    const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping;
    const CoreCoord core;

public:
    EnqueueWriteShardedBufferCommand(
        uint32_t command_queue_id,
        Device* device,
        NOC noc_index,
        const Buffer& buffer,
        const void* src,
        CQManager *cq_manager,
        bool issue_wait,
        uint32_t expected_num_workers_completed,
        uint32_t bank_base_address,
        const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping,
        const CoreCoord& core,
        uint32_t padded_page_size,
        uint32_t dst_page_index = 0,
        std::optional<uint32_t> pages_to_write = std::nullopt) :
        EnqueueWriteBufferCommand(
            command_queue_id,
            device,
            noc_index,
            buffer,
            src,
            cq_manager,
            issue_wait,
            expected_num_workers_completed,
            bank_base_address,
            padded_page_size,
            dst_page_index,
            pages_to_write),
        buffer_page_mapping(buffer_page_mapping),
        core(core) {
        ;
    }
};

class EnqueueProgramCommand : public Command {
private:
    uint32_t command_queue_id;
    Device* device;
    NOC noc_index;
    Program& program;
    CQManager *cq_manager;
    CoreCoord dispatch_core;
    CoreType dispatch_core_type;
    uint32_t expected_num_workers_completed;
    uint32_t packed_write_max_unicast_sub_cmds;

public:
    struct CachedProgramCommandSequence {
        HostMemDeviceCommand preamble_command_sequence;
        HostMemDeviceCommand stall_command_sequence;
        std::vector<HostMemDeviceCommand> runtime_args_command_sequences;
        uint32_t runtime_args_fetch_size_bytes;
        HostMemDeviceCommand program_command_sequence;
        std::vector<uint32_t*> cb_configs_payloads;
        std::vector<std::vector<std::shared_ptr<CircularBuffer>>> circular_buffers_on_core_ranges;
        std::vector<launch_msg_t*> go_signals;
        uint32_t program_config_buffer_data_size_bytes;
    };
    static std::unordered_map<uint64_t, CachedProgramCommandSequence> cached_program_command_sequences;

    EnqueueProgramCommand(
        uint32_t command_queue_id,
        Device* device,
        NOC noc_index,
        Program& program,
        CQManager *cq_manager,
        uint32_t expected_num_workers_completed);

    void assemble_preamble_commands(std::vector<ConfigBufferEntry>& kernel_config_addrs);
    void assemble_stall_commands(bool prefetch_stall);
    void assemble_device_commands(bool is_cached, std::vector<ConfigBufferEntry>& kernel_config_addrs);
    void assemble_runtime_args_commands();

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_PROGRAM; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueTraceCommand : public Command {
private:
    uint32_t command_queue_id;
    Buffer& buffer;
    Device* device;
    CQManager *cq_manager;
    uint32_t& expected_num_workers_completed;
    bool clear_count;

public:
    EnqueueTraceCommand(
        uint32_t command_queue_id,
        Device* device,
        CQManager *cq_manager,
        Buffer& buffer,
        uint32_t& expected_num_workers_completed);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_TRACE; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueTerminateCommand : public Command {
private:
    uint32_t command_queue_id;
    Device* device;
    CQManager *cq_manager;

public:
    EnqueueTerminateCommand(
        uint32_t command_queue_id, 
        Device* device, 
        CQManager *cq_manager);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::TERMINATE; }

    constexpr bool has_side_effects() { return false; }
};

namespace detail {

inline bool LAZY_COMMAND_QUEUE_MODE = false;

/*
 Used so the host knows how to properly copy data into user space from the completion queue (in hugepages)
*/
struct ReadBufferDescriptor {
    TensorMemoryLayout buffer_layout;
    uint32_t page_size;
    uint32_t padded_page_size;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping;
    void* dst;
    uint32_t dst_offset;
    uint32_t num_pages_read;
    uint32_t cur_dev_page_id;

    ReadBufferDescriptor(
        TensorMemoryLayout buffer_layout,
        uint32_t page_size,
        uint32_t padded_page_size,
        void* dst,
        uint32_t dst_offset,
        uint32_t num_pages_read,
        uint32_t cur_dev_page_id,
        const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping = nullptr) :
        buffer_layout(buffer_layout),
        page_size(page_size),
        padded_page_size(padded_page_size),
        buffer_page_mapping(buffer_page_mapping),
        dst(dst),
        dst_offset(dst_offset),
        num_pages_read(num_pages_read),
        cur_dev_page_id(cur_dev_page_id) {}
};

/*
 Used so host knows data in completion queue is just an event ID
*/
struct ReadEventDescriptor {
    uint32_t event_id;
    uint32_t global_offset;

    explicit ReadEventDescriptor(uint32_t event) : event_id(event), global_offset(0) {}

    void set_global_offset(uint32_t offset) { global_offset = offset; }
    uint32_t get_global_event_id() { return global_offset + event_id; }
};

}  // namespace detail

struct AllocBufferMetadata {
    Buffer* buffer;
    std::reference_wrapper<Allocator> allocator;
    BufferType buffer_type;
    uint32_t device_address;
    bool bottom_up;
};

struct RuntimeArgsMetadata {
    CoreCoord core_coord;
    std::shared_ptr<RuntimeArgs> runtime_args_ptr;
    std::shared_ptr<Kernel> kernel;
    std::vector<uint32_t> update_idx;
};

class HWCommandQueue {
public:
    HWCommandQueue(Device* device, uint32_t id, NOC noc_index);

    ~HWCommandQueue();

    NOC noc_index;

    void record_begin(const uint32_t tid, std::shared_ptr<detail::TraceDescriptor> ctx);
    void record_end();
private:
    uint32_t id;
    uint32_t size_B;
    std::optional<uint32_t> tid;
    std::shared_ptr<detail::TraceDescriptor> trace_ctx;
    CQManager *cq_manager;

    // Expected value of DISPATCH_MESSAGE_ADDR in dispatch core L1
    //  Value in L1 incremented by worker to signal completion to dispatch. Value on host is set on each enqueue program
    //  call
    uint32_t expected_num_workers_completed;

    Device* device;

    CoreType get_dispatch_core_type();

    template <typename T>
    void enqueue_command(T& command, bool blocking);

    void enqueue_read_buffer(std::shared_ptr<Buffer>& buffer, void* dst, bool blocking);
    void enqueue_read_buffer(Buffer& buffer, void* dst, bool blocking);
    void enqueue_write_buffer(
        std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, HostDataType src, bool blocking);
    void enqueue_write_buffer(Buffer& buffer, const void* src, bool blocking);
    void enqueue_program(Program& program, bool blocking);
    void enqueue_record_event(const std::shared_ptr<Event>& event, bool clear_count = false);
    void enqueue_wait_for_event(const std::shared_ptr<Event>& sync_event, bool clear_count = false);
    void enqueue_trace(const uint32_t trace_id, bool blocking);
    void finish();
    void terminate();

    friend void EnqueueTraceImpl(CommandQueue& cq, uint32_t trace_id, bool blocking);
    friend void EnqueueProgramImpl(
        CommandQueue& cq,
        Program& program,
        bool blocking);
    friend void EnqueueReadBufferImpl(
        CommandQueue& cq,
        std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
        void* dst,
        bool blocking);
    friend void EnqueueWriteBufferImpl(
        CommandQueue& cq,
        std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
        HostDataType src,
        bool blocking);
    friend void EnqueueAllocateBufferImpl(AllocBufferMetadata alloc_md);
    friend void EnqueueDeallocateBufferImpl(AllocBufferMetadata alloc_md);
    friend void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer);
    friend void EnqueueRecordEventImpl(CommandQueue& cq, const std::shared_ptr<Event>& event);
    friend void EnqueueWaitForEventImpl(CommandQueue& cq, const std::shared_ptr<Event>& event);
    friend void FinishImpl(CommandQueue& cq);
    friend void EnqueueRecordEvent(CommandQueue& cq, const std::shared_ptr<Event>& event);
    friend class CommandQueue;
    friend class Device;
};

// Common interface for all command queue types
struct CommandInterface {
    EnqueueCommandType type;
    std::optional<bool> blocking;
    std::optional<std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>> buffer;
    Program* program;
    std::optional<AllocBufferMetadata> alloc_md;
    std::optional<RuntimeArgsMetadata> runtime_args_md;
    std::optional<const Buffer*> shadow_buffer;
    std::optional<HostDataType> src;
    std::optional<void*> dst;
    std::optional<std::shared_ptr<Event>> event;
    std::optional<uint32_t> trace_id;
};

class CommandQueue {
    friend class Device;
    friend class Trace;

   public:
    enum class CommandQueueMode {
        PASSTHROUGH = 0,
        ASYNC = 1,
        TRACE = 2,
    };

    CommandQueue() = delete;
    ~CommandQueue();

    // Getters for private members
    Device* device() const { return this->device_ptr; }
    uint32_t id() const { return this->cq_id; }

    // Blocking method to wait for all commands to drain from the queue
    // Optional if used in passthrough mode (async_mode = false)
    void wait_until_empty();

    // Schedule a command to be run on the device
    // Blocking if in passthrough mode. Non-blocking if in async mode
    void run_command(const CommandInterface& command);

    // API for setting/getting the mode of the command queue
    void set_mode(const CommandQueueMode& mode);
    CommandQueueMode get_mode() const { return this->mode; }

    // Reference to the underlying hardware command queue, non-const because side-effects are allowed
    HWCommandQueue& hw_command_queue();

    // The empty state of the worker queue
    bool empty() const { 
        // deprecated
        return true; 
    }

    // Dump methods for name and pending commands in the queue
    void dump();
    std::string name();

    static CommandQueueMode default_mode() {
        return CommandQueueMode::PASSTHROUGH;
    }

    // Determine if any CQ is using Async mode
    static bool async_mode_set() { 
        // always passthough mode
        return false; 
    }

private:
    // Command queue constructor
    CommandQueue(Device* device, uint32_t id, CommandQueueMode mode = CommandQueue::default_mode());

    // Initialize Command Queue Mode based on the env-var. This will be default, unless the user excplictly sets the
    // mode using set_mode.
    CommandQueueMode mode;
    uint32_t cq_id;
    Device* device_ptr;

    void run_command_impl(const CommandInterface& command);
};

// Primitives used to place host only operations on the SW Command Queue.
// These are used in functions exposed through tt_metal.hpp or host_api.hpp
void EnqueueAllocateBuffer(CommandQueue& cq, Buffer* buffer, bool bottom_up, bool blocking);
void EnqueueDeallocateBuffer(
    CommandQueue& cq, Allocator& allocator, uint32_t device_address, BufferType buffer_type, bool blocking);
void EnqueueGetBufferAddr(CommandQueue& cq, uint32_t* dst_buf_addr, const Buffer* buffer, bool blocking);
void EnqueueSetRuntimeArgs(
    CommandQueue& cq,
    const std::shared_ptr<Kernel> kernel,
    const CoreCoord& core_coord,
    std::shared_ptr<RuntimeArgs> runtime_args_ptr,
    bool blocking);
void EnqueueAddBufferToProgram(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    Program& program,
    bool blocking);

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, EnqueueCommandType const& type);
std::ostream& operator<<(std::ostream& os, CommandQueue::CommandQueueMode const& type);
