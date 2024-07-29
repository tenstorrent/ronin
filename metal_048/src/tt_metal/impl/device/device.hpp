// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
//#include <mutex> // [RONIN]

#include "hostdevcommon/common_values.hpp"
#include "impl/dispatch/work_executor.hpp"
#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/jit_build/build.hpp"
#include "llrt/tt_cluster.hpp"
#include "dev_msgs.h"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/dispatch/command_queue_manager.hpp" // [RONIN]
#include "program_cache.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;
class Buffer;
class Program;
class JitBuildEnv;
class HWCommandQueue;
class CommandQueue;

namespace detail {
// TODO(agrebenisan): Need device to hold onto command queue programs,
// but the Program type is incomplete by this point. I can have
// a unique_ptr of incomplete type as long as I override the default
// delete function.
struct ProgramDeleter {
    void operator()(Program* p);
};

class TraceDescriptor;

}

using on_close_device_callback = std::function<void ()>;

static constexpr float  EPS_GS = 0.001953125f;
static constexpr float  EPS_WHB0 = 1.19209e-7f;

class ActiveDevices {
    enum class ActiveState {
        UNINITIALIZED = 0,
        INACTIVE = 1,
        ACTIVE = 2,
    };

#if 0 // [RONIN]
    std::mutex lock_;
#endif
    std::vector<enum ActiveState>active_devices_;

public:
    ActiveDevices();
    ~ActiveDevices();

    bool activate_device(chip_id_t id);
    void deactivate_device(chip_id_t id);
    bool is_device_active(chip_id_t id);
};

// A physical PCIexpress Tenstorrent device
class Device {
   public:
    // friend void tt_gdb(Device* device, int chip_id, const vector<CoreCoord> cores, vector<string> ops);
    Device () = delete;
    Device(
        chip_id_t device_id,
        const uint8_t num_hw_cqs,
        std::size_t l1_small_size,
        const std::vector<uint32_t> &l1_bank_remap = {},
        bool minimal = false,
        uint32_t worker_core = 0);

    ~Device();

    // TODO: Add copy/move semantics
    Device(const Device &other) = delete;
    Device& operator=(const Device &other) = delete;

    Device(Device &&other) = default;
    Device& operator=(Device &&other) = default;

    tt::ARCH arch() const;

    chip_id_t id() const { return id_; }

    uint32_t build_key() const { return build_key_; }

    uint8_t num_hw_cqs() const { return num_hw_cqs_; }

    bool is_initialized() const { return this->initialized_; }

    int num_dram_channels() const;

    uint32_t l1_size_per_core() const;
    uint32_t dram_size_per_channel() const;

    CoreCoord logical_grid_size() const;

    CoreCoord compute_with_storage_grid_size() const;

    CoreCoord dram_grid_size() const;

    CoreCoord physical_core_from_logical_core(const CoreCoord &logical_core, const CoreType &core_type) const;
    CoreType core_type_from_physical_core(const CoreCoord &physical_core) const;

    CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const;
    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;

    CoreCoord dram_core_from_logical_core(const CoreCoord &logical_core) const;
    std::vector<CoreCoord> dram_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;

    // Ethernet API
    CoreCoord ethernet_core_from_logical_core(const CoreCoord &logical_core) const;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord &physical_core) const;

    std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;

    std::unordered_set<chip_id_t> get_ethernet_connected_device_ids() const {
        return tt::Cluster::instance().get_ethernet_connected_device_ids(this->id_);
    }

    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores=false) const {
        return tt::Cluster::instance().get_active_ethernet_cores(this->id_, skip_reserved_tunnel_cores);
    }

    bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores=false) const {
        auto active_ethernet_cores = tt::Cluster::instance().get_active_ethernet_cores(this->id_, skip_reserved_tunnel_cores);
        return active_ethernet_cores.find(logical_core) != active_ethernet_cores.end();
    }

    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const {
        return tt::Cluster::instance().get_inactive_ethernet_cores(this->id_);
    }

    bool is_inactive_ethernet_core(CoreCoord logical_core) const {
        auto inactive_ethernet_cores = tt::Cluster::instance().get_inactive_ethernet_cores(this->id_);
        return inactive_ethernet_cores.find(logical_core) != inactive_ethernet_cores.end();
    }

    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const {
        return tt::Cluster::instance().get_connected_ethernet_core(std::make_tuple(this->id_, eth_core));
    }

    std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const {
        return tt::Cluster::instance().get_ethernet_sockets(this->id_, connected_chip_id);
    }

    bool is_mmio_capable() const {
        return tt::Cluster::instance().get_associated_mmio_device(this->id_) == this->id_;
    }

    uint32_t num_banks(const BufferType &buffer_type) const;
    uint32_t bank_size(const BufferType &buffer_type) const;

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord dram_core_from_dram_channel(uint32_t dram_channel) const;
    CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const;

    int32_t bank_offset(BufferType buffer_type, uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    const std::vector<uint32_t> &bank_ids_from_dram_channel(uint32_t dram_channel) const;

    const std::vector<uint32_t> &bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord &logical_core) const;

    allocator::Statistics get_memory_allocation_statistics(const BufferType &buffer_type) const;

    size_t get_l1_small_size() const;

    void dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const;

    // Set of logical storage only core coordinates
    const std::set<CoreCoord> &storage_only_cores() const { return this->storage_only_cores_; }

    // Set of logical dispatch core coordinates

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    const std::set<CoreCoord> &ethernet_cores() const { return this->ethernet_cores_; }

    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& physical_core) const;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& physical_cores) const;

    void deallocate_buffers();

    // machine epsilon
    float sfpu_eps() const;

    const JitBuildEnv& build_env() const { return this->build_env_; }
    const string build_firmware_target_path(JitBuildProcessorType t, int i) const;
    const string build_kernel_target_path(JitBuildProcessorType t, int i, const string& kernel_name) const;
    const JitBuildState& build_firmware_state(JitBuildProcessorType t, int i) const;
    const JitBuildState& build_kernel_state(JitBuildProcessorType t, int i) const;
    const JitBuildStateSubset build_kernel_states(JitBuildProcessorType t) const;
#if 0 // TODO: Revise this
    SystemMemoryManager& sysmem_manager() { return *sysmem_manager_; }
#endif
    // [RONIN]
    CQManager *cq_manager() { return cq_manager_.get(); }
    HWCommandQueue& hw_command_queue(size_t cq_id = 0);
    CommandQueue& command_queue(size_t cq_id = 0);

    // Metal trace device capture mode
    void begin_trace(const uint8_t cq_id, const uint32_t tid, const uint32_t trace_buff_size);
    void end_trace(const uint8_t cq_id, const uint32_t tid);
    void replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking);
    void release_trace(const uint32_t tid);
    std::shared_ptr<TraceBuffer> get_trace(const uint32_t tid);

    bool using_slow_dispatch() const;
    void check_allocator_is_initialized() const;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize(size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap = {}, bool minimal = false);
    void initialize_cluster();
    void initialize_allocator(size_t l1_small_size, const std::vector<uint32_t> &l1_bank_remap = {});
    void initialize_build();
    void build_firmware();
    void initialize_firmware(CoreCoord phys_core, launch_msg_t *launch_msg);
    void initialize_and_launch_firmware();
    void initialize_command_queue();
    void initialize_synchronous_sw_cmd_queue();
    void configure_kernel_variant(Program& program, string path, std::vector<uint32_t> compile_args, CoreCoord kernel_core, CoreCoord Kernel_physical_core,
                                  CoreType dispatch_core_type, CoreCoord upstream_physical_core, CoreCoord downstream_physical_core, std::map<string, string> defines_in, NOC noc_index, bool is_active_eth_core = false);
    // [RONIN] SKIPPED: compile_command_queue_programs
    // [RONIN] SKIPPED: configure_command_queue_programs
#if 0 // TODO: Revise this
    void compile_command_queue_programs();
    void configure_command_queue_programs();
#endif
    void clear_l1_state();
    std::pair<int, int> build_processor_type_to_index(JitBuildProcessorType t) const;

    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    // APIs to access this device's work executor
    void push_work(std::function<void()>&& work, bool blocking = false);
    void push_work(std::shared_ptr<std::function<void()>> work, bool blocking = false);
    void synchronize();
    void set_worker_mode(const WorkExecutorMode& mode);
    void enable_async(bool enable);
    WorkExecutorMode get_worker_mode() { return work_executor.get_worker_mode(); }
    void set_worker_queue_mode(const WorkerQueueMode& mode) { this->work_executor.set_worker_queue_mode(mode); }
    WorkerQueueMode get_worker_queue_mode() { return this->work_executor.get_worker_queue_mode(); }
    // TODO: Uplift usage of friends. Buffer and Program just need access to allocator
    friend class Buffer;
    friend class Program;
#if 0 // TODO: Revise this
    friend class SystemMemoryManager;
#endif

    static constexpr MemoryAllocator allocator_scheme_ = MemoryAllocator::L1_BANKING;
    static ActiveDevices active_devices_;
    chip_id_t id_;
    uint32_t build_key_;
    std::unique_ptr<Allocator> allocator_ = nullptr;
    bool initialized_ = false;

    JitBuildEnv build_env_;
    JitBuildStateSet firmware_build_states_;
    JitBuildStateSet kernel_build_states_;

    std::set<CoreCoord> compute_cores_;
    std::set<CoreCoord> storage_only_cores_;
    std::set<CoreCoord> ethernet_cores_;

    // SystemMemoryManager is the interface to the hardware command queue
    std::vector<std::unique_ptr<HWCommandQueue>> hw_command_queues_;
    std::vector<std::unique_ptr<CommandQueue>> sw_command_queues_;
    // Work Executor for this device - can asynchronously process host side work for
    // all tasks scheduled on this device
    WorkExecutor work_executor;
    uint32_t worker_thread_core;
#if 0 // TODO: Revise this
    std::unique_ptr<SystemMemoryManager> sysmem_manager_;
#endif
    std::unique_ptr<CQManager> cq_manager_;
    uint8_t num_hw_cqs_;

    vector<std::unique_ptr<Program, tt::tt_metal::detail::ProgramDeleter>> command_queue_programs;
    bool using_fast_dispatch = false;
    program_cache::detail::ProgramCache program_cache;

    // Program cache interface. Syncrhonize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    void enable_program_cache() {
        log_info(tt::LogMetal, "Enabling program cache on device {}", this->id_);
        this->synchronize();
        program_cache.enable();
    }
    void disable_and_clear_program_cache() {
        log_info(tt::LogMetal, "Disabling and clearing program cache on device {}", this->id_);
        this->synchronize();
        if (this->program_cache.is_enabled()) {
            program_cache.clear();
            program_cache.disable();
        }
    }
    std::size_t num_program_cache_entries() {
        this->synchronize();
        return program_cache.num_entries();
    }

#if 0 // [RONIN]
    inline bool in_main_thread() {
        // Detect if an instruction is being called in the main thread or worker thread
        return (std::hash<std::thread::id>{}(std::this_thread::get_id()) == this->work_executor.get_parent_thread_id())
                or get_worker_mode() == WorkExecutorMode::SYNCHRONOUS;
    }
#else
    // used by TT-DNN
    inline bool in_main_thread() {
        return true;
    }
#endif

   private:
    std::unordered_map<uint32_t, std::shared_ptr<TraceBuffer>> trace_buffer_pool_;
};

}  // namespace tt_metal

}  // namespace tt
