// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/tt_metal.hpp"

//#include <numa.h> // [RONIN]
#include <algorithm>
#include <filesystem>
//#include <mutex> // [RONIN]
#include <optional>
#include <string>
#include <unordered_set>

#include "dev_msgs.h"
#include "impl/allocator/allocator.hpp"
//#include "impl/debug/dprint_server.hpp" // [TODO]
#include "impl/dispatch/command_queue.hpp"
//#include "tools/profiler/profiler.hpp" // [TODO]
#include "tt_metal/detail/program.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

namespace tt {

namespace tt_metal {

namespace {

void ConfigureKernelGroup(
    const Program &program, const KernelGroup *kernel_group, Device *device, const CoreCoord &logical_core) {
    if (kernel_group->compute_id.has_value()) {
        detail::GetKernel(program, kernel_group->compute_id.value())->configure(device, logical_core);
    }
    if (kernel_group->riscv1_id.has_value()) {
        detail::GetKernel(program, kernel_group->riscv1_id.value())->configure(device, logical_core);
    }
    if (kernel_group->riscv0_id.has_value()) {
        detail::GetKernel(program, kernel_group->riscv0_id.value())->configure(device, logical_core);
    }
    if (kernel_group->erisc_id.has_value()) {
        detail::GetKernel(program, kernel_group->erisc_id.value())->configure(device, logical_core);
    }
}

std::optional<uint32_t> get_semaphore_address(const Program &program, const CoreRange &core_range) {
    std::optional<uint32_t> address = nullopt;
    std::vector<uint32_t> semaphore_histogram(NUM_SEMAPHORES, 0);
    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            CoreCoord logical_core(x, y);
            auto semaphores = program.semaphores_on_core(logical_core);
            if (semaphores.size() == NUM_SEMAPHORES) {
                TT_THROW(
                    "Cannot add semaphore on core " + logical_core.str() + ". Max number of semaphores (" +
                    std::to_string(NUM_SEMAPHORES) + ") reached!");
            }

            for (const auto &semaphore : semaphores) {
                semaphore_histogram[semaphore.get().id()]++;
            }
        }
    }

    std::optional<uint32_t> uninitialized_sem_id = nullopt;
    for (int sem_id = 0; sem_id < semaphore_histogram.size(); sem_id++) {
        if (semaphore_histogram.at(sem_id) == 0) {
            uninitialized_sem_id = sem_id;
            break;
        }
    }

    if (uninitialized_sem_id.has_value()) {
        address = SEMAPHORE_BASE + (L1_ALIGNMENT * uninitialized_sem_id.value());
    } else {
        TT_THROW("Unable to initialize semaphores on core range " + core_range.str());
    }

    return address;
}

inline void SetRuntimeArgs(
    const Program &program, KernelHandle kernel_id, const CoreCoord &c, const std::vector<uint32_t> &runtime_args) {
    if (runtime_args.size() != 0) {
        detail::GetKernel(program, kernel_id)->set_runtime_args(c, runtime_args);
    }
}

inline void SetRuntimeArgs(
    const Program &program,
    KernelHandle kernel_id,
    const CoreRange &core_range,
    const std::vector<uint32_t> &runtime_args) {
    if (runtime_args.size() != 0) {
        auto kernel = detail::GetKernel(program, kernel_id);
        for (auto x = core_range.start.x; x <= core_range.end.x; ++x) {
            for (auto y = core_range.start.y; y <= core_range.end.y; ++y) {
                kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
            }
        }
    }
}

inline void SetRuntimeArgs(
    const Program &program,
    KernelHandle kernel_id,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &runtime_args) {
    if (runtime_args.size() != 0) {
        auto kernel = detail::GetKernel(program, kernel_id);
        for (const auto &core_range : core_range_set.ranges()) {
            for (auto x = core_range.start.x; x <= core_range.end.x; ++x) {
                for (auto y = core_range.start.y; y <= core_range.end.y; ++y) {
                    kernel->set_runtime_args(CoreCoord(x, y), runtime_args);
                }
            }
        }
    }
}

inline void SetRuntimeArgs(
    CommandQueue &cq,
    const std::shared_ptr<Kernel> kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    std::shared_ptr<RuntimeArgs> runtime_args,
    bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode
    std::visit(
        [&](auto &&core_spec) {
            using T = std::decay_t<decltype(core_spec)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                EnqueueSetRuntimeArgs(cq, kernel, core_spec, runtime_args, blocking);
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                for (auto x = core_spec.start.x; x <= core_spec.end.x; x++) {
                    for (auto y = core_spec.start.y; y <= core_spec.end.y; y++) {
                        EnqueueSetRuntimeArgs(cq, kernel, CoreCoord(x, y), runtime_args, blocking);
                    }
                }
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                for (const auto &core_range : core_spec.ranges()) {
                    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                            EnqueueSetRuntimeArgs(cq, kernel, CoreCoord(x, y), runtime_args, blocking);
                        }
                    }
                }
            }
        },
        core_spec);
}

inline void SetRuntimeArgs(
    CommandQueue &cq,
    const std::shared_ptr<Kernel> kernel,
    const std::vector<CoreCoord> &core_spec,
    const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args,
    bool blocking) {
    // SetRuntimeArgs API for Async CQ Mode (support vector of runtime args)
    for (size_t i = 0; i < core_spec.size(); i++) {
        EnqueueSetRuntimeArgs(cq, kernel, core_spec[i], runtime_args[i], blocking);
    }
}

}  // namespace

// #define DEBUG_PRINT_SHARD
namespace device_pool {

// Definition of the global device vector
std::vector<Device *> devices;

}  // namespace device_pool

namespace device_cpu_allocator {

#if 0 // [RONIN]
std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node(std::unordered_set<uint32_t> &free_cores) {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    if (numa_available() != -1) {
        // Host has NUMA enabled. Group CPU IDs by the NUMA nodes they belong to.
        for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            int node = numa_node_of_cpu(cpu);
            if (cpu_cores_per_numa_node.find(node) == cpu_cores_per_numa_node.end()) {
                cpu_cores_per_numa_node.insert({node, {}});
            }
            free_cores.insert(cpu);
            cpu_cores_per_numa_node.at(node).push_back(cpu);
        }
    } else {
        // Host does not have NUMA. Place all CPU Ids under a single node (0).
        log_warning(tt::LogMetal, "Host does not use NUMA. May see reduced performance.");
        for (int cpu = 0; cpu < sysconf(_SC_NPROCESSORS_ONLN); ++cpu) {
            free_cores.insert(cpu);
        }
    }
    return cpu_cores_per_numa_node;
}

int get_cpu_core_for_device_worker_thread(
    int mmio_controlled_device_id,
    const std::unordered_map<int, std::vector<uint32_t>> &cpu_cores_per_numa_node,
    std::unordered_set<uint32_t> &free_cores) {
    int core_assigned_to_device = 0;
    if (numa_available() != -1) {
        // Get NUMA node that the current device is mapped to through UMD
        int numa_node_for_device = tt::Cluster::instance().get_numa_node_for_device(mmio_controlled_device_id);
        if (cpu_cores_per_numa_node.find(numa_node_for_device) != cpu_cores_per_numa_node.end()) {
            // NUMA node reported by UMD exists on host. Choose a core on this numa-node using round robin policy
            int num_cores_in_numa_node = cpu_cores_per_numa_node.at(numa_node_for_device).size();
            core_assigned_to_device =
                cpu_cores_per_numa_node.at(numa_node_for_device).at(mmio_controlled_device_id % num_cores_in_numa_node);
        } else {
            // NUMA node reported by UMD does not exist on host. Use round-robin binding policy for this worker thread.
            log_warning(
                tt::LogMetal,
                "NUMA node {} for device {} does not exist on host.",
                numa_node_for_device,
                mmio_controlled_device_id);
            core_assigned_to_device = mmio_controlled_device_id % sysconf(_SC_NPROCESSORS_ONLN);
        }
    } else {
        // System does not use NUMA. Use-round robin binding strategy.
        core_assigned_to_device = mmio_controlled_device_id % sysconf(_SC_NPROCESSORS_ONLN);
    }
    free_cores.erase(core_assigned_to_device);
    return core_assigned_to_device;
}

std::unordered_map<uint32_t, uint32_t> get_device_id_to_core_map(const std::vector<chip_id_t>& device_ids, std::unordered_set<uint32_t>& free_cores, bool use_numa_node_based_thread_binding) {
    std::unordered_map<uint32_t, uint32_t> device_to_core_map = {};
    if (use_numa_node_based_thread_binding) {
        auto cpu_cores_per_numa_node = device_cpu_allocator::get_cpu_cores_per_numa_node(free_cores);
        for (const auto &device_id : device_ids) {
            device_to_core_map.insert({device_id, device_cpu_allocator::get_cpu_core_for_device_worker_thread(device_id, cpu_cores_per_numa_node, free_cores)});
        }
    } else {
        for (const auto &device_id : device_ids) {
            device_to_core_map.insert({device_id, device_id % sysconf(_SC_NPROCESSORS_ONLN)});
        }
    }
    return device_to_core_map;
}

void bind_current_thread_to_free_cores(const std::unordered_set<uint32_t> &free_cores) {
    cpu_set_t cpuset;
    pthread_t current_thread = pthread_self();
    CPU_ZERO(&cpuset);

    for (const auto &free_core : free_cores) {
        CPU_SET(free_core, &cpuset);
    }
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind main thread to free CPU cores. May see performance degradation. Error Code: {}",
            rc);
    }
}

#else
// One core, no NUMA

std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node(std::unordered_set<uint32_t> &free_cores) {
    free_cores.insert(0);
    return {};
}

int get_cpu_core_for_device_worker_thread(
        int mmio_controlled_device_id,
        const std::unordered_map<int, std::vector<uint32_t>> &cpu_cores_per_numa_node,
        std::unordered_set<uint32_t> &free_cores) {
    int core_assigned_to_device = 0;
    free_cores.erase(core_assigned_to_device);
    return core_assigned_to_device;
}

std::unordered_map<uint32_t, uint32_t> get_device_id_to_core_map(const std::vector<chip_id_t>& device_ids, std::unordered_set<uint32_t>& free_cores, bool use_numa_node_based_thread_binding) {
    std::unordered_map<uint32_t, uint32_t> device_to_core_map = {};
    for (const auto &device_id : device_ids) {
        device_to_core_map.insert({device_id, 0});
    }
    return device_to_core_map;
}

void bind_current_thread_to_free_cores(const std::unordered_set<uint32_t> &free_cores) {
    // nothing to do
}

#endif

}  // namespace device_cpu_allocator

namespace detail {

std::map<chip_id_t, Device *> CreateDevices(
    std::vector<chip_id_t> device_ids,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;
    std::map<chip_id_t, Device *> active_devices;  // TODO: pass this to CloseDevices
    static bool use_numa_node_based_thread_binding = parse_env("TT_METAL_NUMA_BASED_AFFINITY", false);

    std::unordered_set<uint32_t> free_cores = {};
    std::vector<chip_id_t> all_device_ids = {};

    for (const auto &device_id : device_ids) {
        // Get list of all devices in the cluster connected to the passed in device_ids
        const auto &mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        if (std::find(all_device_ids.begin(), all_device_ids.end(), mmio_device_id) == all_device_ids.end()) {
            for (const auto &mmio_controlled_device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
                all_device_ids.push_back(mmio_controlled_device_id);
            }
        }
    }
    // Determine which CPU cores the worker threads need to be placed on for each device
    std::unordered_map<uint32_t, uint32_t> device_to_core_map = device_cpu_allocator::get_device_id_to_core_map(all_device_ids, free_cores, use_numa_node_based_thread_binding);

    for (const auto& device_id : all_device_ids) {
        int core_assigned_to_device = device_to_core_map.at(device_id);
        Device *dev = new Device(
            device_id,
            num_hw_cqs,
            l1_small_size,
            l1_bank_remap,
            false,
            core_assigned_to_device);
        active_devices.insert({device_id, dev});
        detail::InitDeviceProfiler(dev);
    }

    if (use_numa_node_based_thread_binding) {
        // Bind main thread to cores not being used by workers.
        device_cpu_allocator::bind_current_thread_to_free_cores(free_cores);
    }
    // TODO: need to only enable routing for used mmio chips
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    for (auto &active_device: active_devices){
        detail::InitDeviceProfiler(active_device.second);
    }
    return active_devices;
}

void CloseDevices(std::map<chip_id_t, Device *> devices) {
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    for (const auto &[device_id, dev] : devices) {
        dev->close();
    }
}

void print_page(
    uint32_t dev_page_id,
    CoreCoord core,
    uint32_t host_page_id,
    CoreCoord noc_coordinates,
    uint32_t l1_address,
    uint32_t bank_id,
    std::vector<uint32_t> page) {
    std::cout << "dev_page_index " << dev_page_id << " on core " << core.str() << std::endl;
    std::cout << "host_page_index " << host_page_id << std::endl;
    std::cout << "noc coordinates " << noc_coordinates.str() << std::endl;
    std::cout << "l1_address " << l1_address << std::endl;
    std::cout << "bank id " << bank_id << std::endl;

    std::cout << "0x";
    for (auto entry : page) {
        std::cout << std::hex << entry << std::dec;
    }
    std::cout << std::dec << std::endl;
}

void WriteToDeviceSharded(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    uint32_t host_buffer_size_bytes = host_buffer.size() * sizeof(uint32_t);
    TT_FATAL(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer_size_bytes,
        buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(buffer.size() % page_size == 0);

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    TT_ASSERT(page_size % bytes_per_page_entry == 0);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    auto device = buffer.device();

    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
    auto total_pages = buffer.num_pages();
    for (int host_page_id = 0; host_page_id < total_pages; host_page_id++) {
        auto dev_page_id = buffer_page_mapping.host_page_to_dev_page_mapping_[host_page_id];
        auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_id]];
        auto bank_id = device->bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto absolute_address = buffer.sharded_page_address(bank_id, dev_page_id);
        auto data_index = host_page_id * num_entries_per_page;
        std::vector<uint32_t> page;
        page.insert(
            page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);

        auto noc_coordinates = buffer.noc_coordinates(bank_id);
        llrt::write_hex_vec_to_core(device->id(), noc_coordinates, page, absolute_address);
    }
}

void WriteToDeviceInterleavedContiguous(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    uint32_t host_buffer_size_bytes = host_buffer.size() * sizeof(uint32_t);
    TT_FATAL(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer_size_bytes,
        buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_FATAL(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    static constexpr uint32_t bytes_per_page_entry = sizeof(uint32_t);
    TT_FATAL(page_size % bytes_per_page_entry == 0);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());
    uint32_t bank_index = 0;
    int data_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        page.insert(
            page.end(), host_buffer.begin() + data_index, host_buffer.begin() + data_index + num_entries_per_page);
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                llrt::write_hex_vec_to_core(device->id(), noc_coordinates, page, absolute_address);
            } break;
            default: TT_FATAL(false && "Unsupported buffer type to write to device!");
        }

        bank_index = (bank_index + 1) % num_banks;
        data_index += num_entries_per_page;
    }
}

void WriteToDevice(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    ZoneScoped;
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED ||
        buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK) {
        WriteToDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        WriteToDeviceSharded(buffer, host_buffer);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void WriteToBuffer(std::shared_ptr<const Buffer> buffer, const std::vector<uint32_t> &host_buffer) {
    WriteToBuffer(*buffer, host_buffer);
}

void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:  // fallthrough
        case BufferType::L1:    // fallthrough
        case BufferType::L1_SMALL: {
            WriteToDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_FATAL(false && "Writing to host memory is unsupported!");
        } break;
        default: TT_FATAL(false && "Unsupported buffer type!");
    }
}

void ReadFromDeviceInterleavedContiguous(const Buffer &buffer, std::vector<uint32_t> &host_buffer) {
    host_buffer.clear();  // overwrite the data
    uint32_t page_size = buffer.page_size();
    TT_FATAL(buffer.size() % page_size == 0);
    uint32_t num_pages = buffer.size() / page_size;

    auto device = buffer.device();
    auto num_banks = device->num_banks(buffer.buffer_type());

    uint32_t bank_index = 0;
    for (int page_index = 0; page_index < num_pages; page_index++) {
        auto absolute_address = buffer.page_address(bank_index, page_index);
        std::vector<uint32_t> page;
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                auto noc_coordinates = buffer.noc_coordinates(bank_index);
                page = llrt::read_hex_vec_from_core(device->id(), noc_coordinates, absolute_address, page_size);
            } break;
            default: TT_FATAL(false && "Unsupported buffer type to write to device!");
        }

        // Copy page into host buffer
        for (uint32_t entry : page) {
            host_buffer.push_back(entry);
        }

        bank_index = (bank_index + 1) % num_banks;
    }
}

void read_pages_to_host_helper(
    Device *device,
    const Buffer &dev_buffer,
    std::vector<uint32_t> &host_buffer,
    const uint32_t &page_size,
    const uint32_t &host_page_id,
    const uint32_t &dev_page_id,
    const uint32_t &bank_id) {
    auto absolute_address = dev_buffer.sharded_page_address(bank_id, dev_page_id);
    auto noc_coordinates = dev_buffer.noc_coordinates(bank_id);
    uint32_t num_entries_per_page = page_size / sizeof(uint32_t);
    uint32_t host_buffer_start = host_page_id * num_entries_per_page;
    tt::Cluster::instance().read_core(host_buffer.data() + host_buffer_start, page_size, tt_cxy_pair(device->id(), noc_coordinates), absolute_address);
}

void ReadFromDeviceSharded(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    TensorMemoryLayout buffer_layout = buffer.buffer_layout();

    auto device = buffer.device();
#ifdef DEBUG_PRINT_SHARD
    std::cout << "Reading From Device Height Sharded " << std::endl;
#endif

    int output_page_index = 0;
    auto total_pages = buffer.num_dev_pages();
    uint32_t page_size = buffer.page_size();
    uint32_t bytes_per_page_entry = sizeof(uint32_t);
    uint32_t num_entries_per_page = page_size / bytes_per_page_entry;

    host_buffer = std::vector<uint32_t>(total_pages * num_entries_per_page);

    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
    for (int dev_page_id = 0; dev_page_id < total_pages; dev_page_id++) {
        auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_id]];
        auto bank_id = device->bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto host_page_id = buffer_page_mapping.dev_page_to_host_page_mapping_[dev_page_id];
        if (host_page_id.has_value()) {
            if (!shard_order) {
                read_pages_to_host_helper(
                    device, buffer, host_buffer, page_size, host_page_id.value(), dev_page_id, bank_id);
            } else {
                read_pages_to_host_helper(device, buffer, host_buffer, page_size, dev_page_id, dev_page_id, bank_id);
            }
        }
    }
}

void ReadFromDevice(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    ZoneScoped;
    host_buffer.clear();  // overwrite the data
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED ||
        buffer.buffer_layout() == TensorMemoryLayout::SINGLE_BANK) {
        ReadFromDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        ReadFromDeviceSharded(buffer, host_buffer, shard_order);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void ReadFromBuffer(std::shared_ptr<const Buffer> buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    ReadFromBuffer(*buffer, host_buffer, shard_order);
}

void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer, bool shard_order) {
    Device *device = buffer.device();
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::L1:  // fallthrough
        case BufferType::L1_SMALL: {
            if (buffer.buffer_type() == BufferType::DRAM) {
                tt::Cluster::instance().dram_barrier(device->id());
            } else {
                tt::Cluster::instance().l1_barrier(device->id());
            }
            ReadFromDevice(buffer, host_buffer, shard_order);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_FATAL(false && "Reading from host memory is unsupported!");
        } break;
        default: TT_FATAL(false && "Unsupported buffer type!");
    }
}

void ReadShard(const Buffer &buffer, std::vector<uint32_t> &host_buffer, const uint32_t &core_id) {
    Device *device = buffer.device();
    TT_ASSERT(is_sharded(buffer.buffer_layout()));
    host_buffer.clear();  // overwrite the data

    uint32_t num_entries_per_page = buffer.page_size() / sizeof(uint32_t);
    uint32_t num_entries_per_shard = num_entries_per_page * buffer.shard_spec().size();
    host_buffer = std::vector<uint32_t>(num_entries_per_shard);

    std::vector<uint32_t> page_ids;
    auto buffer_page_mapping = generate_buffer_page_mapping(buffer);
    for (uint32_t i = 0; i < buffer_page_mapping.dev_page_to_core_mapping_.size(); i++) {
        if (buffer_page_mapping.dev_page_to_core_mapping_[i] == core_id) {
            page_ids.push_back(i);
        }
    }

    uint32_t host_page_id = 0;
    for (auto dev_page_id : page_ids) {
        auto core = buffer_page_mapping.all_cores_[buffer_page_mapping.dev_page_to_core_mapping_[dev_page_id]];
        auto bank_id = device->bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        read_pages_to_host_helper(device, buffer, host_buffer, buffer.page_size(), host_page_id, dev_page_id, bank_id);
        host_page_id++;
    }
}

void LaunchProgram(Device *device, std::shared_ptr<Program> program, bool wait_until_cores_done) {
    LaunchProgram(device, *program, wait_until_cores_done);
}

void LaunchProgram(Device *device, Program &program, bool wait_until_cores_done) {
    {  // Profiler scope start
        ZoneScoped;
        detail::DispatchStateCheck(false);
        detail::CompileProgram(device, program);
        detail::WriteRuntimeArgsToDevice(device, program);
        detail::ConfigureDeviceWithProgram(device, program);
        auto device_id = device->id();

        tt::Cluster::instance().dram_barrier(device_id);

        // Note: the l1_barrier below is needed to be sure writes to cores that
        // don't get the GO mailbox (eg, storage cores) have all landed
        tt::Cluster::instance().l1_barrier(device->id());

        std::unordered_map<CoreType, std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
        std::unordered_set<CoreCoord> not_done_cores;
        for (const auto &[core_type, logical_cores] : logical_cores_used_in_program) {
            for (const auto &logical_core : logical_cores) {
                launch_msg_t *msg = &program.kernels_on_core(logical_core, core_type)->launch_msg;
                auto physical_core = device->physical_core_from_logical_core(logical_core, core_type);
                not_done_cores.insert(physical_core);
                tt::llrt::write_launch_msg_to_core(device->id(), physical_core, msg);
            }
        }
        if (wait_until_cores_done) {
            // Wait for all cores to be done
            llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);
        }
    }  // Profiler scope end
    if (wait_until_cores_done) {
        DumpDeviceProfileResults(device, program);
    }
}

void WaitProgramDone(Device *device, Program &program) {
    auto device_id = device->id();
    std::unordered_map<CoreType, std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
    std::unordered_set<CoreCoord> not_done_cores;
    for (const auto &[core_type, logical_cores] : logical_cores_used_in_program) {
        for (const auto &logical_core : logical_cores) {
            auto physical_core = device->physical_core_from_logical_core(logical_core, core_type);
            not_done_cores.insert(physical_core);
        }
    }
    // Wait for all cores to be done
    llrt::internal_::wait_until_cores_done(device_id, RUN_MSG_GO, not_done_cores);
    DumpDeviceProfileResults(device, program);
}

bool ConfigureDeviceWithProgram(Device *device, Program &program, bool fd_bootloader_mode) {
    ZoneScoped;
    bool pass = true;
    // This is function is shared between FD and SD.
    // We call this function when initializing HW Command Queues (tracked as fd_bootloader_mode) for Fast Dispatch.
    // Used to Launch programs for Slow dispatch.
    bool using_fast_dispatch = fd_bootloader_mode;
    detail::DispatchStateCheck(using_fast_dispatch);

    auto device_id = device->id();

    program.allocate_circular_buffers();
    detail::ValidateCircularBufferRegion(program, device);

    std::unordered_map<CoreType, std::vector<CoreCoord>> logical_cores_used_in_program = program.logical_cores();
    for (const auto &[core_type, logical_cores] : logical_cores_used_in_program) {
        for (const auto &logical_core : logical_cores) {
            KernelGroup *kernel_group = program.kernels_on_core(logical_core, core_type);
            CoreCoord physical_core = device->physical_core_from_logical_core(logical_core, core_type);

            ConfigureKernelGroup(program, kernel_group, device, logical_core);
            // TODO: add support for CB for ethernet cores
            if (core_type == CoreType::WORKER) {
                // CircularBufferConfigVec -- common across all kernels, so written once to the core
                llrt::CircularBufferConfigVec circular_buffer_config_vec = llrt::create_circular_buffer_config_vector();

                auto cbs_on_core = program.circular_buffers_on_core(logical_core);
                for (auto circular_buffer : cbs_on_core) {
                    for (uint32_t buffer_index : circular_buffer->buffer_indices()) {
                        llrt::set_config_for_circular_buffer(
                            circular_buffer_config_vec,
                            buffer_index,
                            circular_buffer->address(),
                            circular_buffer->size(),
                            circular_buffer->num_pages(buffer_index));
                    }
                }  // PROF_END("CBS")

                if (cbs_on_core.size()) {
                    llrt::write_circular_buffer_config_vector_to_core(
                        device_id, physical_core, circular_buffer_config_vec);
                }
            }
            program.init_semaphores(*device, logical_core, core_type);
        }
    }

    return pass;
}

// Return base address in L1 for Runtime Args given processor type (and eth mode in case of ERISC).
uint32_t GetL1ArgBaseAddr(std::shared_ptr<Kernel> kernel) {
    const RISCV &riscv = kernel->processor();
    uint32_t l1_arg_base = 0;

    switch (riscv) {
        case RISCV::BRISC: {
            l1_arg_base = BRISC_L1_ARG_BASE;
        } break;
        case RISCV::NCRISC: {
            l1_arg_base = NCRISC_L1_ARG_BASE;
        } break;
        case RISCV::ERISC: {
            auto config = std::get<EthernetConfig>(kernel->config());
            if (config.eth_mode == Eth::IDLE) {
                l1_arg_base = IDLE_ERISC_L1_ARG_BASE;
            } else {
                l1_arg_base = eth_l1_mem::address_map::ERISC_L1_ARG_BASE;
            }
        } break;
        case RISCV::COMPUTE: {
            l1_arg_base = TRISC_L1_ARG_BASE;
        } break;
        default: TT_THROW("Unsupported {} processor does not support runtime args", riscv);
    }
    return l1_arg_base;
}

void WriteRuntimeArgsToDevice(Device *device, const Program &program) {
    ZoneScoped;
    auto device_id = device->id();
    detail::DispatchStateCheck(false);

    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        const auto kernel = detail::GetKernel(program, kernel_id);
        auto args_base_addr = detail::GetL1ArgBaseAddr(kernel);

        for (const auto &logical_core : kernel->cores_with_runtime_args()) {
            auto physical_core = device->physical_core_from_logical_core(logical_core, kernel->get_kernel_core_type());
            const auto &rt_args = kernel->runtime_args(logical_core);
            log_trace(
                tt::LogMetal,
                "{} - Writing {} unique rtargs to core {} (physical: {}) addr 0x{:x} => args: {}",
                __FUNCTION__,
                rt_args.size(),
                logical_core.str(),
                physical_core.str(),
                args_base_addr,
                rt_args);
            tt::llrt::write_hex_vec_to_core(device_id, physical_core, rt_args, args_base_addr);
        }

        // Unicast common runtime args to all cores for kernel. Fast-Dispatch will multicast as perf opt.
        const auto &common_rt_args = kernel->common_runtime_args();
        auto common_rt_args_offset = kernel->get_common_runtime_args_offset();

        if (common_rt_args.size() > 0) {
            for (auto &core_range : kernel->logical_coreranges()) {
                for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                    for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                        CoreCoord logical_core({x, y});
                        auto physical_core =
                            device->physical_core_from_logical_core(logical_core, kernel->get_kernel_core_type());
                        const auto common_args_addr =
                            args_base_addr +
                            common_rt_args_offset;  // Common args are placed after unique args per core.
                        log_trace(
                            tt::LogMetal,
                            "{} - Writing {} common rtargs to core {} (physical: {}) addr 0x{:x} => args: {}",
                            __FUNCTION__,
                            common_rt_args.size(),
                            logical_core.str(),
                            physical_core.str(),
                            common_args_addr,
                            common_rt_args);
                        tt::llrt::write_hex_vec_to_core(device_id, physical_core, common_rt_args, common_args_addr);
                    }
                }
            }
        }
    }
}

void CompileProgram(Device *device, Program &program) {
    ZoneScoped;
    program.compile(device);
}

void AllocateBuffer(Buffer *buffer, bool bottom_up) {
    EnqueueAllocateBuffer(buffer->device()->command_queue(), buffer, bottom_up, false);
}

void DeallocateBuffer(Buffer *buffer) {
    EnqueueDeallocateBuffer(
        buffer->device()->command_queue(),
        *(buffer->device()->allocator_),
        buffer->address(),
        buffer->buffer_type(),
        false);
}

void GetBufferAddress(const Buffer *buffer, uint32_t *address_on_host) {
    EnqueueGetBufferAddr(buffer->device()->command_queue(), address_on_host, buffer, false);
}

Device *GetDeviceHandle(chip_id_t device_id) {
    ZoneScoped;
    TT_ASSERT(device_id < device_pool::devices.size());
    TT_ASSERT(device_pool::devices[device_id] != nullptr);
    return device_pool::devices[device_id];
}

void DisableAllocs(Device *device) { tt::tt_metal::allocator::disable_allocs(*(device->allocator_)); }

void EnableAllocs(Device *device) { tt::tt_metal::allocator::enable_allocs(*(device->allocator_)); }

}  // namespace detail

size_t GetNumAvailableDevices() {
#ifdef TT_METAL_VERSIM_DISABLED
    return tt::Cluster::instance().number_of_devices();
#else
    return 1;
#endif
}

size_t GetNumPCIeDevices() {
#ifdef TT_METAL_VERSIM_DISABLED
    return tt::Cluster::instance().number_of_pci_devices();
#else
    return 1;
#endif
}

Device *CreateDevice(
    chip_id_t device_id,
    const uint8_t num_hw_cqs,
    const size_t l1_small_size,
    const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;
    static bool use_numa_node_based_thread_binding = parse_env("TT_METAL_NUMA_BASED_AFFINITY", false);
    std::unordered_set<uint32_t> free_cores = {};
    int core_assigned_to_device = device_cpu_allocator::get_device_id_to_core_map({device_id}, free_cores, use_numa_node_based_thread_binding)[device_id];
    Device *dev = new Device(device_id, num_hw_cqs, l1_small_size, l1_bank_remap, false, core_assigned_to_device);
    if (use_numa_node_based_thread_binding) {
        // Bind main thread to cores not being used by workers.
        device_cpu_allocator::bind_current_thread_to_free_cores(free_cores);
    }
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    detail::InitDeviceProfiler(dev);
    return dev;
}

Device *CreateDeviceMinimal(chip_id_t device_id) {
    ZoneScoped;
    Device *dev = new Device(device_id, 1, DEFAULT_L1_SMALL_SIZE, {}, true);
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
    return dev;
}

bool CloseDevice(Device *device) {
    ZoneScoped;
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    auto device_id = device->id();
    TT_ASSERT(device_id < device_pool::devices.size());
    if (device_pool::devices[device_id] != nullptr) {
        device_pool::devices[device_id] = nullptr;
    }
    return device->close();
}

Program CreateProgram() { return Program(); }

KernelHandle CreateKernel(
    Program &program,
    const std::string &file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> &config) {
    return std::visit(
        [&](auto &&cfg) -> KernelHandle {
            CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
            std::shared_ptr<Kernel> kernel;
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                detail::CheckDataMovementConfig(program, file_name, core_ranges);
                kernel = std::make_shared<DataMovementKernel>(file_name, core_ranges, cfg);
                return detail::AddKernel(program, kernel, CoreType::WORKER);
            } else if constexpr (std::is_same_v<T, ComputeConfig>) {
                kernel = std::make_shared<ComputeKernel>(file_name, core_ranges, cfg);
                return detail::AddKernel(program, kernel, CoreType::WORKER);
            } else if constexpr (std::is_same_v<T, EthernetConfig>) {
                kernel = std::make_shared<EthernetKernel>(file_name, core_ranges, cfg);
                return detail::AddKernel(program, kernel, CoreType::ETH);
            }
        },
        config);
}

CBHandle CreateCircularBuffer(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const CircularBufferConfig &config) {
    CoreRangeSet core_ranges = detail::GetCoreRangeSet(core_spec);
    return program.add_circular_buffer(core_ranges, config);
}

const CircularBufferConfig &GetCircularBufferConfig(Program &program, CBHandle cb_handle) {
    return detail::GetCircularBuffer(program, cb_handle)->config();
}

void UpdateCircularBufferTotalSize(Program &program, CBHandle cb_handle, uint32_t total_size) {
    std::shared_ptr<CircularBuffer> circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    if (not circular_buffer->globally_allocated()) {
        program.invalidate_circular_buffer_allocation();
    }
    circular_buffer->config().set_total_size(total_size);
}

void UpdateCircularBufferPageSize(Program &program, CBHandle cb_handle, uint8_t buffer_index, uint32_t page_size) {
    detail::GetCircularBuffer(program, cb_handle)->config().set_page_size(buffer_index, page_size);
}

void UpdateDynamicCircularBufferAddress(Program &program, CBHandle cb_handle, const Buffer &buffer) {
    auto circular_buffer = detail::GetCircularBuffer(program, cb_handle);
    circular_buffer->config().set_globally_allocated_address(buffer);
    circular_buffer->assign_global_address();
}

uint32_t CreateSemaphore(
    Program &program,
    const std::variant<CoreRange, CoreRangeSet> &core_spec,
    uint32_t initial_value,
    CoreType core_type) {
    return std::visit(
        [&](auto &&c) -> uint32_t {
            using T = std::decay_t<decltype(c)>;
            CoreRangeSet crs({});
            if constexpr (std::is_same_v<T, CoreRange>) {
                crs = CoreRangeSet({c});
            } else {
                crs = c;
            }
            std::optional<uint32_t> address;
            TT_FATAL(crs.ranges().size() > 0, "Expecting a non-empty CoreRangeSet!");
            for (const auto &core_range : crs.ranges()) {
                CoreCoord start_core = core_range.start;
                CoreCoord end_core = core_range.end;
                std::optional<uint32_t> addr_candidate = get_semaphore_address(program, core_range);
                if (!address.has_value()) {
                    address = addr_candidate;
                } else {
                    address = std::max(address.value(), addr_candidate.value());
                }
            }
            TT_FATAL(address.has_value(), "Unable to initialize Semaphore!");

            program.add_semaphore(crs, address.value(), initial_value, core_type);

            return address.value();
        },
        core_spec);
}

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig &config) {
    return std::make_shared<Buffer>(
        config.device, config.size, config.page_size, config.buffer_type, config.buffer_layout);
}

std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig &config) {
    return std::make_shared<Buffer>(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        config.shard_parameters);
}

void DeallocateBuffer(Buffer &buffer) { buffer.deallocate(); }

void AssignGlobalBufferToProgram(
    std::shared_ptr<Buffer> buffer, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program>> program) {
    detail::DispatchStateCheck(not buffer->device()->using_slow_dispatch());
    EnqueueAddBufferToProgram(buffer->device()->command_queue(), buffer, program, false);
}

void SetRuntimeArgs(
    const Program &program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast "
        "Dispatch.");
    std::visit([&](auto &&core_spec) { SetRuntimeArgs(program, kernel_id, core_spec, runtime_args); }, core_spec);
}

void SetRuntimeArgs(
    const Program &program,
    KernelHandle kernel,
    const std::vector<CoreCoord> &core_spec,
    const std::vector<std::vector<uint32_t>> &runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast "
        "Dispatch.");
    TT_FATAL(
        core_spec.size() == runtime_args.size(),
        "Mistmatch between number of cores {} and number of runtime args {} getting updated",
        core_spec.size(),
        runtime_args.size());
    auto k = detail::GetKernel(program, kernel);
    for (size_t i = 0; i < core_spec.size(); i++) k->set_runtime_args(core_spec[i], runtime_args[i]);
}

void SetRuntimeArgs(
    Device *device,
    const std::shared_ptr<Kernel> kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    std::shared_ptr<RuntimeArgs> runtime_args) {
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgs(device->command_queue(), kernel, core_spec, runtime_args, false);
}

void SetRuntimeArgs(
    Device *device,
    const std::shared_ptr<Kernel> kernel,
    const std::vector<CoreCoord> &core_spec,
    const std::vector<std::shared_ptr<RuntimeArgs>> runtime_args) {
    TT_FATAL(
        core_spec.size() == runtime_args.size(),
        "Mismatch between number of cores {} and number of runtime args {} getting updated",
        core_spec.size(),
        runtime_args.size());
    detail::DispatchStateCheck(not device->using_slow_dispatch());
    SetRuntimeArgs(device->command_queue(), kernel, core_spec, runtime_args, false);
}

void SetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id, const std::vector<uint32_t> &runtime_args) {
    ZoneScoped;
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "This variant of SetCommonRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for "
        "Fast Dispatch.");
    if (runtime_args.size() != 0) {
        detail::GetKernel(program, kernel_id)->set_common_runtime_args(runtime_args);
    }
}

RuntimeArgsData &GetRuntimeArgs(const Program &program, KernelHandle kernel_id, const CoreCoord &logical_core) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return detail::GetKernel(program, kernel_id)->runtime_args_data(logical_core);
}

std::vector<std::vector<RuntimeArgsData>> &GetRuntimeArgs(const Program &program, KernelHandle kernel_id) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return detail::GetKernel(program, kernel_id)->runtime_args_data();
}

RuntimeArgsData &GetCommonRuntimeArgs(const Program &program, KernelHandle kernel_id) {
    TT_FATAL(
        not CommandQueue::async_mode_set(),
        "GetRuntimeArgs can only be called when Asynchronous SW Command Queues are disabled for Fast Dispatch.");
    return detail::GetKernel(program, kernel_id)->common_runtime_args_data();
}

uint32_t BeginTraceCapture(Device *device, const uint8_t cq_id, const uint32_t trace_buff_size) {
    const uint32_t tid = Trace::next_id();
    device->begin_trace(cq_id, tid, trace_buff_size);
    return tid;
}

void EndTraceCapture(Device *device, const uint8_t cq_id, const uint32_t tid) { device->end_trace(cq_id, tid); }

void ReplayTrace(Device *device, const uint8_t cq_id, const uint32_t tid, const bool blocking) {
    device->replay_trace(cq_id, tid, blocking);
}

void ReleaseTrace(Device *device, const uint32_t tid) { device->release_trace(tid); }

#if 1 // [RONIN]
void DumpDeviceProfileResults(Device *device, const Program &program) {
    // Not supported; originally in [tools/profiler/tt_metal_profiler.cpp]
}
#endif

}  // namespace tt_metal

}  // namespace tt
