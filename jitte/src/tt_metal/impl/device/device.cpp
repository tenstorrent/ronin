// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <chrono>
#include "tt_metal/host_api.hpp"
#include "tt_metal/jit_build/genfiles.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "tt_metal/common/core_descriptor.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/env_lib.hpp"
#include "common/utils.hpp"
#include "llrt/llrt.hpp"
#include "dev_msgs.h"
#include "noc/noc_parameters.h"
#include "tt_metal/impl/device/device_pool.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "llrt/hal.hpp"

namespace tt {

namespace tt_metal {

void ::detail::ProgramDeleter::operator()(Program *p) {
    delete p;
}

Device::Device(
    chip_id_t device_id, const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, const std::vector<uint32_t> &l1_bank_remap, bool minimal, uint32_t worker_core) :
    id_(device_id), worker_thread_core(worker_core), work_executor(worker_core, device_id) {
    ZoneScoped;
    tunnel_device_dispatch_workers_ = {};
    this->initialize(num_hw_cqs, l1_small_size, trace_region_size, l1_bank_remap, minimal);
}

/* Get all dispatch cores associated with this device. On return, my_dispatch_cores contains dispatch cores used by
 * this device (split between cores on this device itself and if this is a remote device, the mmio device dispatch
 * cores being used by this device). On return, other_dispatch_cores contains dispatch cores on this device that are
 * used by other (remote) devices.
*/
void Device::get_associated_dispatch_phys_cores(
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &my_dispatch_cores,
    std::unordered_map<chip_id_t,std::unordered_set<CoreCoord>> &other_dispatch_cores) {
    if (this->is_mmio_capable()) {
        for (const chip_id_t &device_id : tt::Cluster::instance().get_devices_controlled_by_mmio_device(this->id_)) {
            uint8_t num_hw_cqs = this->num_hw_cqs();
            uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
            CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device_id);
            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                if (device_id == this->id_) {
                    //mmio device.
                    if (dispatch_core_manager::instance().is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::instance().dispatcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                        my_dispatch_cores[this->id_].insert(phys_core);
                        log_debug(tt::LogMetal, "MMIO Device Dispatch core: Logical: {} - Physical: {}", dispatch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::instance().is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::instance().prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                        my_dispatch_cores[this->id_].insert(phys_core);
                        log_debug(tt::LogMetal, "MMIO Device Prefetch core: Logical: {} - Physical: {}", prefetch_location.str(), phys_core.str());
                    }
                } else if (tt::DevicePool::instance().is_device_active(device_id)) {
                    //non mmio devices serviced by this mmio capable device.
                    //skip remote dispatch cores only if respective remote device is active.
                    if (dispatch_core_manager::instance().is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair dispatch_location = dispatch_core_manager::instance().dispatcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will keep running on MMIO Device.", dispatch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::instance().is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair prefetch_location = dispatch_core_manager::instance().prefetcher_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will keep running on MMIO Device.", prefetch_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::instance().is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair mux_location = dispatch_core_manager::instance().mux_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will keep running on MMIO Device.", mux_location.str(), phys_core.str());
                    }
                    if (dispatch_core_manager::instance().is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                        tt_cxy_pair demux_location = dispatch_core_manager::instance().demux_core(device_id, curr_channel, cq_id);
                        CoreCoord phys_core = get_physical_core_coordinate(demux_location, dispatch_core_type);
                        other_dispatch_cores[this->id_].insert(phys_core);
                        log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will keep running on MMIO Device.", demux_location.str(), phys_core.str());
                    }
                }
            }
        }
    } else {
        //remote device that is active
        uint8_t num_hw_cqs = this->num_hw_cqs();
        auto device_id = this->id_;
        uint16_t curr_channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id);
        CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device_id);
        for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
            if (dispatch_core_manager::instance().is_dispatcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair dispatch_location = dispatch_core_manager::instance().dispatcher_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                my_dispatch_cores[dispatch_location.chip].insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Dispatch core: Logical: {} - Physical: {} will be reset on MMIO Device.", dispatch_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::instance().is_prefetcher_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair prefetch_location = dispatch_core_manager::instance().prefetcher_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                my_dispatch_cores[prefetch_location.chip].insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Prefetch core: Logical: {} - Physical: {} will be reset on MMIO Device.", prefetch_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::instance().is_mux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair mux_location = dispatch_core_manager::instance().mux_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
                my_dispatch_cores[mux_location.chip].insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Mux core: Logical: {} - Physical: {} will be reset on MMIO Device.", mux_location.str(), phys_core.str());
            }
            if (dispatch_core_manager::instance().is_demux_core_allocated(device_id, curr_channel, cq_id)) {
                tt_cxy_pair demux_location = dispatch_core_manager::instance().demux_core(device_id, curr_channel, cq_id);
                CoreCoord phys_core = get_physical_core_coordinate(demux_location, dispatch_core_type);
                my_dispatch_cores[demux_location.chip].insert(phys_core);
                log_debug(tt::LogMetal, "Remote Device Demux core: Logical: {} - Physical: {} will be reset on MMIO Device.", demux_location.str(), phys_core.str());
            }
                CoreCoord phys_core;
                tt_cxy_pair dispatch_location = dispatch_core_manager::instance().dispatcher_d_core(device_id, curr_channel, cq_id);
                phys_core = get_physical_core_coordinate(dispatch_location, dispatch_core_type);
                my_dispatch_cores[dispatch_location.chip].insert(phys_core);
                tt_cxy_pair prefetch_location = dispatch_core_manager::instance().prefetcher_d_core(device_id, curr_channel, cq_id);
                phys_core = get_physical_core_coordinate(prefetch_location, dispatch_core_type);
                my_dispatch_cores[dispatch_location.chip].insert(phys_core);
                tt_cxy_pair mux_location = dispatch_core_manager::instance().mux_d_core(device_id, curr_channel, cq_id);
                phys_core = get_physical_core_coordinate(mux_location, dispatch_core_type);
                my_dispatch_cores[dispatch_location.chip].insert(phys_core);
                tt_cxy_pair demux_location = dispatch_core_manager::instance().demux_d_core(device_id, curr_channel, cq_id);
                phys_core = get_physical_core_coordinate(demux_location, dispatch_core_type);
                my_dispatch_cores[dispatch_location.chip].insert(phys_core);
        }
    }
}

void Device::initialize_cluster() {
    ZoneScoped;
    if (llrt::OptionsG.get_clear_l1()) {
        this->clear_l1_state();
    }
}

void Device::initialize_allocator(size_t l1_small_size, size_t trace_region_size, const std::vector<uint32_t> &l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(this->id_);
    // Construct allocator config from soc_desc
    AllocatorConfig config(
        {.num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_channels()),
         .dram_bank_size = soc_desc.dram_bank_size,
         .dram_bank_offsets = {},
         .worker_grid_size = this->logical_grid_size(),
         .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
         .l1_bank_size = static_cast<size_t>(get_l1_bank_size(id_, num_hw_cqs_, dispatch_core_type)),
         .l1_small_size = l1_small_size,
         .trace_region_size = trace_region_size,
         .core_type_from_noc_coord_table = {},  // Populated later
         .worker_log_to_physical_routing_x = soc_desc.worker_log_to_physical_routing_x,
         .worker_log_to_physical_routing_y = soc_desc.worker_log_to_physical_routing_y,
         .l1_bank_remap = l1_bank_remap,
         .compute_grid_size = this->compute_with_storage_grid_size()});
    TT_FATAL(config.l1_small_size < config.l1_bank_size, "Reserved size must be less than bank size");
    TT_FATAL(
        config.l1_small_size % ALLOCATOR_ALIGNMENT == 0,
        "Reserved size must be aligned to ALLOCATOR_ALIGNMENT {}",
        ALLOCATOR_ALIGNMENT);
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_channels(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const auto& core: soc_desc.physical_cores) {
        config.core_type_from_noc_coord_table.insert({core.first, AllocCoreType::Invalid});
    }

    for (const CoreCoord& core : tt::get_logical_compute_cores(id_, num_hw_cqs_, dispatch_core_type)) {
        this->compute_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const CoreCoord& core : tt::get_logical_storage_cores(id_, num_hw_cqs_, dispatch_core_type)) {
        this->storage_only_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const CoreCoord &core : tt::get_logical_dispatch_cores(id_, num_hw_cqs_, dispatch_core_type)) {
        const auto noc_coord = this->physical_core_from_logical_core(core, dispatch_core_type);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    for (const auto &core : soc_desc.get_logical_ethernet_cores()) {
        this->ethernet_cores_.insert(core);
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    TT_ASSERT(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    this->allocator_ = std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_build() {
    ZoneScoped;

    this->build_env_.init(this->build_key(), this->arch());

    auto init_helper = [this] (bool is_fw) -> JitBuildStateSet {
        std::vector<std::shared_ptr<JitBuildState>> build_states;

        build_states.resize(arch() == tt::ARCH::GRAYSKULL ? 5 : 7);

        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 0] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::DATA_MOVEMENT).first + 1] =
            std::make_shared<JitBuildDataMovement>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 0] =
            std::make_shared<JitBuildCompute>(this->build_env_, 0, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 1] =
            std::make_shared<JitBuildCompute>(this->build_env_, 1, is_fw);
        build_states[build_processor_type_to_index(JitBuildProcessorType::COMPUTE).first + 2] =
            std::make_shared<JitBuildCompute>(this->build_env_, 2, is_fw);

        if (arch() != tt::ARCH::GRAYSKULL) {
            build_states[build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 0] =
                std::make_shared<JitBuildEthernet>(this->build_env_, 0, is_fw);
            build_states[build_processor_type_to_index(JitBuildProcessorType::ETHERNET).first + 1] =
                std::make_shared<JitBuildEthernet>(this->build_env_, 1, is_fw);
        }

       return build_states;
    };

    this->firmware_build_states_ = init_helper(true);
    this->kernel_build_states_ = init_helper(false);
}

void Device::build_firmware() {
    log_debug(tt::LogMetal, "Building base firmware for device {}", this->id_);
    ZoneScoped;

    this->generate_device_headers(this->build_env_.get_out_firmware_root_path());
    jit_build_set(this->firmware_build_states_, nullptr, "");
}

void Device::initialize_firmware(CoreCoord phys_core, launch_msg_t *launch_msg) {
    ZoneScoped;

    if (llrt::is_ethernet_core(phys_core, this->id())) {
        // SKIP
    } else {
        llrt::program_risc_startup_addr(this->id(), phys_core);
        for (int riscv_id = 0; riscv_id < 5; riscv_id++) {
            ll_api::memory binary_mem =
                llrt::get_risc_binary(firmware_build_states_[riscv_id]->get_target_out_path(""));
            uint32_t kernel_size16 = llrt::get_binary_code_size16(binary_mem, riscv_id);
            if (riscv_id == 1) {
                launch_msg->kernel_config.ncrisc_kernel_size16 = kernel_size16;
            }
            log_debug(LogDevice, "RISC {} fw binary size: {} in bytes", riscv_id, kernel_size16 * 16);
            if (not llrt::OptionsG.get_skip_loading_fw()) {
                llrt::test_load_write_read_risc_binary(binary_mem, this->id(), phys_core, riscv_id);
            }
        }
    }
    //This is an initialization launch message.
    //Clears launch message fields to 0 in target core L1.
    //Sets launch.run to RUN_MSG_INIT.
    llrt::write_launch_msg_to_core(this->id(), phys_core, launch_msg,
        this->get_dev_addr(phys_core, HalMemAddrType::LAUNCH));
}

void Device::reset_cores() {
    ZoneScoped;

    auto kernel_still_running = [](launch_msg_t *launch_msg) {
        return launch_msg->go.run == RUN_MSG_GO && launch_msg->kernel_config.exit_erisc_kernel == 0;
    };

    auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id_);
    // Assert worker cores + dispatch cores, in case they were in a bad state from before.
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> dispatch_cores, other_dispatch_cores, device_to_early_exit_cores;
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord physical_core = this->ethernet_core_from_logical_core(eth_core);
        std::vector<uint32_t> data(sizeof(launch_msg_t) / sizeof(uint32_t));
        DeviceAddr launch_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalMemAddrType::LAUNCH);
        data = tt::llrt::read_hex_vec_from_core(
            this->id(), physical_core, launch_addr, sizeof(launch_msg_t));
        launch_msg_t *launch_msg = (launch_msg_t *)(&data[0]);
        if (kernel_still_running(launch_msg)) {
            log_info(
                tt::LogMetal,
                "While initializing Device {}, ethernet tunneler core {} on Device {} detected as still running, issuing exit signal.",
                this->id(),
                physical_core.str(),
                this->id());
            launch_msg->kernel_config.exit_erisc_kernel = 1;
            llrt::write_launch_msg_to_core(this->id(), physical_core, launch_msg, launch_addr, false);
            device_to_early_exit_cores[this->id()].insert(physical_core);
        }
    }

    this->get_associated_dispatch_phys_cores(dispatch_cores, other_dispatch_cores);
    // Ignore other_dispatch_cores, they will be reset by the devices that use them.
    for (auto &id_and_cores : dispatch_cores) {
        for (auto it = id_and_cores.second.begin(); it != id_and_cores.second.end(); it++) {
            const auto &phys_core = *it;
            // Only need to manually reset ethernet dispatch cores, tensix cores are all reset below.
            if (llrt::is_ethernet_core(phys_core, id_and_cores.first)) {
                // Ethernet cores won't be reset, so just signal the dispatch cores to early exit.
                std::vector<uint32_t> data(sizeof(launch_msg_t) / sizeof(uint32_t));
                DeviceAddr launch_addr = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalMemAddrType::LAUNCH);
                data = tt::llrt::read_hex_vec_from_core(
                    id_and_cores.first, phys_core, launch_addr, sizeof(launch_msg_t));
                launch_msg_t *launch_msg = (launch_msg_t *)(&data[0]);
                if (kernel_still_running(launch_msg)) {
                    log_info(
                        tt::LogMetal,
                        "While initializing device {}, ethernet dispatch core {} on Device {} detected as still running, issuing exit signal.",
                        this->id(),
                        phys_core.str(),
                        id_and_cores.first);
                    launch_msg->kernel_config.exit_erisc_kernel = 1;
                    llrt::write_launch_msg_to_core(id_and_cores.first, phys_core, launch_msg, launch_addr, false);
                    device_to_early_exit_cores[id_and_cores.first].insert(phys_core);
                }
            }
        }
    }

    // Early exiting dispatch cores should show RUN_MSG_DONE when they exit.
    for (auto &id_and_cores : device_to_early_exit_cores) {
        const int timeout_ms = 10000; // 10 seconds for now
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(id_and_cores.first, RUN_MSG_GO, id_and_cores.second, timeout_ms);
            } catch (std::runtime_error &) {
                TT_THROW("Device {} init: failed to reset cores! Try resetting the board.", this->id());
            }
        }
    }

    // Reset Tensix cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            // Don't reset dispatch cores for other devices, in case they're still running.
            if (other_dispatch_cores[this->id_].find(worker_core) == other_dispatch_cores[this->id_].end()) {
                if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
                }
            }
        }
    }
}

void Device::initialize_and_launch_firmware() {
    ZoneScoped;

    launch_msg_t launch_msg;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    launch_msg.kernel_config.mode = DISPATCH_MODE_HOST,
    launch_msg.go.run = RUN_MSG_INIT;

    // Populate core info, which will be written to device
    vector<uint32_t> core_info_vec(sizeof(core_info_msg_t) / sizeof(uint32_t));
    core_info_msg_t *core_info = (core_info_msg_t *) core_info_vec.data();

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(this->id());
    // unused in emulator context
    uint64_t pcie_chan_base_addr = 0;
    uint64_t pcie_chan_end_addr = 0;
    core_info->noc_pcie_addr_base = pcie_chan_base_addr;
    core_info->noc_pcie_addr_end = pcie_chan_end_addr;
    core_info->noc_dram_addr_base = 0;
    core_info->noc_dram_addr_end = soc_d.dram_core_size;

    const std::vector<CoreCoord> &pcie_cores = soc_d.get_pcie_cores();
    const std::vector<CoreCoord> &dram_cores = soc_d.get_dram_cores();
    const std::vector<CoreCoord> &eth_cores = soc_d.get_physical_ethernet_cores();
    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= MAX_NON_WORKER_CORES,
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    for (int idx = 0; idx < MAX_NON_WORKER_CORES; idx++) {
        core_info->non_worker_cores[idx] = {CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }
    int non_worker_cores_idx = 0;
    for (const CoreCoord &core : pcie_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = 
            {uint8_t(core.x), uint8_t(core.y), AddressableCoreType::PCIE};
    }
    for (const CoreCoord &core : dram_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = 
            {uint8_t(core.x), uint8_t(core.y), AddressableCoreType::DRAM};
    }
    for (const CoreCoord &core : eth_cores) {
        core_info->non_worker_cores[non_worker_cores_idx++] = 
            {uint8_t(core.x), uint8_t(core.y), AddressableCoreType::ETH};
    }

    // Determine which noc-coords are harvested
    // TODO(PGK/Almeet): fix this w/ new UMD
    vector<uint32_t> harvested_rows;
    uint32_t harvested_noc_rows = tt::Cluster::instance().get_harvested_rows(this->id());
    for (uint32_t y = 0; y < soc_d.grid_size.y; y++) {
        bool row_harvested = (harvested_noc_rows >> y) & 0x1;
        if (row_harvested) {
            harvested_rows.push_back(y);
        }
    }
    TT_ASSERT(harvested_rows.size() <= MAX_HARVESTED_ROWS, "Detected more harvested rows than fit in mailbox.");
    for (int idx = 0; idx < MAX_HARVESTED_ROWS; idx++) {
        core_info->harvested_y[idx] = (idx < harvested_rows.size()) ? harvested_rows[idx] : CORE_COORD_INVALID;
    }

    core_info->noc_size_x = soc_d.grid_size.x;
    core_info->noc_size_y = soc_d.grid_size.y;

    // Download to worker cores
    log_debug("Initializing firmware");
    CoreCoord grid_size = this->logical_grid_size();
    std::unordered_set<CoreCoord> not_done_cores;

    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
#if 0 // Temporary patch: allow storage-only cores (need regular solution later)
            if (!this->storage_only_cores_.count(logical_core)) {
#endif
                CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

                tt::llrt::write_hex_vec_to_core(
                    this->id(), worker_core, core_info_vec, this->get_dev_addr(worker_core, HalMemAddrType::CORE_INFO));
                this->initialize_firmware(worker_core, &launch_msg);
                not_done_cores.insert(worker_core);
#if 0 // See above
            }
#endif
        }
    }

    // Barrier between L1 writes above and deassert below
    tt::Cluster::instance().l1_barrier(this->id());

    // Deassert worker cores
    for(const auto& worker_core : not_done_cores)
        tt::Cluster::instance().deassert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug("Waiting for firmware init complete");
    llrt::internal_::wait_until_cores_done(this->id(), RUN_MSG_INIT, not_done_cores);
    log_debug("Firmware init complete");
}

void Device::clear_l1_state() {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", this->id_);
    // Clear all clearable Tensix and Eth L1
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(this->l1_size_per_core() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(this->l1_size_per_core() / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            detail::WriteToDeviceL1(this, logical_core, start_address, zero_vec);
        }
    }
}

void Device::update_dispatch_cores_for_multi_cq_eth_dispatch() {
    // When running Multiple CQs using Ethernet Dispatch, we may need more dispatch cores than those allocated in the
    // core descriptor (ex: 2 CQs on N300 need 10 dispatch cores and the core descriptor only allocates 6).
    // Infer the remaining dispatch cores from the idle eth core list (this is device dependent).
    if (dispatch_core_manager::instance().get_dispatch_core_type(this->id()) == CoreType::ETH) {
        auto& dispatch_core_manager = dispatch_core_manager::instance();
        for (const auto& idle_eth_core : this->get_inactive_ethernet_cores()) {
            dispatch_core_manager.add_dispatch_core_to_device(this->id(), idle_eth_core);
        }
    }
}

void Device::init_command_queue_host() {
    using_fast_dispatch = true;
    // TODO: Pass Prefetch API to CQManager constructor
    this->cq_manager_ = 
        std::make_unique<CQManager>(
            tt::Cluster::instance().get_command_processor(this->id()), 
            this->num_hw_cqs());
    hw_command_queues_.resize(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        hw_command_queues_[cq_id] = std::make_unique<HWCommandQueue>(this, cq_id, NOC::NOC_0);
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
    }
}

void Device::init_command_queue_device() {
    // SKIP
}

void Device::initialize_synchronous_sw_cmd_queue() {
    // Initialize a single Software Command Queue for SD, using passthrough mode.
    // This queue is used for all host bound functions using the Software CQ in SD mode.
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        // Need to do this since CommandQueue constructor is private
        sw_command_queues_.push_back(std::unique_ptr<CommandQueue>(new CommandQueue(this, cq_id)));
        sw_command_queues_[cq_id]->set_mode(CommandQueue::CommandQueueMode::PASSTHROUGH);
    }
}

bool Device::initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, const std::vector<uint32_t> &l1_bank_remap, bool minimal) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}. Program cache is {}enabled", this->id_, this->program_cache.is_enabled() ? "": "NOT ");
    log_debug(tt::LogMetal, "Running with {} cqs ", num_hw_cqs);
    TT_FATAL(num_hw_cqs > 0 and num_hw_cqs <= dispatch_core_manager::MAX_NUM_HW_CQS, "num_hw_cqs can be between 1 and {}", dispatch_core_manager::MAX_NUM_HW_CQS);
    hal.initialize(this->arch());
    this->using_fast_dispatch = false;
    this->num_hw_cqs_ = num_hw_cqs;
    constexpr uint32_t harvesting_map_bits = 12;
    this->build_key_ = ((uint32_t)this->num_hw_cqs_ << harvesting_map_bits) | tt::Cluster::instance().get_harvesting_mask(this->id());
    this->initialize_cluster();
    this->initialize_allocator(l1_small_size, trace_region_size, l1_bank_remap);
    this->initialize_build();
    // For minimal setup, don't initialize FW, watcher, dprint. They won't work if we're attaching to a hung chip.
    if (minimal)
        return true;

    // Mark initialized before compiling and sending dispatch kernels to device because compilation expects device to be initialized
    this->work_executor.initialize();
    this->initialized_ = true;

    return true;
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }

    for (const std::unique_ptr<HWCommandQueue> &hw_command_queue : hw_command_queues_) {
        if (hw_command_queue->cq_manager->get_bypass_mode()) {
            hw_command_queue->record_end();
        }
        hw_command_queue->terminate();
    }
    this->work_executor.reset();
    tt_metal::detail::DumpDeviceProfileResults(this, true);

    this->trace_buffer_pool_.clear();
    this->EnableAllocs();

    this->deallocate_buffers();

    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> not_done_dispatch_cores;
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> cores_to_skip;
    this->get_associated_dispatch_phys_cores(not_done_dispatch_cores, cores_to_skip);

    auto mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->id_);
    std::unordered_set<CoreCoord> wait_for_cores = not_done_dispatch_cores[mmio_device_id];

    llrt::internal_::wait_until_cores_done(mmio_device_id, RUN_MSG_GO, wait_for_cores);

    // Assert worker cores
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (cores_to_skip[mmio_device_id].find(worker_core) == cores_to_skip[mmio_device_id].end()) {
                if (this->storage_only_cores_.find(logical_core) == this->storage_only_cores_.end()) {
                    tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(this->id(), worker_core));
                }
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), this->id());
            }
        }
    }

    if (this->id_ != mmio_device_id) {
        for (auto it = not_done_dispatch_cores[mmio_device_id].begin(); it != not_done_dispatch_cores[mmio_device_id].end(); it++) {
            const auto &phys_core = *it;
            if(llrt::is_ethernet_core(phys_core, this->id_)) {
                log_debug(tt::LogMetal, "Ethernet dispatch core {} on Device {} is idle. Closing Device {}", phys_core.str(), mmio_device_id, this->id());
            } else {
                log_debug(tt::LogMetal, "Resetting core {} on Device {} when closing Device {}", phys_core.str(), mmio_device_id, this->id());
                tt::Cluster::instance().assert_risc_reset_at_core(tt_cxy_pair(mmio_device_id, phys_core));
            }
        }
    }

    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);

    tt::Cluster::instance().l1_barrier(id_);
    allocator::clear(*this->allocator_);
    // After device close, no buffers on this device should be used
    for (const auto &[buf_attr, buf] : detail::BUFFER_MAP.value()) {
        if (std::get<0>(buf_attr) == this->id()) {
            DeallocateBuffer(*buf);
        }
    }

    this->compute_cores_.clear();
    this->storage_only_cores_.clear();
    this->ethernet_cores_.clear();
    this->disable_and_clear_program_cache();
    this->command_queue_programs.clear();
    this->sw_command_queues_.clear();
    this->hw_command_queues_.clear();
    this->allocator_.reset();
    this->tunnel_device_dispatch_workers_.clear();
    this->initialized_ = false;

    return true;
}

Device::~Device() {
    log_debug(tt::LogMetal, "Device {} destructor", this->id_);
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const {
    return tt::Cluster::instance().arch();
}

int Device::num_dram_channels() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_num_dram_channels();
}

uint32_t Device::l1_size_per_core() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return tt::Cluster::instance().get_soc_desc(id_).dram_bank_size;
}

CoreCoord Device::grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).grid_size;
}

CoreCoord Device::logical_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).worker_grid_size;
}

CoreCoord Device::compute_with_storage_grid_size() const {
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(id_);
    return tt::get_compute_grid_size(id_, num_hw_cqs_, dispatch_core_type);
}

CoreCoord Device::dram_grid_size() const {
    return tt::Cluster::instance().get_soc_desc(id_).get_dram_grid_size();
}

CoreCoord Device::physical_core_from_logical_core(const CoreCoord &logical_coord, const CoreType &core_type) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_core_from_logical_core(logical_coord, core_type);
}

CoreType Device::core_type_from_physical_core(const CoreCoord &physical_coord) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    if (soc_desc.physical_cores.find(physical_coord) == soc_desc.physical_cores.end())
        TT_THROW("Physical core {} doesn't exist in metal_SocDescriptor.", physical_coord);

    return soc_desc.physical_cores.at(physical_coord).type;
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        worker_cores[idx] = worker_core_from_logical_core(logical_cores[idx]);

    return worker_cores;
}

CoreCoord Device::dram_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_physical_dram_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::dram_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> dram_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        dram_cores[idx] = dram_core_from_logical_core(logical_cores[idx]);

    return dram_cores;
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    return tt::Cluster::instance().ethernet_core_from_logical_core(id_, logical_core);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord &physical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return soc_desc.get_logical_ethernet_core_from_physical(physical_core);
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> ethernet_cores(logical_cores.size());

    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        ethernet_cores[idx] = ethernet_core_from_logical_core(logical_cores[idx]);
    return ethernet_cores;
}

uint32_t Device::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& physical_core) const {
    const auto& grid_size = this->grid_size();
    return NOC_XY_ENCODING(
        NOC_0_X(noc_index, grid_size.x, physical_core.x),
        NOC_0_Y(noc_index, grid_size.y, physical_core.y)
    );
}

uint32_t Device::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& physical_cores) const {
    const auto& grid_size = this->grid_size();

    // NOC 1 mcasts from bottom left to top right, so we need to reverse the coords
    if (noc_index == 0) {
        return NOC_MULTICAST_ENCODING(
            NOC_0_X(noc_index, grid_size.x, physical_cores.start_coord.x),
            NOC_0_Y(noc_index, grid_size.y, physical_cores.start_coord.y),
            NOC_0_X(noc_index, grid_size.x, physical_cores.end_coord.x),
            NOC_0_Y(noc_index, grid_size.y, physical_cores.end_coord.y)
        );
    } else {
        return NOC_MULTICAST_ENCODING(
            NOC_0_X(noc_index, grid_size.x, physical_cores.end_coord.x),
            NOC_0_Y(noc_index, grid_size.y, physical_cores.end_coord.y),
            NOC_0_X(noc_index, grid_size.x, physical_cores.start_coord.x),
            NOC_0_Y(noc_index, grid_size.y, physical_cores.start_coord.y)
        );
    }
}

void Device::check_allocator_is_initialized() const {
    if (this->allocator_ == nullptr) {
        TT_THROW("No memory allocator! Device has not been initialized, did you forget to call InitializeDevice?");
    }
}

uint32_t Device::num_banks(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::num_banks(*this->allocator_, buffer_type);
}

uint32_t Device::bank_size(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::bank_size(*this->allocator_, buffer_type);
}

uint32_t Device::dram_channel_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::dram_channel_from_bank_id(*this->allocator_, bank_id);
}

CoreCoord Device::dram_core_from_dram_channel(uint32_t dram_channel) const {
    return tt::Cluster::instance().get_soc_desc(id_).get_preferred_worker_core_for_dram_channel(dram_channel);
}

CoreCoord Device::logical_core_from_dram_channel(uint32_t dram_channel) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return tt::Cluster::instance().get_soc_desc(id_).get_logical_core_for_dram_channel(dram_channel);
}

uint32_t Device::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(this->id_);
    return tt::Cluster::instance().get_soc_desc(id_).get_dram_channel_from_logical_core(logical_core);
}

int32_t Device::bank_offset(BufferType buffer_type, uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::bank_offset(*this->allocator_, buffer_type, bank_id);
}

CoreCoord Device::logical_core_from_bank_id(uint32_t bank_id) const {
    this->check_allocator_is_initialized();
    return allocator::logical_core_from_bank_id(*this->allocator_, bank_id);
}

const std::vector<uint32_t> &Device::bank_ids_from_dram_channel(uint32_t dram_channel) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_dram_channel(*this->allocator_, dram_channel);
}

const std::vector<uint32_t> &Device::bank_ids_from_logical_core(
    BufferType buffer_type, const CoreCoord &logical_core) const {
    this->check_allocator_is_initialized();
    return allocator::bank_ids_from_logical_core(*this->allocator_, buffer_type, logical_core);
}

allocator::Statistics Device::get_memory_allocation_statistics(const BufferType &buffer_type) const {
    this->check_allocator_is_initialized();
    return allocator::get_statistics(*this->allocator_, buffer_type);
}

size_t Device::get_l1_small_size() const {
    this->check_allocator_is_initialized();
    return this->allocator_->config.l1_small_size;
}

void Device::dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const {
    this->check_allocator_is_initialized();
    return allocator::dump_memory_blocks(*this->allocator_, buffer_type, out);
}

void Device::deallocate_buffers(){
    allocator::deallocate_buffers(*allocator_);
}

float Device::sfpu_eps() const {
    switch (arch()) {
        case tt::ARCH::GRAYSKULL: return tt::tt_metal::EPS_GS;
        case tt::ARCH::WORMHOLE_B0: return tt::tt_metal::EPS_WHB0;
        case tt::ARCH::BLACKHOLE: return tt::tt_metal::EPS_BH;
        default: return std::numeric_limits<float>::epsilon();
    }

    return std::numeric_limits<float>::epsilon();
}

float Device::sfpu_nan() const {
    switch (arch()) {
        case tt::ARCH::GRAYSKULL: return tt::tt_metal::NAN_GS;
        case tt::ARCH::WORMHOLE_B0: return tt::tt_metal::NAN_WHB0;
        case tt::ARCH::BLACKHOLE: return tt::tt_metal::NAN_BH;
        default: return std::numeric_limits<float>::quiet_NaN();
    }

    return std::numeric_limits<float>::quiet_NaN();
}

// machine inf
float Device::sfpu_inf() const{

    switch (arch()) {
        case tt::ARCH::GRAYSKULL:
            return tt::tt_metal::INF_GS;
        case tt::ARCH::WORMHOLE_B0:
            return tt::tt_metal::INF_WHB0;
        case tt::ARCH::BLACKHOLE:
            return tt::tt_metal::INF_BH;
        default:
            return std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::infinity();
}

pair<int, int> Device::build_processor_type_to_index(JitBuildProcessorType t) const {
    constexpr int DataMovementBuildCount = 2;
    constexpr int ComputeBuildCount = 3;
    constexpr int EthernetBuildCount = 2;

    switch (t) {
    case JitBuildProcessorType::DATA_MOVEMENT: return pair<int, int>(0, DataMovementBuildCount);
    case JitBuildProcessorType::COMPUTE: return pair<int, int>(DataMovementBuildCount, ComputeBuildCount);
    case JitBuildProcessorType::ETHERNET: return pair<int, int>(DataMovementBuildCount + ComputeBuildCount, EthernetBuildCount);
    default: TT_THROW("Bad processor type: {}", static_cast<std::underlying_type<JitBuildProcessorType>::type>(t));
    }

    // shh the warnings
    return pair<int, int>(0, 0);
}

// Ideally the firmware getter would be private to the device, however, tests look for this
const JitBuildState& Device::build_firmware_state(JitBuildProcessorType t, int i) const {
    return *(this->firmware_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildState& Device::build_kernel_state(JitBuildProcessorType t, int i) const {
    return *(this->kernel_build_states_[build_processor_type_to_index(t).first + i]);
}

const JitBuildStateSubset Device::build_kernel_states(JitBuildProcessorType t) const {
    pair<int, int> bptti = build_processor_type_to_index(t);
    JitBuildStateSubset subset = {
        &this->kernel_build_states_[bptti.first],
        bptti.second
    };
    return subset;
}

const string Device::build_firmware_target_path(JitBuildProcessorType t, int i) const {
    const JitBuildState& bs = build_firmware_state(t, i);
    return bs.get_target_out_path("");
}

const string Device::build_kernel_target_path(JitBuildProcessorType t, int i, const string& kernel_name) const {
    const JitBuildState& bs = build_kernel_state(t, i);
    return bs.get_target_out_path(kernel_name);
}

HWCommandQueue& Device::hw_command_queue(size_t cq_id) {
    detail::DispatchStateCheck(true);
    TT_FATAL( cq_id < hw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *hw_command_queues_[cq_id];
}

CommandQueue &Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch);
    TT_FATAL( cq_id < sw_command_queues_.size(), "cq_id {} is out of range", cq_id );
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *sw_command_queues_[cq_id];
}

void Device::push_work(std::function<void()>&& work, bool blocking) {
    this->work_executor.push_work(work, blocking);
}

void Device::push_work(std::shared_ptr<std::function<void()>> work, bool blocking) {
    this->work_executor.push_work(work, blocking);
}

void Device::synchronize() {
    this->work_executor.synchronize();
}

void Device::set_worker_mode(const WorkExecutorMode& mode) {
    this->work_executor.set_worker_mode(mode);
}

void Device::enable_async(bool enable) {
    auto mode = enable ? WorkExecutorMode::ASYNCHRONOUS : WorkExecutorMode::SYNCHRONOUS;
    this->set_worker_mode(mode);
}

bool Device::using_slow_dispatch() const {
    return not (this->using_fast_dispatch);
}

void Device::begin_trace(const uint8_t cq_id, const uint32_t tid) {
    TT_FATAL(this->trace_buffer_pool_.count(tid) == 0, "Trace already exists for tid {} on device", tid);
    TT_FATAL(!this->hw_command_queues_[cq_id]->tid.has_value(), "CQ {} is already being used for tracing tid {}", (uint32_t)cq_id, tid);
    this->EnableAllocs();
    // Create an empty trace buffer here. This will get initialized in end_trace
    this->trace_buffer_pool_.insert({tid, Trace::create_empty_trace_buffer()});
    this->hw_command_queues_[cq_id]->record_begin(tid, this->trace_buffer_pool_[tid]->desc);
}

void Device::end_trace(const uint8_t cq_id, const uint32_t tid) {
    TT_FATAL(this->hw_command_queues_[cq_id]->tid == tid, "CQ {} is not being used for tracing tid {}", (uint32_t)cq_id, tid);
    TT_FATAL(this->trace_buffer_pool_.count(tid) > 0, "Trace instance {} must exist on device", tid);
    this->hw_command_queues_[cq_id]->record_end();
    auto &trace_data = this->trace_buffer_pool_[tid]->desc->data;
    trace_data = std::move(this->cq_manager()->get_bypass_data());
    // Add command to terminate the trace buffer
    DeviceCommand command_sequence(CQ_PREFETCH_CMD_BARE_MIN_SIZE);
    command_sequence.add_prefetch_exec_buf_end();
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        trace_data.push_back(((uint32_t*)command_sequence.data())[i]);
    }
    Trace::initialize_buffer(this->command_queue(cq_id), this->trace_buffer_pool_[tid]);
    this->DisableAllocs();
}

void Device::replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking) {
    constexpr bool check = false;
    TT_FATAL(this->trace_buffer_pool_.count(tid) > 0, "Trace instance {}  must exist on device" , tid);
    if constexpr (check) {
        Trace::validate_instance(*this->trace_buffer_pool_[tid]);
    }
    this->command_queue(cq_id).run_command(CommandInterface{
        .type = EnqueueCommandType::ENQUEUE_TRACE,
        .blocking = blocking,
        .trace_id = tid
    });
}

void Device::release_trace(const uint32_t tid) {
    uint32_t erased = this->trace_buffer_pool_.erase(tid);
    // Only enable allocations once all captured traces are released
    if (this->trace_buffer_pool_.empty()) {
        this->EnableAllocs();
    }
}

std::shared_ptr<TraceBuffer> Device::get_trace(const uint32_t tid) {
    if (auto trace = this->trace_buffer_pool_.find(tid); trace != this->trace_buffer_pool_.end()) {
        return trace->second;
    } else {
        return nullptr;
    }
}

void Device::DisableAllocs() {
    tt::tt_metal::allocator::disable_allocs(*(this->allocator_));
}

void Device::EnableAllocs() {
    tt::tt_metal::allocator::enable_allocs(*(this->allocator_));
}

void Device::generate_device_headers(const std::string &path) const
{

    // Basic Allocator generates number of banks which may not be power of 2, so we could just pad and alias for now
    const size_t num_dram_banks = this->num_banks(BufferType::DRAM);
    const size_t num_dram_banks_pow2 = std::pow(2, std::ceil(std::log2(num_dram_banks)));
    std::vector<CoreCoord> dram_noc_coord_per_bank(num_dram_banks);
    std::vector<int32_t> dram_offsets_per_bank(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        dram_noc_coord_per_bank[bank_id] = this->dram_core_from_dram_channel(this->dram_channel_from_bank_id(bank_id));
        dram_offsets_per_bank[bank_id] = this->bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = this->num_banks(BufferType::L1); // 128
    const size_t num_l1_banks_pow2 = std::pow(2, std::ceil(std::log2(num_l1_banks)));
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    std::vector<int32_t> l1_offset_per_bank(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] = this->worker_core_from_logical_core(this->logical_core_from_bank_id(bank_id));
        l1_offset_per_bank[bank_id] = this->bank_offset(BufferType::L1, bank_id);
    }

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(this->id());

    // Generate header file in proper location
    jit_build_genfiles_bank_to_noc_coord_descriptor (
        path,
        soc_d.grid_size,
        dram_noc_coord_per_bank,
        dram_offsets_per_bank,
        l1_noc_coord_per_bank,
        l1_offset_per_bank,
        soc_d.profiler_ceiled_core_count_perf_dram_bank,
        soc_d.physical_routing_to_profiler_flat_id
    );
}

}  // namespace tt_metal

}  // namespace tt
