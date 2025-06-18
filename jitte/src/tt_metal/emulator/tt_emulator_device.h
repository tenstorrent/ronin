// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <memory>

#include "device/api/device_api.hpp"

#include "tt_metal/third_party/umd/device/command_processor.h"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor_types.h"
#include "tt_metal/third_party/umd/device/tt_device.h"

//
//    CommandProcessorImpl
//

class CommandProcessorImpl: public CommandProcessor {
public:
    CommandProcessorImpl(tt::metal::device::Device *device);
    ~CommandProcessorImpl();
public:
    void configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) override;
    void run_commands(const uint8_t *cmd_reg, uint32_t cmd_seq_size) override;
    void launch_kernels() override;
private:
    tt::metal::device::Device *m_device;
};

//
//    tt_EmulatorDevice
//

class tt_EmulatorDevice: public tt_device {
public:
    tt_EmulatorDevice(const std::string &sdesc_path, const std::string &ndesc_path);
    ~tt_EmulatorDevice();
public:
    void set_device_l1_address_params(
        const tt_device_l1_address_params &l1_address_params_) override;
    void set_device_dram_address_params(
        const tt_device_dram_address_params &dram_address_params_) override;
    std::unordered_map<chip_id_t, tt_SocDescriptor> &get_virtual_soc_descriptors() override;
    void start(
        std::vector<std::string> plusargs, 
        std::vector<std::string> dump_cores, 
        bool no_checkers, 
        bool init_device, 
        bool skip_driver_allocs);
    void start_device(const tt_device_params &device_params) override;
    void close_device() override;
    void deassert_risc_reset() override;
    void deassert_risc_reset_at_core(tt_cxy_pair core) override;
    void assert_risc_reset() override;
    void assert_risc_reset_at_core(tt_cxy_pair core) override;
#if 0 // TODO: Revise this
    void write_to_device(
        std::vector<uint32_t> &vec, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use, 
        bool send_epoch_cmd = false, 
        bool last_send_epoch_cmd = true, 
        bool ordered_with_prev_remote_write = false) override;
#endif
    void write_to_device(
        std::vector<uint32_t> &vec, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use) override;
#if 0 // TODO: Revise this
    void rolled_write_to_device(
        std::vector<uint32_t> &vec, 
        uint32_t unroll_count, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use) override;
#endif
    void read_from_device(
        std::vector<uint32_t> &vec, 
        tt_cxy_pair core, 
        uint64_t addr, 
        uint32_t size, 
        const std::string &tlb_to_use) override;
#if 0 // TODO: Revise this
    void rolled_write_to_device(
        uint32_t *mem_ptr, 
        uint32_t size_in_bytes, 
        uint32_t unroll_count, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &fallback_tlb) override;
#endif
#if 0 // TODO: Revise this
    void write_to_device(
        const void *mem_ptr, 
        uint32_t size_in_bytes, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use, 
        bool send_epoch_cmd = false, 
        bool last_send_epoch_cmd = true, 
        bool ordered_with_prev_remote_write = false) override;
#endif
    void write_to_device(
        const void *mem_ptr, 
        uint32_t size_in_bytes, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use) override;
    void read_from_device(
        void *mem_ptr, 
        tt_cxy_pair core, 
        uint64_t addr, 
        uint32_t size, 
        const std::string &tlb_to_use) override; 
    void write_to_sysmem(
        std::vector<uint32_t> &vec, 
        uint64_t addr, 
        uint16_t channel, 
        chip_id_t src_device_id) override;
    void write_to_sysmem(
        const void *mem_ptr, 
        std::uint32_t size,  
        uint64_t addr, 
        uint16_t channel, 
        chip_id_t src_device_id) override;
    void read_from_sysmem(
        std::vector<uint32_t> &vec, 
        uint64_t addr, 
        uint16_t channel, 
        uint32_t size, 
        chip_id_t src_device_id) override;
    void read_from_sysmem(
        void *mem_ptr, 
        uint64_t addr, 
        uint16_t channel, 
        uint32_t size, 
        chip_id_t src_device_id) override;
    void wait_for_non_mmio_flush() override;
    void l1_membar(
        const chip_id_t chip, 
        const std::string &fallback_tlb, 
        const std::unordered_set<tt_xy_pair> &cores = { }) override;
    void dram_membar(
        const chip_id_t chip, 
        const std::string &fallback_tlb, 
        const std::unordered_set<uint32_t> &channels) override;
    void dram_membar(
        const chip_id_t chip, 
        const std::string &fallback_tlb, 
        const std::unordered_set<tt_xy_pair> &cores = { }) override;
    void translate_to_noc_table_coords(
        chip_id_t device_id, 
        std::size_t &r, 
        std::size_t &c) override;
    bool using_harvested_soc_descriptors() override;
    std::unordered_map<chip_id_t, uint32_t> get_harvesting_masks_for_soc_descriptors() override;
#if 0 // TODO: Revise this
    bool noc_translation_en() override;
#endif
    std::set<chip_id_t> get_target_mmio_device_ids() override;
    std::set<chip_id_t> get_target_remote_device_ids() override;
    tt_ClusterDescriptor *get_cluster_description() override;
    int get_number_of_chips_in_cluster() override;
    std::unordered_set<chip_id_t> get_all_chips_in_cluster() override;
    static int detect_number_of_chips();
    std::map<int, int> get_clocks() override;
    std::uint32_t get_num_dram_channels(std::uint32_t device_id) override;
    std::uint64_t get_dram_channel_size(std::uint32_t device_id, std::uint32_t channel) override;
    std::uint32_t get_num_host_channels(std::uint32_t device_id) override;
    std::uint32_t get_host_channel_size(std::uint32_t device_id, std::uint32_t channel) override;
    void *host_dma_address(
        std::uint64_t offset, 
        chip_id_t src_device_id, 
        uint16_t channel) const override;
    CommandProcessor *get_command_processor() const override;
private:
    bool stop();
private:
    tt_device_l1_address_params m_l1_address_params;
    tt_device_dram_address_params m_dram_address_params;
    std::shared_ptr<tt_ClusterDescriptor> m_ndesc;
    std::unique_ptr<tt::metal::device::Device> m_device;
    std::unique_ptr<CommandProcessorImpl> m_command_processor;
};

