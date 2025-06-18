// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <memory>

#include "device/api/device_api.hpp"

#include "tt_metal/third_party/umd/device/tt_cluster_descriptor_types.h"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor.h"
#include "tt_metal/third_party/umd/device/tt_device.h"
#include "tt_metal/emulator/tt_emulator_device.h"

namespace {

template <typename T>
void size_buffer_to_capacity(std::vector<T> &data_buf, size_t size_in_bytes) {
    size_t target_size = 0;
    if (size_in_bytes > 0) {
        target_size = ((size_in_bytes - 1) / sizeof(T)) + 1;
    }
    data_buf.resize(target_size);
} 

} // namespace

//
//    CommandProcessorImpl
//

CommandProcessorImpl::CommandProcessorImpl(tt::metal::device::Device *device):
        m_device(device) { }

CommandProcessorImpl::~CommandProcessorImpl() { }

void CommandProcessorImpl::configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) {
    m_device->configure_read_buffer(padded_page_size, dst, dst_offset, num_pages_read);
}

void CommandProcessorImpl::run_commands(const uint8_t *cmd_reg, uint32_t cmd_seq_size) {
    m_device->run_commands(cmd_reg, cmd_seq_size);
}

void CommandProcessorImpl::launch_kernels() {
    m_device->launch_kernels();
}

//
//    tt_EmulatorDevice
//

tt_EmulatorDevice::tt_EmulatorDevice(
        const std::string &sdesc_path, const std::string &ndesc_path):
            tt_device(sdesc_path) {
    tt_device::soc_descriptor_per_chip.emplace(0, tt_SocDescriptor(sdesc_path));
    std::set<chip_id_t> target_devices = {0};
    if (ndesc_path == "") {
        m_ndesc = tt_ClusterDescriptor::create_for_grayskull_cluster(target_devices, {});
    } else {
        m_ndesc = tt_ClusterDescriptor::create_from_yaml(ndesc_path);
    } 
    tt_SocDescriptor &soc_descriptor = soc_descriptor_per_chip.begin()->second;
    using tt::metal::device::Device;
    Device::Arch arch;
    switch (soc_descriptor.arch) {
    case tt::ARCH::GRAYSKULL:
        arch = Device::Arch::GRAYSKULL;
        break;
    case tt::ARCH::WORMHOLE_B0:
        arch = Device::Arch::WORMHOLE_B0;
        break;
    default:
        assert(false);
        break;
    }
    m_device.reset(Device::create(arch));
    m_command_processor.reset(new CommandProcessorImpl(m_device.get()));
}

tt_EmulatorDevice::~tt_EmulatorDevice() {
    m_ndesc.reset();
}

void tt_EmulatorDevice::set_device_l1_address_params(
        const tt_device_l1_address_params &l1_address_params_) {
    m_l1_address_params = l1_address_params_;
}

void tt_EmulatorDevice::set_device_dram_address_params(
        const tt_device_dram_address_params &dram_address_params_) {
    m_dram_address_params = dram_address_params_;
}

std::unordered_map<chip_id_t, tt_SocDescriptor> &
        tt_EmulatorDevice::get_virtual_soc_descriptors() {
    return tt_device::soc_descriptor_per_chip;
}

void tt_EmulatorDevice::start(
        std::vector<std::string> plusargs, 
        std::vector<std::string> dump_cores, 
        bool no_checkers, 
        bool init_device, 
        bool skip_driver_allocs) {
    m_device->start();
}

void tt_EmulatorDevice::start_device(const tt_device_params &device_params) {
    bool no_checkers = true;
    // TODO: Figure out purpose of 'device_params.unroll_vcd_dump_cores()'
    //     (Is it Versim-specific?)
    std::vector<std::string> dump_cores = 
        device_params.unroll_vcd_dump_cores(get_soc_descriptor(0)->grid_size);
    start(
        device_params.expand_plusargs(), 
        dump_cores, 
        no_checkers, 
        device_params.init_device, 
        false); 
}

void tt_EmulatorDevice::close_device() {
    stop();
}

void tt_EmulatorDevice::deassert_risc_reset() {
    // nothing to do so far
}

void tt_EmulatorDevice::deassert_risc_reset_at_core(tt_cxy_pair core) {
    // This function deasserts reset on the full emulator device
    // (don't need core level granularity for emulator)
    deassert_risc_reset(); 
}

void tt_EmulatorDevice::assert_risc_reset() {
    // nothing to do so far
}

void tt_EmulatorDevice::assert_risc_reset_at_core(tt_cxy_pair core) {
    // This function asserts reset on the full emulator device
    // (don't need core level granularity for emulator)
    assert_risc_reset(); 
}

#if 0 // TODO: Revise this
void tt_EmulatorDevice::write_to_device(
        std::vector<uint32_t> &vec, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use, 
        bool send_epoch_cmd/* = false*/, 
        bool last_send_epoch_cmd/* = true*/, 
        bool ordered_with_prev_remote_write/* = false*/) {
    write_to_device(
        vec.data(), 
        vec.size() * sizeof(uint32_t), 
        core, 
        addr, 
        tlb_to_use, 
        send_epoch_cmd, 
        last_send_epoch_cmd); 
}
#endif

void tt_EmulatorDevice::write_to_device(
        std::vector<uint32_t> &vec, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use) {
    write_to_device(vec.data(), vec.size() * sizeof(uint32_t), core, addr, tlb_to_use); 
}

#if 0 // TODO: Revise this
void tt_EmulatorDevice::rolled_write_to_device(
        std::vector<uint32_t> &vec, 
        uint32_t unroll_count, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use) {
    uint32_t byte_increment = uint32_t(vec.size()) * 4; 
    for (uint32_t i = 0; i < unroll_count; i++) {
        vec[0] = i; // slot id for debug
        write_to_device(vec, core, addr + i * byte_increment, tlb_to_use);
    } 
}
#endif

void tt_EmulatorDevice::read_from_device(
        std::vector<uint32_t> &vec, 
        tt_cxy_pair core, 
        uint64_t addr, 
        uint32_t size, 
        const std::string &tlb_to_use) {
    size_buffer_to_capacity(vec, size);
    read_from_device(vec.data(), core, addr, size, tlb_to_use); 
}

#if 0 // TODO: Revise this
void tt_EmulatorDevice::rolled_write_to_device(
        uint32_t *mem_ptr, 
        uint32_t size_in_bytes, 
        uint32_t unroll_count, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &fallback_tlb) {
    // TODO: Figure out whether 'size_in_bytes' is actually len in 32-bit words
    std::vector<uint32_t> mem_vector(mem_ptr, mem_ptr + size_in_bytes);
    rolled_write_to_device(mem_vector, unroll_count, core, addr, fallback_tlb); 
}
#endif

#if 0 // TODO: Revise this
void tt_EmulatorDevice::write_to_device(
        const void *mem_ptr, 
        uint32_t size_in_bytes, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use, 
        bool send_epoch_cmd/* = false*/, 
        bool last_send_epoch_cmd/* = true*/, 
        bool ordered_with_prev_remote_write/* = false*/) {
    // ignore 'tlb_to_use', 'send_epoch_cmd', and 'last_send_epoch_cmd'
    m_device->write(mem_ptr, size_in_bytes, int(core.x), int(core.y), addr);
}
#endif

void tt_EmulatorDevice::write_to_device(
        const void *mem_ptr, 
        uint32_t size_in_bytes, 
        tt_cxy_pair core, 
        uint64_t addr, 
        const std::string &tlb_to_use) {
    // ignore 'tlb_to_use', 'send_epoch_cmd', and 'last_send_epoch_cmd'
    m_device->write(mem_ptr, size_in_bytes, int(core.x), int(core.y), addr);
}

void tt_EmulatorDevice::read_from_device(
        void *mem_ptr, 
        tt_cxy_pair core, 
        uint64_t addr, 
        uint32_t size, 
        const std::string &tlb_to_use) { 
    // ignore 'tlb_to_use'
    m_device->read(mem_ptr, size, int(core.x), int(core.y), addr);
}

void tt_EmulatorDevice::write_to_sysmem(
        std::vector<uint32_t> &vec, 
        uint64_t addr, 
        uint16_t channel, 
        chip_id_t src_device_id) {
    uint32_t size = uint32_t(vec.size() * sizeof(uint32_t));
    write_to_sysmem(vec.data(), size, addr, channel, src_device_id);
}

void tt_EmulatorDevice::write_to_sysmem(
        const void *mem_ptr, 
        std::uint32_t size,  
        uint64_t addr, 
        uint16_t channel, 
        chip_id_t src_device_id) {
    // ignore 'channel' and 'src_device_id'
    m_device->write_to_sysmem(mem_ptr, size, addr);
}

void tt_EmulatorDevice::read_from_sysmem(
        std::vector<uint32_t> &vec, 
        uint64_t addr, 
        uint16_t channel, 
        uint32_t size, 
        chip_id_t src_device_id) {
    size_buffer_to_capacity(vec, size);
    read_from_sysmem(vec.data(), addr, channel, size, src_device_id); 
}

void tt_EmulatorDevice::read_from_sysmem(
        void *mem_ptr, 
        uint64_t addr, 
        uint16_t channel, 
        uint32_t size, 
        chip_id_t src_device_id) {
    // ignore 'channel' and 'src_device_id'
    m_device->read_from_sysmem(mem_ptr, size, addr);
}

void tt_EmulatorDevice::wait_for_non_mmio_flush() {
    // nothing to do
}

void tt_EmulatorDevice::l1_membar(
        const chip_id_t chip, 
        const std::string &fallback_tlb, 
        const std::unordered_set<tt_xy_pair> &cores/* = { }*/) {
    // nothing to do (emulator does not reorder loads/stores)
}

void tt_EmulatorDevice::dram_membar(
        const chip_id_t chip, 
        const std::string &fallback_tlb, 
        const std::unordered_set<uint32_t> &channels) {
    // nothing to do (emulator does not reorder loads/stores)
}

void tt_EmulatorDevice::dram_membar(
        const chip_id_t chip, 
        const std::string &fallback_tlb, 
        const std::unordered_set<tt_xy_pair> &cores/* = { }*/) {
    // nothing to do (emulator does not reorder loads/stores)
}

void tt_EmulatorDevice::translate_to_noc_table_coords(
        chip_id_t device_id, 
        std::size_t &r, 
        std::size_t &c) {
    // nothing to do (no translation is performed)
}

bool tt_EmulatorDevice::using_harvested_soc_descriptors() {
    return false;
}

std::unordered_map<chip_id_t, uint32_t> 
        tt_EmulatorDevice::get_harvesting_masks_for_soc_descriptors() {
    return {{0, 0}};
}

#if 0 // TODO: Revise this
bool tt_EmulatorDevice::noc_translation_en() {
    return false;
}
#endif

std::set<chip_id_t> tt_EmulatorDevice::get_target_mmio_device_ids() {
    // nothing to do (must only be used for silicon)
    return { };
}

std::set<chip_id_t> tt_EmulatorDevice::get_target_remote_device_ids() {
    // nothing to do (must only be used for silicon)
    return { };
}

tt_ClusterDescriptor *tt_EmulatorDevice::get_cluster_description() {
    return m_ndesc.get();
}

int tt_EmulatorDevice::get_number_of_chips_in_cluster() {
    return detect_number_of_chips();
}

std::unordered_set<chip_id_t> tt_EmulatorDevice::get_all_chips_in_cluster() {
    return {0};
}

int tt_EmulatorDevice::detect_number_of_chips() {
    return 1;
}

std::map<int, int> tt_EmulatorDevice::get_clocks() {
      return std::map<int, int>();
}

std::uint32_t tt_EmulatorDevice::get_num_dram_channels(std::uint32_t device_id) {
    return get_soc_descriptor(device_id)->get_num_dram_channels();
}

std::uint64_t tt_EmulatorDevice::get_dram_channel_size(
        std::uint32_t device_id, 
        std::uint32_t channel) {
    // Space per channel is identical for now
    return get_soc_descriptor(device_id)->dram_bank_size;
}

std::uint32_t tt_EmulatorDevice::get_num_host_channels(std::uint32_t device_id) {
    // host buffers not allocated for emulator devices
    return 0; 
}

std::uint32_t tt_EmulatorDevice::get_host_channel_size(
        std::uint32_t device_id, 
        std::uint32_t channel) {
    // host buffers not allocated for emulator devices
    return 0; 
}

void *tt_EmulatorDevice::host_dma_address(
        std::uint64_t offset, 
        chip_id_t src_device_id, 
        uint16_t channel) const {
    // ignore 'src_device_id' and 'channel'
    return m_device->host_dma_address(offset);
}

CommandProcessor *tt_EmulatorDevice::get_command_processor() const {
    return m_command_processor.get();
}

bool tt_EmulatorDevice::stop() {
    m_device->stop();
    return true;
}

