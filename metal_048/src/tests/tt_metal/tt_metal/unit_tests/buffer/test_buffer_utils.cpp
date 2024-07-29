// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "test_buffer_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace tt::test::buffer::detail {
void writeL1Backdoor(tt::tt_metal::Device* device, CoreCoord coord, uint32_t address, std::vector<uint32_t>& data) {
    tt::log_info("{} -- coord={} address={}", __FUNCTION__, coord.str(), address);
    tt_metal::detail::WriteToDeviceL1(device, coord, address, data);
}
void readL1Backdoor(
    tt::tt_metal::Device* device, CoreCoord coord, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    tt::log_info("{} -- coord={} address={} byte_size={}", __FUNCTION__, coord.str(), address, byte_size);
    tt_metal::detail::ReadFromDeviceL1(device, coord, address, byte_size, data);
}
void writeDramBackdoor(tt::tt_metal::Device* device, uint32_t channel, uint32_t address, std::vector<uint32_t>& data) {
    tt::log_info("{} -- channel={} address={}", __FUNCTION__, channel, address);
    tt_metal::detail::WriteToDeviceDRAMChannel(device, channel, address, data);
}
void readDramBackdoor(
    tt::tt_metal::Device* device, uint32_t channel, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    tt::log_info("{} -- channel={} address={} byte_size={}", __FUNCTION__, channel, address, byte_size);
    tt_metal::detail::ReadFromDeviceDRAMChannel(device, channel, address, byte_size, data);
}
}  // namespace tt::test::buffer::detail
