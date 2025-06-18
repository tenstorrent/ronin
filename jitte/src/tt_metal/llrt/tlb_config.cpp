// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tlb_config.hpp"

// This module is not used in emulator context
// TODO: Consider complete removal.

namespace ll_api {

void configure_static_tlbs(tt::ARCH arch, chip_id_t mmio_device_id, const metal_SocDescriptor &sdesc, tt_device &device_driver) {
    // SKIP
}

std::unordered_map<std::string, std::int32_t> get_dynamic_tlb_config(tt::ARCH arch) {
    // SKIP
    return {};
}

}  // namespace ll_api

