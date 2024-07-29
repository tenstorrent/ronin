// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "third_party/umd/device/device_api.h"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/common/metal_soc_descriptor.h"

#include <unordered_map>

namespace ll_api {

void configure_static_tlbs(tt::ARCH arch, chip_id_t mmio_device_id, const metal_SocDescriptor &sdesc, tt_device &device_driver);

std::unordered_map<std::string, std::int32_t> get_dynamic_tlb_config();

}  // namespace ll_api
