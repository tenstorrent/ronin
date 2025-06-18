// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace tt {
namespace metal {
namespace device {

class KernelBuilder {
public:
    KernelBuilder() { }
    virtual ~KernelBuilder() { }
public:
    static KernelBuilder *create();
public:
    virtual void configure(
        const std::string &cpp_cmd_base,
        const std::vector<std::pair<std::string, std::string>> &prefix_map,
        const std::string &src_base_dir,
        const std::string &temp_dir) = 0;
    virtual void build(
        const std::string &name,
        bool is_compute,
        const std::string &defines,
        uint32_t code_base,
        std::vector<uint8_t> &code, 
        uint32_t &start_pc) = 0;
};

} // namespace device
} // namespace metal
} // namespace tt

