// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace riscv {
namespace linker {

class Linker {
public:
    Linker() { }
    virtual ~Linker() { }
public:
    static Linker *create();
    virtual void add_builtin(const std::string &name, uint64_t value) = 0;
    virtual void link(
        const std::string &fname, 
        uint64_t code_base,
        std::vector<uint8_t> &result,
        uint64_t &start_pc) = 0;
};

} // linker
} // riscv

