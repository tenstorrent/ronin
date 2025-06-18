// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace riscv {
namespace core {

class Riscv32Core;

class Riscv32BuiltinHandler {
public:
    Riscv32BuiltinHandler() { }
    virtual ~Riscv32BuiltinHandler() { }
public:
    virtual void call(Riscv32Core *core, int id) = 0;
};

class Riscv32Core {
public:
    Riscv32Core() { }
    virtual ~Riscv32Core() { }
public:
    virtual void set_builtin_handler(Riscv32BuiltinHandler *builtin_handler) = 0;
    virtual void set_memory_layout(
        uint32_t code_base,
        uint32_t code_size,
        uint32_t local_base,
        uint32_t local_size) = 0;
    virtual uint32_t code_base() = 0;
    virtual uint32_t code_size() = 0;
    virtual uint32_t local_base() = 0;
    virtual uint32_t local_size() = 0;
    virtual void write_code(uint32_t addr, const std::vector<uint8_t> &code) = 0; 
    virtual void run(uint32_t start_pc) = 0;
    virtual uint32_t get_arg(int index) = 0;
    virtual void set_ret(int index, uint32_t value) = 0;
    virtual uint8_t *map_addr(uint32_t addr) = 0;
};

class Riscv32System {
public:
    Riscv32System() { }
    virtual ~Riscv32System() { }
public:
    static Riscv32System *create(
        int core_count,
        uint32_t mem_size, 
        uint32_t page_size, 
        uint32_t region_size);
    virtual int core_count() = 0;
    virtual Riscv32Core *core_at(int index) = 0;
    virtual uint8_t *map_addr(uint32_t addr) = 0;
};

} // namespace core
} // namespace riscv

