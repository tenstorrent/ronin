#pragma once

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "interp/Memory.hpp"
#include "interp/Hart.hpp"

#include "riscv/riscv32.hpp"

namespace riscv {
namespace core {

using WdRiscv::Memory;
using Hart32 = WdRiscv::Hart<uint32_t>;

class Riscv32CoreImpl: public Riscv32Core, public Hart32 {
public:
    Riscv32CoreImpl(int id, Memory *memory);
    ~Riscv32CoreImpl();
public:
    void set_builtin_handler(Riscv32BuiltinHandler *builtin_handler) override;
    void set_memory_layout(
        uint32_t code_base,
        uint32_t code_size,
        uint32_t local_base,
        uint32_t local_size) override;
    uint32_t code_base() override {
        return m_code_base;
    }
    uint32_t code_size() override {
        return m_code_size;
    }
    uint32_t local_base() override {
        return m_local_base;
    }
    uint32_t local_size() override {
        return m_local_size;
    }
    void write_code(uint32_t addr, const std::vector<uint8_t> &code) override; 
    void run(uint32_t start_pc) override;
    uint32_t get_arg(int index) override;
    void set_ret(int index, uint32_t value) override;
    uint8_t *map_addr(uint32_t addr) override;
protected:
    int execJalrHook(uint32_t pc) override;
private:
    void set_int_reg(const std::string &reg_name, uint32_t val);
private:
    Memory *m_memory;
    Riscv32BuiltinHandler *m_builtin_handler;
    uint32_t m_code_base;
    uint32_t m_code_size;
    uint32_t m_local_base;
    uint32_t m_local_size;
};

class Riscv32SystemImpl: public Riscv32System {
public:
    Riscv32SystemImpl(
        int core_count,
        uint32_t mem_size, 
        uint32_t page_size, 
        uint32_t region_size);
    ~Riscv32SystemImpl();
public:
    int core_count() override;
    Riscv32Core *core_at(int index) override;
    uint8_t *map_addr(uint32_t addr) override;
private:
    std::unique_ptr<Memory> m_memory;
    std::vector<std::unique_ptr<Riscv32CoreImpl>> m_cores;
};

} // namespace core
} // namespace riscv

