
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

#include "interp/port.hpp"
#include "interp/Hart.hpp"

#include "riscv/riscv32.hpp"
#include "riscv/riscv32impl.hpp"

namespace riscv {
namespace core {

using WdRiscv::Memory;
using WdRiscv::Hart;

//
//    Riscv32Core
//

Riscv32CoreImpl::Riscv32CoreImpl(int id, Memory *memory):
        Hart32(unsigned(id), uint32_t(id), *memory),
        m_memory(memory),
        m_builtin_handler(nullptr),
        m_code_base(0), 
        m_code_size(0), 
        m_local_base(0), 
        m_local_size(0) { }

Riscv32CoreImpl::~Riscv32CoreImpl() { }

void Riscv32CoreImpl::set_builtin_handler(Riscv32BuiltinHandler *builtin_handler) {
    m_builtin_handler = builtin_handler;
}

void Riscv32CoreImpl::set_memory_layout(
        uint32_t code_base,
        uint32_t code_size,
        uint32_t local_base,
        uint32_t local_size) {
    uint32_t memory_size = uint32_t(m_memory->size());
    if (!(code_base >= 0 && code_size > 0 && code_base + code_size <= memory_size)) {
        throw std::runtime_error("Invalid code memory layout");
    }
    if (!(local_base >= 0 && local_size > 0 && local_base + local_size <= memory_size)) {
        throw std::runtime_error("Invalid local memory layout");
    }
    m_code_base = code_base;
    m_code_size = code_size;
    m_local_base = local_base;
    m_local_size = local_size;
}

void Riscv32CoreImpl::write_code(uint32_t addr, const std::vector<uint8_t> &code) {
    uint32_t size = uint32_t(code.size());
    if (!(addr >= m_code_base && addr + size <= m_code_base + m_code_size)) {
        throw std::runtime_error("Code does not fit into code region");
    }
    const uint8_t *src = code.data();
    uint8_t *dst = m_memory->data() + addr;
    memcpy(dst, src, size);
}

void Riscv32CoreImpl::run(uint32_t start_pc) {
    if (m_code_size == 0 || m_local_size == 0) {
        throw std::runtime_error("Memory layout is not set");
    }
    if (!(start_pc >= m_code_base && start_pc < m_code_base + m_code_size)) {
        throw std::runtime_error("Start PC is out of code region");
    }
    set_int_reg("sp", m_local_base + m_local_size - 4);
    // cannot use shift by 31 because of linker requirements
    set_int_reg("ra", (uint32_t(1) << 30));
    Hart32::pokePc(start_pc);
    Hart32::invalidateDecodeCache();
    bool ok = Hart32::run(nullptr);
    // ok: unused
}

#if 0 // TODO: Revise this
uint32_t Riscv32CoreImpl::get_arg(int index) {
    static constexpr int reg_a0 = 10;
    if (!(index >= 0 && index < 8)) {
        throw std::runtime_error("Argument index is out of range");
    }
    uint32_t value = Hart32::peekIntReg(unsigned(reg_a0 + index));
    return value;
}
#endif

uint32_t Riscv32CoreImpl::get_arg(int index) {
    static constexpr int reg_a0 = 10;
    static constexpr int reg_sp = 2;
    if (index < 0) {
        throw std::runtime_error("Argument index is out of range");
    }
    if (index < 8) {
        uint32_t value = Hart32::peekIntReg(unsigned(reg_a0 + index));
        return value;
    } else {
        uint32_t sp = Hart::peekIntReg(unsigned(reg_sp));
        uint32_t addr = sp - uint32_t(index - 8);
        uint32_t *ptr = reinterpret_cast<uint32_t *>(map_addr(addr));
        return *ptr;
    }
}

void Riscv32CoreImpl::set_ret(int index, uint32_t value) {
    static constexpr int reg_a0 = 10;
    if (!(index >= 0 && index < 2)) {
        throw std::runtime_error("Argument index is out of range");
    }
    Hart32::pokeIntReg(unsigned(reg_a0 + index), value);
}

uint8_t *Riscv32CoreImpl::map_addr(uint32_t addr) {
    return m_memory->data() + addr;
}

int Riscv32CoreImpl::execJalrHook(uint32_t pc) {
    // cannot use shift by 31 because of linker requirements
    static constexpr uint32_t mask = uint32_t(1) << 30;
    if ((pc & mask) == 0) {
        return 0;
    }
    uint32_t id = pc & ~mask;
    if (id == 0) {
        return -1;
    }
    if (m_builtin_handler == nullptr) {
        throw std::runtime_error("Missing builtin handler");
    }
    m_builtin_handler->call(this, int(id));
    return 1;
}

void Riscv32CoreImpl::set_int_reg(const std::string &reg_name, uint32_t val) {
    bool ok = true;
    unsigned reg = 0;
    if (!Hart32::findIntReg(reg_name, reg)) {
        throw std::runtime_error("No such RISCV register: " + reg_name);
    }
    Hart32::pokeIntReg(reg, val);
}

//
//    Riscv32System
//

Riscv32System *Riscv32System::create(
        int core_count,
        uint32_t mem_size, 
        uint32_t page_size, 
        uint32_t region_size) {
    return new Riscv32SystemImpl(core_count, mem_size, page_size, region_size);
}

//
//    Riscv32SystemImpl
//

Riscv32SystemImpl::Riscv32SystemImpl(
        int core_count,
        uint32_t mem_size, 
        uint32_t page_size, 
        uint32_t region_size) {
    m_memory.reset(
        new Memory(
            size_t(mem_size), 
            size_t(page_size), 
            size_t(region_size)));
    m_memory->setHartCount(core_count);
    for (int i = 0; i < core_count; i++) {
        m_cores.emplace_back(new Riscv32CoreImpl(i, m_memory.get()));
    }
}

Riscv32SystemImpl::~Riscv32SystemImpl() { }

int Riscv32SystemImpl::core_count() {
    return int(m_cores.size());
}

Riscv32Core *Riscv32SystemImpl::core_at(int index) {
    return m_cores[index].get();
}

uint8_t *Riscv32SystemImpl::map_addr(uint32_t addr) {
    return m_memory->data() + addr;
}

} // namespace core
} // namespace riscv

