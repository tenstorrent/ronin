// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <vector>
#include <memory>

#include "whisper/riscv/riscv32.hpp"

#include "core/memory.hpp"
#include "core/riscv_api.hpp"
#include "core/machine.hpp"

#include "riscv/builtin_handler.hpp"
#include "riscv/riscv_impl.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

namespace {

namespace rv32 = ::riscv::core;

//
//    MemoryImpl    
//

class MemoryImpl: public Memory {
public:
    MemoryImpl(rv32::Riscv32System *system, uint32_t size):
        m_system(system), m_size(size) { }
    ~MemoryImpl() { }
public:
    uint32_t size() override {
        return m_size;
    }
    uint8_t *map_addr(uint32_t addr) override {
        return m_system->map_addr(addr);
    }
private:
    rv32::Riscv32System *m_system;
    uint32_t m_size;
};

//
//    RiscvCoreImpl
//

class RiscvCoreImpl: public RiscvCore {
public:
    RiscvCoreImpl(rv32::Riscv32Core *core):
        m_core(core) { }
    ~RiscvCoreImpl() { }
public:
    void set_builtin_handler(BuiltinHandler *builtin_handler) {
        m_core->set_builtin_handler(builtin_handler);
    }
    void set_memory_layout(
            uint32_t code_base,
            uint32_t code_size,
            uint32_t local_base,
            uint32_t local_size) override {
        m_core->set_memory_layout(code_base, code_size, local_base, local_size);
    }
    uint32_t code_base() override {
        return m_core->code_base();
    }
    uint32_t code_size() override {
        return m_core->code_size();
    }
    void write_code(const std::vector<uint8_t> &code) override {
        // TODO: Check whether 'addr' argument of 'Riscv32Core::write_code' is really needed
        m_core->write_code(m_core->code_base(), code);
    }
    void run(uint32_t start_pc) override {
        m_core->run(start_pc);
    }
private:
    rv32::Riscv32Core *m_core;
};

//
//    RiscvSystemImpl
//

class RiscvSystemImpl: public RiscvSystem {
public:
    RiscvSystemImpl(
        BuiltinHandler *builtin_handler, 
        rv32::Riscv32System *system,
        uint32_t mem_size);
    ~RiscvSystemImpl();
public:
    Memory *memory() override {
        return m_memory.get();
    }
    int core_count() override {
        return m_system->core_count();
    }
    RiscvCore *core_at(int index) override {
        return m_cores[index].get();
    }
private:
    std::unique_ptr<rv32::Riscv32System> m_system;
    std::unique_ptr<MemoryImpl> m_memory;
    std::vector<std::unique_ptr<RiscvCoreImpl>> m_cores;
};

RiscvSystemImpl::RiscvSystemImpl(
        BuiltinHandler *builtin_handler, 
        rv32::Riscv32System *system,
        uint32_t mem_size):
            m_system(system) {
    m_memory.reset(new MemoryImpl(m_system.get(), mem_size));
    int core_count = m_system->core_count();
    m_cores.resize(core_count);
    for (int i = 0; i < core_count; i++) {
        m_cores[i].reset(new RiscvCoreImpl(m_system->core_at(i)));
        m_cores[i]->set_builtin_handler(builtin_handler);
    }
}

RiscvSystemImpl::~RiscvSystemImpl() { }

//
//    RiscvClusterImpl
//

class RiscvClusterImpl: public RiscvCluster {
public:
    RiscvClusterImpl(Machine *machine);
    ~RiscvClusterImpl();
public:
    RiscvSystem *create_system(int core_count, uint32_t mem_size) override;
private:
    BuiltinHandler m_builtin_handler;
};

RiscvClusterImpl::RiscvClusterImpl(Machine *machine):
        m_builtin_handler(machine) { }

RiscvClusterImpl::~RiscvClusterImpl() { }

RiscvSystem *RiscvClusterImpl::create_system(int core_count, uint32_t mem_size) {
#if 1
    // ACHTUNG: Whisper requires memory size be multiple of 1024 * 1024 bytes (for PMP)
    // As temporary workaround, mem_size is aligned here
    uint32_t sec_size = 1024 * 1024;
    mem_size = ((mem_size + sec_size - 1) / sec_size) * sec_size;
#endif
    rv32::Riscv32System *system = 
        rv32::Riscv32System::create(core_count, mem_size, 4 * 1024, 0);
    return new RiscvSystemImpl(&m_builtin_handler, system, mem_size);
}

} // namespace

//
//    Public functions
//

RiscvCluster *create_riscv_cluster(Machine *machine) {
    return new RiscvClusterImpl(machine);
}

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

