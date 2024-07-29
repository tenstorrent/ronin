#pragma once

#include <cstdint>
#include <vector>

#include "core/memory.hpp"

namespace tt {
namespace metal {
namespace device {

class RiscvCore {
public:
    RiscvCore() { }
    virtual ~RiscvCore() { }
public:
    virtual void set_memory_layout(
        uint32_t code_base,
        uint32_t code_size,
        uint32_t local_base,
        uint32_t local_size) = 0;
    virtual uint32_t code_base() = 0;
    virtual uint32_t code_size() = 0;
    virtual void write_code(const std::vector<uint8_t> &code) = 0; 
    virtual void run(uint32_t start_pc) = 0;
};

class RiscvSystem {
public:
    RiscvSystem() { }
    virtual ~RiscvSystem() { }
public:
    virtual Memory *memory() = 0;
    virtual int core_count() = 0;
    virtual RiscvCore *core_at(int index) = 0;
};

class RiscvCluster {
public:
    RiscvCluster() { }
    virtual ~RiscvCluster() { }
public:
    virtual RiscvSystem *create_system(int core_count, uint32_t mem_size) = 0;
};

} // namespace device
} // namespace metal
} // namespace tt

