#pragma once

#include <cstdint>

#include "whisper/riscv/riscv32.hpp"

#include "core/kernel_structs.hpp"
#include "core/llk_defs.hpp"
#include "core/machine.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

class ComputeHandler {
public:
    ComputeHandler(Machine *machine);
    ~ComputeHandler();
public:
    void call(Riscv32Core *core, int id);
private:
    Machine *m_machine;
};

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

