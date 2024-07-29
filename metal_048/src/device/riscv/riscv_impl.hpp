#pragma once

#include <cstdint>
#include <vector>

#include "core/memory.hpp"
#include "core/riscv_api.hpp"
#include "core/machine.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

RiscvCluster *create_riscv_cluster(Machine *machine);

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

