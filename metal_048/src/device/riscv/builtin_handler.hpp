#pragma once

#include "whisper/riscv/riscv32.hpp"

#include "core/machine.hpp"

#include "riscv/compute_handler.hpp"
#include "riscv/compute_tanto_handler.hpp"
#include "riscv/dataflow_handler.hpp"
#include "riscv/dataflow_tanto_handler.hpp"
#include "riscv/stdlib_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32BuiltinHandler;
using ::riscv::core::Riscv32Core;

class BuiltinHandler: public Riscv32BuiltinHandler {
public:
    BuiltinHandler(Machine *machine);
    ~BuiltinHandler();
public:
    void call(Riscv32Core *core, int id) override;
private:
    ComputeHandler m_compute_handler;
    ComputeTantoHandler m_compute_tanto_handler;
    DataflowHandler m_dataflow_handler;
    DataflowTantoHandler m_dataflow_tanto_handler;
    StdlibHandler m_stdlib_handler;
};

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

