
#include <cstring>
#include <cassert>
#include <cstdint>
#include <cassert>

#include "whisper/riscv/riscv32.hpp"

#include "core/machine.hpp"

#include "riscv/builtin_stdlib.hpp"
#include "riscv/stdlib_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

namespace {

void memset(Machine *machine, Riscv32Core *core) {
    uint32_t dest = core->get_arg(0);
    uint32_t ch = core->get_arg(1);
    uint32_t count = core->get_arg(2);
    Memory *l1 = machine->get_worker_l1();
    assert(dest + count <= l1->size());
    uint8_t *ptr = l1->map_addr(dest);
    std::memset(ptr, ch, count);
}

} // namespace

//
//    StdlibHandler
//

StdlibHandler::StdlibHandler(Machine *machine):
        m_machine(machine) { }

StdlibHandler::~StdlibHandler() { }

#define DECL_BUILTIN(name, count) \
    case StdlibBuiltinId::name: \
        name(m_machine, core); \
        break;

void StdlibHandler::call(Riscv32Core *core, int id) {
    switch (StdlibBuiltinId(id)) {
STDLIB_BUILTINS
    default:
        assert(false);
        break;
    }
}

#undef DECL_BUILTIN

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

