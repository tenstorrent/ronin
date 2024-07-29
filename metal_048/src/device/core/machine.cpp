
#include "core/machine.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

MachineBuilder *g_machine_builder = nullptr;

} // namespace

//
//    Global functions
//

void set_machine_builder(MachineBuilder *machine_builder) {
    g_machine_builder = machine_builder;
}

MachineBuilder *get_machine_builder() {
    return g_machine_builder;
}

} // namespace device
} // namespace metal
} // namespace tt

