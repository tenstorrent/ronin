#pragma once

#include "arch/soc_arch.hpp"
#include "arch/noc_arch.hpp"
#include "arch/mem_map.hpp"

#include "core/memory.hpp"
#include "core/soc.hpp"
#include "core/compute_api.hpp"
#include "core/dataflow_api.hpp"

namespace tt {
namespace metal {
namespace device {

//
//    Machine
//

class Machine {
public:
    Machine() { }
    virtual ~Machine() { }
public:
    virtual Soc *soc() = 0;
    virtual Compute *get_compute_api() = 0;
    virtual Dataflow *get_dataflow_api() = 0;
    virtual Memory *get_worker_l1() = 0;
    virtual void launch_kernels() = 0;
    virtual void stop() = 0;
};

//
//    MachineBuilder
//

class MachineBuilder {
public:
    MachineBuilder() { }
    virtual ~MachineBuilder() { }
public:
    virtual Machine *create_machine(
        SocArch *soc_arch, 
        NocArch *noc_arch,
        MemMap *mem_map) = 0;
};

//
//    Global functions
//

void set_machine_builder(MachineBuilder *machine_builder);
MachineBuilder *get_machine_builder();

} // namespace device
} // namespace metal
} // namespace tt

