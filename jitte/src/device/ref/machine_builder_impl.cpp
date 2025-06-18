// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "arch/soc_arch.hpp"
#include "arch/noc_arch.hpp"
#include "arch/mem_map.hpp"

#include "core/machine.hpp"

#include "ref/machine_impl.hpp"
#include "ref/tensix_impl.hpp"
#include "ref/machine_builder_impl.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

//
//    MachineBuilderImpl
//

MachineBuilderImpl::MachineBuilderImpl() { }

MachineBuilderImpl::~MachineBuilderImpl() { }

Machine *MachineBuilderImpl::create_machine(
        SocArch *soc_arch, 
        NocArch *noc_arch,
        MemMap *mem_map) {
    TensixBuilderImpl tensix_builder;
    return new MachineImpl(soc_arch, noc_arch, mem_map, &tensix_builder);
}

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

