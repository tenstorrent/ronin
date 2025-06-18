// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "arch/soc_arch.hpp"
#include "arch/noc_arch.hpp"
#include "arch/mem_map.hpp"

#include "core/machine.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

class MachineBuilderImpl: public MachineBuilder {
public:
    MachineBuilderImpl();
    ~MachineBuilderImpl();
public:
    Machine *create_machine(
        SocArch *soc_arch, 
        NocArch *noc_arch,
        MemMap *mem_map) override;
};

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

