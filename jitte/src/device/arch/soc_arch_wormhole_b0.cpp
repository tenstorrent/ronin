// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "arch/soc_arch.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

//
//    SocArchWormholeB0
//

class SocArchWormholeB0: public SocArch {
public:
    SocArchWormholeB0();
    ~SocArchWormholeB0();
private:
    void init();
};


SocArchWormholeB0::SocArchWormholeB0() {
    init();
}

SocArchWormholeB0::~SocArchWormholeB0() { }

void SocArchWormholeB0::init() {
    SocArch::init(
        10,                    // x_size
        12,                    // y_size
        1499136,               // worker_l1_size
        1499136,               // storage_core_l1_bank_size
        1073741824,            // dram_bank_size
        262144,                // eth_l1_size
        12);                   // num_dram_channels

    set_core_type(CoreType::ARC, 0, 10); 

    set_core_type(CoreType::PCIE, 0, 3); 

    set_core_type(CoreType::DRAM, 0, 0);   // channel 0, 1
    set_core_type(CoreType::DRAM, 0, 1); 
    set_core_type(CoreType::DRAM, 0, 11); 
    set_core_type(CoreType::DRAM, 0, 5);   // channel 2, 3 
    set_core_type(CoreType::DRAM, 0, 6); 
    set_core_type(CoreType::DRAM, 0, 7); 
    set_core_type(CoreType::DRAM, 5, 0);   // channel 4, 5 
    set_core_type(CoreType::DRAM, 5, 1); 
    set_core_type(CoreType::DRAM, 5, 11); 
    set_core_type(CoreType::DRAM, 5, 2);   // channel 6, 7
    set_core_type(CoreType::DRAM, 5, 9); 
    set_core_type(CoreType::DRAM, 5, 10); 
    set_core_type(CoreType::DRAM, 5, 3);   // channel 8, 9 
    set_core_type(CoreType::DRAM, 5, 4); 
    set_core_type(CoreType::DRAM, 5, 8); 
    set_core_type(CoreType::DRAM, 5, 5);   // channel 10, 11 
    set_core_type(CoreType::DRAM, 5, 6); 
    set_core_type(CoreType::DRAM, 5, 7); 

    set_core_type(CoreType::ETH, 1, 0); 
    set_core_type(CoreType::ETH, 2, 0); 
    set_core_type(CoreType::ETH, 3, 0); 
    set_core_type(CoreType::ETH, 4, 0); 
    set_core_type(CoreType::ETH, 6, 0); 
    set_core_type(CoreType::ETH, 7, 0); 
    set_core_type(CoreType::ETH, 8, 0); 
    set_core_type(CoreType::ETH, 9, 0); 
    set_core_type(CoreType::ETH, 1, 6); 
    set_core_type(CoreType::ETH, 2, 6); 
    set_core_type(CoreType::ETH, 3, 6); 
    set_core_type(CoreType::ETH, 4, 6); 
    set_core_type(CoreType::ETH, 6, 6); 
    set_core_type(CoreType::ETH, 7, 6); 
    set_core_type(CoreType::ETH, 8, 6); 
    set_core_type(CoreType::ETH, 9, 6); 

    set_core_type(CoreType::WORKER, 1, 1, 5); 
    set_core_type(CoreType::WORKER, 1, 7, 11); 
    set_core_type(CoreType::WORKER, 2, 1, 5); 
    set_core_type(CoreType::WORKER, 2, 7, 11); 
    set_core_type(CoreType::WORKER, 3, 1, 5); 
    set_core_type(CoreType::WORKER, 3, 7, 11); 
    set_core_type(CoreType::WORKER, 4, 1, 5); 
    set_core_type(CoreType::WORKER, 4, 7, 11); 
    set_core_type(CoreType::WORKER, 6, 1, 5); 
    set_core_type(CoreType::WORKER, 6, 7, 11); 
    set_core_type(CoreType::WORKER, 7, 1, 5); 
    set_core_type(CoreType::WORKER, 7, 7, 11); 
    set_core_type(CoreType::WORKER, 8, 1, 5); 
    set_core_type(CoreType::WORKER, 8, 7, 11); 
    set_core_type(CoreType::WORKER, 9, 1, 5); 
    set_core_type(CoreType::WORKER, 9, 7, 11); 

    // routing, absolute (TODO: Revise this)
    set_worker_core_type(WorkerCoreType::DISPATCH, 1, 11); 
#if 0
    set_worker_core_type(WorkerCoreType::DISPATCH, 2, 11); 
    set_worker_core_type(WorkerCoreType::DISPATCH, 3, 11); 
    set_worker_core_type(WorkerCoreType::DISPATCH, 4, 11); 
    set_worker_core_type(WorkerCoreType::DISPATCH, 5, 11); 
    set_worker_core_type(WorkerCoreType::DISPATCH, 6, 11); 
    set_worker_core_type(WorkerCoreType::DISPATCH, 7, 11); 
    set_worker_core_type(WorkerCoreType::DISPATCH, 8, 11); 
#endif

    set_core_type(CoreType::ROUTER_ONLY, 0, 2); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 4); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 8); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 9); 

    set_dram_preferred_worker_endpoint(0, 0, 11);
    set_dram_preferred_worker_endpoint(1, 0, 1);
    set_dram_preferred_worker_endpoint(2, 0, 5);
    set_dram_preferred_worker_endpoint(3, 0, 7);
    set_dram_preferred_worker_endpoint(4, 5, 1);
    set_dram_preferred_worker_endpoint(5, 5, 11);
    set_dram_preferred_worker_endpoint(6, 5, 2);
    set_dram_preferred_worker_endpoint(7, 5, 9);
    set_dram_preferred_worker_endpoint(8, 5, 8);
    set_dram_preferred_worker_endpoint(9, 5, 3);
    set_dram_preferred_worker_endpoint(10, 5, 5);
    set_dram_preferred_worker_endpoint(11, 5, 7);

    finalize();
}

SocArchWormholeB0 g_soc_arch_wormhole_b0;

} // namespace

//
//    Public functions
//

SocArch *get_soc_arch_wormhole_b0() {
    return &g_soc_arch_wormhole_b0;
}

} // namespace device
} // namespace metal
} // namespace tt

