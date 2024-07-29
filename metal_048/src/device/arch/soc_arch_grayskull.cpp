
#include "arch/soc_arch.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

//
//    SocArchGrayskull
//

class SocArchGrayskull: public SocArch {
public:
    SocArchGrayskull();
    ~SocArchGrayskull();
private:
    void init();
};

SocArchGrayskull::SocArchGrayskull() {
    init();
}

SocArchGrayskull::~SocArchGrayskull() { }

void SocArchGrayskull::init() {
    SocArch::init(
        13,                  // x_size
        12,                  // y_size
        1048576,             // worker_l1_size
        524288,              // storage_core_l1_bank_size
        1073741824,          // dram_bank_size
        0,                   // eth_l1_size
        8);                  // num_dram_channels

    set_core_type(CoreType::ARC, 0, 2); 

    set_core_type(CoreType::PCIE, 0, 4); 

    set_core_type(CoreType::DRAM, 1, 0); 
    set_core_type(CoreType::DRAM, 1, 6); 
    set_core_type(CoreType::DRAM, 4, 0); 
    set_core_type(CoreType::DRAM, 4, 6); 
    set_core_type(CoreType::DRAM, 7, 0); 
    set_core_type(CoreType::DRAM, 7, 6); 
    set_core_type(CoreType::DRAM, 10, 0); 
    set_core_type(CoreType::DRAM, 10, 6); 

    set_core_type(CoreType::WORKER, 1, 1, 5); 
    set_core_type(CoreType::WORKER, 1, 7, 11); 
    set_core_type(CoreType::WORKER, 2, 1, 5); 
    set_core_type(CoreType::WORKER, 2, 7, 11); 
    set_core_type(CoreType::WORKER, 3, 1, 5); 
    set_core_type(CoreType::WORKER, 3, 7, 11); 
    set_core_type(CoreType::WORKER, 4, 1, 5); 
    set_core_type(CoreType::WORKER, 4, 7, 11); 
    set_core_type(CoreType::WORKER, 5, 1, 5); 
    set_core_type(CoreType::WORKER, 5, 7, 11); 
    set_core_type(CoreType::WORKER, 6, 1, 5); 
    set_core_type(CoreType::WORKER, 6, 7, 11); 
    set_core_type(CoreType::WORKER, 7, 1, 5); 
    set_core_type(CoreType::WORKER, 7, 7, 11); 
    set_core_type(CoreType::WORKER, 8, 1, 5); 
    set_core_type(CoreType::WORKER, 8, 7, 11); 
    set_core_type(CoreType::WORKER, 9, 1, 5); 
    set_core_type(CoreType::WORKER, 9, 7, 11); 
    set_core_type(CoreType::WORKER, 10, 1, 5); 
    set_core_type(CoreType::WORKER, 10, 7, 11); 
    set_core_type(CoreType::WORKER, 11, 1, 5); 
    set_core_type(CoreType::WORKER, 11, 7, 11); 
    set_core_type(CoreType::WORKER, 12, 1, 5); 
    set_core_type(CoreType::WORKER, 12, 7, 11); 

    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 1, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 1, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 2, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 2, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 3, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 3, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 4, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 4, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 5, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 5, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 6, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 6, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 7, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 7, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 8, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 8, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 9, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 9, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 10, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 10, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 11, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 11, 7, 10); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 12, 1, 5); 
    set_worker_core_type(WorkerCoreType::COMPUTE_AND_STORAGE, 12, 7, 10); 

    // routing, absolute
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 2, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 3, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 4, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 5, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 6, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 8, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 9, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 10, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 11, 11); 
    set_worker_core_type(WorkerCoreType::STORAGE_ONLY, 12, 11); 

    // routing, absolute
    set_worker_core_type(WorkerCoreType::DISPATCH, 1, 11); 
    set_worker_core_type(WorkerCoreType::DISPATCH, 7, 11); 

    set_core_type(CoreType::ROUTER_ONLY, 0, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 11); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 1); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 10); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 9); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 3); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 8); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 7); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 5); 
    set_core_type(CoreType::ROUTER_ONLY, 0, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 12, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 11, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 2, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 3, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 9, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 8, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 5, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 6, 0); 
    set_core_type(CoreType::ROUTER_ONLY, 12, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 11, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 2, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 3, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 9, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 8, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 5, 6); 
    set_core_type(CoreType::ROUTER_ONLY, 6, 6); 

    set_dram_preferred_worker_endpoint(0, 1, 0);
    set_dram_preferred_worker_endpoint(1, 1, 6);
    set_dram_preferred_worker_endpoint(2, 4, 0);
    set_dram_preferred_worker_endpoint(3, 4, 6);
    set_dram_preferred_worker_endpoint(4, 7, 0);
    set_dram_preferred_worker_endpoint(5, 7, 6);
    set_dram_preferred_worker_endpoint(6, 10, 0);
    set_dram_preferred_worker_endpoint(7, 10, 6);

    finalize();
}

SocArchGrayskull g_soc_arch_grayskull;

} // namespace

//
//    Public functions
//

SocArch *get_soc_arch_grayskull() {
    return &g_soc_arch_grayskull;
}

} // namespace device
} // namespace metal
} // namespace tt

