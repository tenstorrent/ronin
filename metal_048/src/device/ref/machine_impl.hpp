#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>

#include "schedule/schedule.hpp"

#include "arch/soc_arch.hpp"
#include "arch/noc_arch.hpp"
#include "arch/mem_map.hpp"

#include "core/memory.hpp"
#include "core/soc.hpp"
#include "core/sync.hpp"
#include "core/compute_api.hpp"
#include "core/dataflow_api.hpp"
#include "core/machine.hpp"
#include "core/riscv_api.hpp"

#include "riscv/builtin_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

using schedule::Scheduler;
using schedule::Worker;

using riscv::BuiltinHandler;

class Thread;
class Tensix;
class MachineImpl;

//
//    ThreadWorker
//

class ThreadWorker: public Worker {
public:
    ThreadWorker();
    ~ThreadWorker();
public:
    void init(Thread *thread);
public:
    void run() override;
    void on_resume() override;
    bool is_active() override;
private:
    Thread *m_thread;
};

//
//    Thread
//

class Thread {
public:
    Thread(
        Tensix *tensix, 
        Compute *compute,
        Dataflow *dataflow,
        std::function<void ()> main);
    ~Thread();
public:
    Worker *worker() {
        return &m_worker;
    }
    Tensix *tensix() {
        return m_tensix;
    }
    Compute *get_compute_api() {
        return m_compute.get();
    }
    Dataflow *get_dataflow_api() {
        return m_dataflow.get();
    }
    bool is_active() {
        return m_active;
    }
    void set_active(bool active) {
        m_active = active;
    }
    void set_arg(int index, uint32_t value);
    const void *get_arg_ptr(int index);
    void run();
    void set_curr();
private:
    static constexpr int ARGS_SIZE = 256;
private:
    ThreadWorker m_worker;
    Tensix *m_tensix;
    std::unique_ptr<Compute> m_compute;
    std::unique_ptr<Dataflow> m_dataflow;
    std::function<void ()> m_main;
    bool m_active;
    uint32_t m_args[ARGS_SIZE];
};

//
//    Tensix
//

class Tensix {
public:
    Tensix() { }
    virtual ~Tensix() { }
public:
    virtual Memory *get_l1() = 0;
    virtual void set_curr_thread(Thread *thread) = 0;
    virtual Compute *get_compute_api() = 0;
    virtual Dataflow *get_dataflow_api() = 0;
    virtual void launch_kernels() = 0;
    virtual void kernels_done() = 0;
    virtual void stop() = 0;
};

//
//    TensixBuilder
//

class TensixBuilder {
public:
    TensixBuilder() { }
    virtual ~TensixBuilder() { }
public:
    virtual Tensix *create_worker_tensix(
        MachineImpl *machine,
        uint32_t logical_x,
        uint32_t logical_y) = 0;
};

//
//    MachineImpl
//

class MachineImpl: public Machine {
public:
    MachineImpl(
        SocArch *soc_arch,
        NocArch *noc_arch,
        MemMap *mem_map,
        TensixBuilder *tensix_builder);
    ~MachineImpl();
public:
    NocArch *noc_arch() {
        return m_noc_arch;
    }
    SocArch *soc_arch() {
        return m_soc_arch;
    }
    MemMap *mem_map() {
        return m_mem_map;
    }
    RiscvCluster *riscv_cluster() {
        return m_riscv_cluster.get();
    }
    Sync *sync() {
        return &m_sync;
    }
    void add_worker(Worker *worker) {
        m_scheduler.add_worker(worker);
    }
    void set_curr_tensix(Tensix *tensix) {
        m_curr_tensix = tensix;
    }
public:
    Soc *soc() override;
    Compute *get_compute_api() override;
    Dataflow *get_dataflow_api() override;
    Memory *get_worker_l1() override;
    void launch_kernels() override;
    void stop() override;
private:
    uint32_t linear_tensix_index(uint32_t x, uint32_t y) {
        return x * m_size_y + y;
    }
private:
    SocArch *m_soc_arch;
    NocArch *m_noc_arch;
    MemMap *m_mem_map;
    std::unique_ptr<RiscvCluster> m_riscv_cluster;
    Soc m_soc;
    Scheduler m_scheduler;
    Sync m_sync;
    BuiltinHandler m_builtin_handler;
    uint32_t m_worker_l1_size;
    uint32_t m_size_x;
    uint32_t m_size_y;
    std::vector<std::unique_ptr<Tensix>> m_tensix;
    Tensix *m_curr_tensix;
};

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

