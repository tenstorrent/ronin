#pragma once

#include <cstdint>
#include <array>
#include <memory>

#include "core/memory.hpp"
#include "core/cb_api.hpp"
#include "core/noc_api.hpp"
#include "core/sync.hpp"
#include "core/dataflow_impl.hpp"
#include "core/riscv_api.hpp"

#include "ref/machine_impl.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

//
//    ThreadRunner
//

class ThreadRunner {
public:
    ThreadRunner(
        Sync *sync,
        Thread *thread, 
        RiscvCore *riscv_core);
    ~ThreadRunner();
public:
    void go();
    void stop();
    void main_loop();
private:
    uint32_t get_start_pc();
private:
    enum class Signal {
        NONE,
        GO,
        STOP
    };
private:
    Sync *m_sync;
    Thread *m_thread;
    RiscvCore *m_riscv_core;
    Signal m_signal;
};

//
//    TensixImpl
//

class TensixImpl: public Tensix {
public:
    TensixImpl(
        MachineImpl *machine,
        uint32_t logical_x,
        uint32_t logical_y);
    ~TensixImpl();
public:
    Memory *get_l1() override;
    void set_curr_thread(Thread *thread) override;
    Compute *get_compute_api() override;
    Dataflow *get_dataflow_api() override;
    void launch_kernels() override;
    void kernels_done() override;
    void stop() override;
private:
    void setup_cb();
    void clear_code();
private:
    enum {
        BRISC = 0,
        TRISC = 1,
        NCRISC = 2
    };
private:
    MachineImpl *m_machine;
    Memory *m_l1;
    std::unique_ptr<CB> m_cb;
    std::unique_ptr<Noc> m_noc;
    std::unique_ptr<RiscvSystem> m_riscv_system;
    std::array<std::unique_ptr<Thread>, 3> m_threads;
    std::array<std::unique_ptr<ThreadRunner>, 3> m_thread_runners;
    Thread *m_curr_thread;
};

//
//    TensixBuilderImpl
//

class TensixBuilderImpl: public TensixBuilder {
public:
    TensixBuilderImpl();
    ~TensixBuilderImpl();
public:
    Tensix *create_worker_tensix(
        MachineImpl *machine,
        uint32_t logical_x,
        uint32_t logical_y) override;
};

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

