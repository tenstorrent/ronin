
#include <cstdint>
#include <cassert>
#include <functional>

#include "schedule/schedule.hpp"

#include "arch/soc_arch.hpp"
#include "arch/noc_arch.hpp"
#include "arch/mem_map.hpp"

#include "core/riscv_api.hpp"
#include "core/soc.hpp"
#include "core/sync.hpp"
#include "core/compute_api.hpp"
#include "core/dataflow_api.hpp"
#include "core/machine.hpp"

#include "riscv/riscv_impl.hpp"
#include "riscv/builtin_handler.hpp"

#include "ref/machine_impl.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

using schedule::Scheduler;
using schedule::Worker;

using riscv::create_riscv_cluster;
using riscv::BuiltinHandler;

//
//    ThreadWorker
//

ThreadWorker::ThreadWorker():
        m_thread(nullptr) { }

ThreadWorker::~ThreadWorker() { }

void ThreadWorker::init(Thread *thread) {
    m_thread = thread;
}

void ThreadWorker::run() {
    assert(m_thread != nullptr);
    m_thread->run();
}

void ThreadWorker::on_resume() {
    m_thread->set_curr();
}

bool ThreadWorker::is_active() {
    return m_thread->is_active();
}

//
//    Thread
//

Thread::Thread(
        Tensix *tensix, 
        Compute *compute,
        Dataflow *dataflow,
        std::function<void ()> main):
            m_tensix(tensix), 
            m_compute(compute), 
            m_dataflow(dataflow),
            m_main(main),
            m_active(false) { 
    m_worker.init(this);
    for (int i = 0; i < ARGS_SIZE; i++) {
        m_args[i] = 0;
    }
}

Thread::~Thread() { }

void Thread::set_arg(int index, uint32_t value) {
    assert(index >= 0 && index < ARGS_SIZE);
    m_args[index] = value;
}

const void *Thread::get_arg_ptr(int index) {
    assert(index >= 0 && index < ARGS_SIZE);
    return &m_args[index];
}

void Thread::run() {
    if (m_compute != nullptr) {
        m_compute->reset();
    }
    if (m_dataflow != nullptr) {
        m_dataflow->reset();
    }
    m_main();
}

void Thread::set_curr() {
    m_tensix->set_curr_thread(this);
}

//
//    MachineImpl
//

MachineImpl::MachineImpl(
        SocArch *soc_arch,
        NocArch *noc_arch,
        MemMap *mem_map,
        TensixBuilder *tensix_builder):
            m_soc_arch(soc_arch),
            m_noc_arch(noc_arch),
            m_mem_map(mem_map),
            m_soc(soc_arch),
            m_scheduler(),
            m_sync(&m_scheduler),
            m_builtin_handler(this),
            m_worker_l1_size(soc_arch->worker_l1_size()),
            m_size_x(soc_arch->worker_x_size()),
            m_size_y(soc_arch->worker_y_size()),
            m_curr_tensix(nullptr) {
    m_riscv_cluster.reset(create_riscv_cluster(this));
    m_tensix.resize(m_size_x * m_size_y);
    for (uint32_t x = 0; x < m_size_x; x++) {
        for (uint32_t y = 0; y < m_size_y; y++) {
            uint32_t index = linear_tensix_index(x, y); 
            Tensix *tensix = tensix_builder->create_worker_tensix(this, x, y);
            m_tensix[index].reset(tensix);
            // deferred L1 setting
            m_soc.set_worker_l1(x, y, tensix->get_l1());
        }
    }
}

MachineImpl::~MachineImpl() { }

Soc *MachineImpl::soc() {
    return &m_soc;
}

Compute *MachineImpl::get_compute_api() {
    assert(m_curr_tensix != nullptr);
    return m_curr_tensix->get_compute_api();
}

Dataflow *MachineImpl::get_dataflow_api() {
    assert(m_curr_tensix != nullptr);
    return m_curr_tensix->get_dataflow_api();
}

Memory *MachineImpl::get_worker_l1() {
    assert(m_curr_tensix != nullptr);
    return m_curr_tensix->get_l1();
}

void MachineImpl::launch_kernels() {
    for (auto &tensix: m_tensix) {
        tensix->launch_kernels();
    }
    m_scheduler.run();
    for (auto &tensix: m_tensix) {
        tensix->kernels_done();
    }
}

void MachineImpl::stop() {
    for (auto &tensix: m_tensix) {
        tensix->stop();
    }
    m_scheduler.run();
}

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

