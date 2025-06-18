// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <memory>

#include "arch/soc_arch.hpp"
#include "arch/noc_arch.hpp"
#include "arch/mem_map.hpp"

#include "core/addr_map.hpp"
#include "core/base_addr.hpp"
#include "core/memory.hpp"
#include "core/cb_impl.hpp"
#include "core/sync.hpp"
#include "core/dataflow_impl.hpp"
#include "core/riscv_api.hpp"

#include "ref/noc_impl.hpp"
#include "ref/compute_impl.hpp"
#include "ref/machine_impl.hpp"
#include "ref/tensix_impl.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

//
//    ThreadRunner
//

ThreadRunner::ThreadRunner(
        Sync *sync,
        Thread *thread, 
        RiscvCore *riscv_core):
            m_sync(sync),
            m_thread(thread),
            m_riscv_core(riscv_core),
            m_signal(Signal::NONE) { }

ThreadRunner::~ThreadRunner() { }

void ThreadRunner::go() {
    assert(m_signal == Signal::NONE);
    m_signal = Signal::GO;
}

void ThreadRunner::stop() {
    assert(m_signal == Signal::NONE);
    m_signal = Signal::STOP;
}

void ThreadRunner::main_loop() {
    for ( ; ; ) {
        m_sync->wait([&]() -> bool {
            return (m_signal != Signal::NONE); 
        });
        if (m_signal == Signal::STOP) {
            break;
        }
        assert(m_signal == Signal::GO);
        m_signal = Signal::NONE;
        uint32_t start_pc = get_start_pc();
        if (start_pc != 0) {
            m_thread->set_active(true);
            m_riscv_core->run(start_pc);
            m_thread->set_active(false);
        }
    }
}

uint32_t ThreadRunner::get_start_pc() {
    Tensix *tensix = m_thread->tensix();
    Memory *l1 = tensix->get_l1();
    uint32_t code_base = m_riscv_core->code_base();
    uint8_t *ptr = l1->map_addr(code_base);
    return *reinterpret_cast<uint32_t *>(ptr);
}

//
//    TensixImpl
//

TensixImpl::TensixImpl(
        MachineImpl *machine,
        uint32_t logical_x,
        uint32_t logical_y):
            m_machine(machine),
            m_my_x(0),     // deferred
            m_my_y(0),     // deferred
            m_l1(nullptr), // deferred
            m_curr_thread(nullptr) {
    Sync *sync = machine->sync();
    m_cb.reset(new CBImpl(sync));
    Soc *soc = machine->soc();
    NocArch *noc_arch = machine->noc_arch();
    SocArch *soc_arch = machine->soc_arch();
    m_my_x = soc_arch->worker_logical_to_routing_x(logical_x);
    m_my_y = soc_arch->worker_logical_to_routing_y(logical_y);
    m_noc.reset(new NocImpl(soc, noc_arch, m_my_x, m_my_y));
    RiscvCluster *riscv_cluster = machine->riscv_cluster();
    uint32_t mem_size = soc_arch->worker_l1_size();
    m_riscv_system.reset(riscv_cluster->create_system(3, mem_size));
    // this implementation keeps locals in 'init_local_l1' rather than 'local' memory
    MemMap *mem_map = machine->mem_map();
    RiscvCore *brisc_core = m_riscv_system->core_at(BRISC);
    brisc_core->set_memory_layout(
        mem_map->brisc_firmware_base(),
        mem_map->brisc_firmware_size(),
        mem_map->brisc_init_local_l1_base(),    
        mem_map->brisc_local_size());
    RiscvCore *ncrisc_core = m_riscv_system->core_at(NCRISC);
    ncrisc_core->set_memory_layout(
        mem_map->ncrisc_firmware_base(),
        mem_map->ncrisc_firmware_size(),
        mem_map->ncrisc_init_local_l1_base(),    
        mem_map->ncrisc_local_size());
    RiscvCore *trisc_core = m_riscv_system->core_at(TRISC);
    trisc_core->set_memory_layout(
        mem_map->trisc0_base(),
        mem_map->trisc0_size(),
        mem_map->trisc0_init_local_l1_base(),    
        mem_map->trisc_local_size());
    m_l1 = m_riscv_system->memory();
    Dataflow *brisc_dataflow = 
        new DataflowImpl(
            sync,
            m_l1,
            m_cb.get(),
            noc_arch,
            m_noc.get(),
            0, // noc_index
            m_my_x,
            m_my_y);
    auto brisc_main = [&]() {
        m_thread_runners[BRISC]->main_loop();
    };
    m_threads[BRISC].reset(new Thread(this, nullptr, brisc_dataflow, brisc_main));
    Compute *compute = new ComputeImpl(m_l1, m_cb.get());
    auto trisc_main = [&]() {
        m_thread_runners[TRISC]->main_loop();
    };
    m_threads[TRISC].reset(new Thread(this, compute, nullptr, trisc_main));
    Dataflow *ncrisc_dataflow = 
        new DataflowImpl(
            sync,
            m_l1,
            m_cb.get(),
            noc_arch,
            m_noc.get(),
            1, // noc_index
            m_my_x,
            m_my_y);
    auto ncrisc_main = [&]() {
        m_thread_runners[NCRISC]->main_loop();
    };
    m_threads[NCRISC].reset(new Thread(this, nullptr, ncrisc_dataflow, ncrisc_main));
    for (int i = 0; i < 3; i++) {
        m_machine->add_worker(m_threads[i]->worker());
        m_thread_runners[i].reset(
            new ThreadRunner(
                m_machine->sync(), 
                m_threads[i].get(), 
                m_riscv_system->core_at(i)));
    }
}

TensixImpl::~TensixImpl() { }

Memory *TensixImpl::get_l1() {
    return m_l1;
}

void TensixImpl::set_curr_thread(Thread *thread) {
    m_curr_thread = thread;
    m_machine->set_curr_tensix(this);
}

Compute *TensixImpl::get_compute_api() {
    assert(m_curr_thread != nullptr);
    return m_curr_thread->get_compute_api();
}

Dataflow *TensixImpl::get_dataflow_api() {
    assert(m_curr_thread != nullptr);
    return m_curr_thread->get_dataflow_api();
}

void TensixImpl::launch_kernels() {
    setup_cb();
    for (int i = 0; i < 3; i++) {
        m_thread_runners[i]->go();
    }
}

void TensixImpl::kernels_done() {
    clear_code();
}

void TensixImpl::stop() {
    for (int i = 0; i < 3; i++) {
        m_thread_runners[i]->stop();
    }
}

void TensixImpl::setup_cb() {
    uint32_t *ptr = BaseAddr::get_cb_base(m_l1);
    for (uint32_t cb_id = 0; cb_id < 32; cb_id++) { 
        uint32_t fifo_addr = ptr[0];
        uint32_t fifo_size = ptr[1];
        uint32_t fifo_num_pages = ptr[2];
        uint32_t fifo_page_size = ptr[3];
        ptr += AddrMap::UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG;
        m_cb->setup_read_write_interfaces(
            cb_id, 
            fifo_addr, 
            fifo_size, 
            fifo_num_pages, 
            fifo_page_size);
        // TODO: Implement regular data format setup
        //     (following is cheating useful just for prototyping)
        DataFormat default_format = DataFormat::Float16_b;
        m_cb->setup_data_formats(
            cb_id, 
            default_format,
            default_format,
            default_format,
            default_format);
    }
}

void TensixImpl::clear_code() {
    for (int i = 0; i < 3; i++) {
        RiscvCore *core = m_riscv_system->core_at(i);
        uint8_t *ptr = m_l1->map_addr(core->code_base());
        memset(ptr, 0, core->code_size());
    }
}

//
//    TensixBuilderImpl
//

TensixBuilderImpl::TensixBuilderImpl() { }

TensixBuilderImpl::~TensixBuilderImpl() { }

Tensix *TensixBuilderImpl::create_worker_tensix(
        MachineImpl *machine,
        uint32_t logical_x,
        uint32_t logical_y) {
    return new TensixImpl(machine, logical_x, logical_y);
}

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

