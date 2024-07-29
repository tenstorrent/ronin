
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <stdexcept>

#include "arch/soc_arch.hpp"
#include "arch/noc_arch.hpp"
#include "arch/mem_map.hpp"

#include "core/addr_map.hpp"
#include "core/soc.hpp"
#include "core/machine.hpp"

#include "ref/machine_builder_impl.hpp"

#include "api/device_api.hpp"
#include "api/device_impl.hpp"

namespace tt {
namespace metal {
namespace device {

//
//    DeviceImpl
//

DeviceImpl::DeviceImpl(Arch arch):
        m_soc(nullptr) {
    SocArch *soc_arch = nullptr;
    NocArch *noc_arch = nullptr;
    MemMap *mem_map = nullptr;
    switch (arch) {
    case Arch::GRAYSKULL:
        soc_arch = get_soc_arch_grayskull();
        noc_arch = get_noc_arch_grayskull();
        mem_map = get_mem_map_grayskull();
        break;
    case Arch::WORMHOLE_B0:
        soc_arch = get_soc_arch_wormhole_b0();
        noc_arch = get_noc_arch_wormhole_b0();
        mem_map = get_mem_map_wormhole_b0();
        break;
    default:
        assert(false);
        break;
    }
    m_default_machine_builder.reset(new ref::MachineBuilderImpl());
    MachineBuilder *machine_builder = get_machine_builder();
    if (machine_builder == nullptr) {
        machine_builder = m_default_machine_builder.get();
    }
    m_machine.reset(machine_builder->create_machine(soc_arch, noc_arch, mem_map));
    m_soc = m_machine->soc();
    m_dispatch.reset(new dispatch::Dispatch(m_soc, noc_arch)),
    m_prefetch.reset(new dispatch::Prefetch(m_soc, noc_arch, m_dispatch.get()));
}

DeviceImpl::~DeviceImpl() { }

void DeviceImpl::start() {
    // TODO
}

void DeviceImpl::stop() {
    m_machine->stop();
}

void DeviceImpl::deassert_risc_reset() {
    // TODO
}

void DeviceImpl::assert_risc_reset() {
    // TODO
}

void DeviceImpl::write(
        const void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) {
    CoreType core_type = m_soc->core_type(x, y);
    if (core_type == CoreType::DRAM) {
        write_dram(data, size, x, y, addr);
    } else if (core_type == CoreType::WORKER) {
        write_worker(data, size, x, y, addr);
    } else {
        throw std::runtime_error(
            "Unsupported device write for core type " + std::to_string(int(core_type)));
    }
}

void DeviceImpl::read(
        void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) {
    CoreType core_type = m_soc->core_type(x, y);
    if (core_type == CoreType::DRAM) {
        read_dram(data, size, x, y, addr);
    } else if (core_type == CoreType::WORKER) {
        read_worker(data, size, x, y, addr);
    } else {
        throw std::runtime_error(
            "Unsupported device read for core type " + std::to_string(int(core_type)));
    }
}

void DeviceImpl::write_to_sysmem(
        const void *data,
        uint32_t size,
        uint64_t addr) {
    if (addr + uint64_t(size) > uint64_t(m_soc->sysmem_size())) {
        throw std::runtime_error("Invalid sysmem address range");
    }
    uint32_t addr32 = uint32_t(addr);
    assert(uint64_t(addr32) == addr);
    uint8_t *ptr = m_soc->map_sysmem_addr(addr32);
    memcpy(ptr, data, size);
}

void DeviceImpl::read_from_sysmem(
        void *data,
        uint32_t size,
        uint64_t addr) {
    if (addr + uint64_t(size) > uint64_t(m_soc->sysmem_size())) {
        throw std::runtime_error("Invalid sysmem address range");
    }
    uint32_t addr32 = uint32_t(addr);
    assert(uint64_t(addr32) == addr);
    uint8_t *ptr = m_soc->map_sysmem_addr(addr32);
    // temporary workaround: stub for unexistent HOST_CQ_* fields
    // (just sifficient to break from host busy wait 'while' loops
    if (addr32 == AddrMap::HOST_CQ_READ_PTR) {
        // see SystemMemoryWriter::cq_reserve_back in "impl/dispatch/command_queue_interface.cpp"
        *reinterpret_cast<uint32_t *>(ptr) = 0;
    } else if (addr32 == AddrMap::HOST_CQ_FINISH_PTR) {
        // see CommandQueue::finish in "impl/dispatch/command_queue.cpp"
        *reinterpret_cast<uint32_t *>(ptr) = 1;
    }
    memcpy(data, ptr, size);
}

void *DeviceImpl::host_dma_address(uint64_t offset) {
    if (offset >= uint64_t(m_soc->sysmem_size())) {
        throw std::runtime_error("Invalid sysmem offset");
    }
    uint32_t addr32 = uint32_t(offset);
    assert(uint64_t(addr32) == offset);
    return m_soc->map_sysmem_addr(addr32);
}

void DeviceImpl::configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) {
    m_dispatch->configure_read_buffer(padded_page_size, dst, dst_offset, num_pages_read);
}

void DeviceImpl::run_commands(const uint8_t *cmd_reg, uint32_t cmd_seq_size) {
    m_prefetch->run(cmd_reg, cmd_seq_size);
}

void DeviceImpl::launch_kernels() {
    m_machine->launch_kernels();
}

void DeviceImpl::write_dram(
        const void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) {
    uint32_t channel = m_soc->core_dram_channel(x, y);
    if (addr + uint64_t(size) > uint64_t(m_soc->dram_size(channel))) {
        throw std::runtime_error("Invalid DRAM address range");
    }
    uint32_t addr32 = uint32_t(addr);
    assert(uint64_t(addr32) == addr);
    uint8_t *ptr = m_soc->map_dram_addr(channel, addr32);
    memcpy(ptr, data, size);
}

void DeviceImpl::write_worker(
        const void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) {
    if (addr + uint64_t(size) > uint64_t(m_soc->l1_size(x, y))) {
        throw std::runtime_error("Invalid L1 address range");
    }
    uint32_t addr32 = uint32_t(addr);
    assert(uint64_t(addr32) == addr);
    uint8_t *ptr = m_soc->map_l1_addr(x, y, addr32);
    memcpy(ptr, data, size);
    WorkerCoreType worker_core_type = m_soc->worker_core_type(x, y);
#if 0 // TODO: Implement new dispatch
    // special support for dispatch cores
    if (m_soc->worker_core_type(x, y) == WorkerCoreType::DISPATCH) {
        if (addr32 == AddrMap::CQ_WRITE_PTR) {
            uint32_t write_ptr_and_toggle = *reinterpret_cast<uint32_t *>(ptr);
            bool launch_program = false;
            m_dispatch->run(write_ptr_and_toggle, launch_program);
            if (launch_program) {
                m_machine->launch_kernels();
            }
        }
    }
#endif
}

void DeviceImpl::read_dram(
        void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) {
    uint32_t channel = m_soc->core_dram_channel(x, y);
    if (addr + uint64_t(size) > uint64_t(m_soc->dram_size(channel))) {
        throw std::runtime_error("Invalid DRAM address range");
    }
    uint32_t addr32 = uint32_t(addr);
    assert(uint64_t(addr32) == addr);
    uint8_t *ptr = m_soc->map_dram_addr(channel, addr32);
    memcpy(data, ptr, size);
}

void DeviceImpl::read_worker(
        void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) {
    if (addr + uint64_t(size) > uint64_t(m_soc->l1_size(x, y))) {
        throw std::runtime_error("Invalid L1 address range");
    }
    uint32_t addr32 = uint32_t(addr);
    assert(uint64_t(addr32) == addr);
    uint8_t *ptr = m_soc->map_l1_addr(x, y, addr32);
    memcpy(data, ptr, size);
    // workaround for mailbox RUN_MSG field
    if (addr32 == 28 && size == 4) {
        memset(ptr, 0, 4);
    }
}

} // namespace device
} // namespace metal
} // namespace tt

