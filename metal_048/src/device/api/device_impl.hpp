#pragma once

#include <cstdint>
#include <memory>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"
#include "core/machine.hpp"

#include "dispatch/dispatch.hpp"
#include "dispatch/prefetch.hpp"

#include "api/device_api.hpp"

namespace tt {
namespace metal {
namespace device {

class DeviceImpl: public Device {
public:
    DeviceImpl(Arch arch);
    ~DeviceImpl();
public:
    void start() override;
    void stop() override;
    void deassert_risc_reset() override;
    void assert_risc_reset() override;
    void write(
        const void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) override;
    void read(
        void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) override;
    void write_to_sysmem(
        const void *data,
        uint32_t size,
        uint64_t addr) override;
    void read_from_sysmem(
        void *data,
        uint32_t size,
        uint64_t addr) override;
    void *host_dma_address(uint64_t offset) override;
    // command processor interface
    void configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) override;
    void run_commands(const uint8_t *cmd_reg, uint32_t cmd_seq_size) override;
    void launch_kernels() override;
private:
    void write_dram(
        const void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr);
    void write_worker(
        const void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr);
    void read_dram(
        void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr);
    void read_worker(
        void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr);
private:
    Soc *m_soc;
    std::unique_ptr<MachineBuilder> m_default_machine_builder;
    std::unique_ptr<Machine> m_machine;
    std::unique_ptr<dispatch::Dispatch> m_dispatch;
    std::unique_ptr<dispatch::Prefetch> m_prefetch;
};

} // namespace device
} // namespace metal
} // namespace tt

