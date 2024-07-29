#pragma once

#include <cstdint>

namespace tt {
namespace metal {
namespace device {

class Device {
public:
    Device() { }
    virtual ~Device() { }
public:
    enum class Arch {
        GRAYSKULL,
        WORMHOLE_B0
};
public:
    static Device *create(Arch arch);
public:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void deassert_risc_reset() = 0;
    virtual void assert_risc_reset() = 0;
    virtual void write(
        const void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) = 0;
    virtual void read(
        void *data,
        uint32_t size,
        uint32_t x,
        uint32_t y,
        uint64_t addr) = 0;
    virtual void write_to_sysmem(
        const void *data,
        uint32_t size,
        uint64_t addr) = 0;
    virtual void read_from_sysmem(
        void *data,
        uint32_t size,
        uint64_t addr) = 0;
    virtual void *host_dma_address(uint64_t offset) = 0;
    // command processor interface
    virtual void configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) = 0;
    virtual void run_commands(const uint8_t *cmd_reg, uint32_t cmd_seq_size) = 0;
    virtual void launch_kernels() = 0;
};

} // namespace device
} // namespace metal
} // namespace tt

