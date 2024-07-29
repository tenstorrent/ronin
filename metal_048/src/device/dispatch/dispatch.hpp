#pragma once

#include <cstdint>
#include <vector>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"

#include "dispatch/noc.hpp"

namespace tt {
namespace metal {
namespace device {
namespace dispatch {

class Dispatch {
public:
    Dispatch(Soc *soc, NocArch *noc_arch);
    ~Dispatch();
public:
    void configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read);
    void run(const uint8_t *cmd_reg, uint32_t cmd_seq_size);
private:
    void process_cmd();
    void process_write();
    void process_write_linear(bool multicast, uint32_t num_mcast_dests);
    void process_write_paged(bool is_dram);
    void process_write_packed(bool mcast, uint32_t flags);
    void process_write_host();
    void process_wait();
    void process_terminate();
    void check_cmd_reg_limit(uint32_t length);
private:
    struct ReadBufferDescriptor {
        uint32_t padded_page_size = 0;
        void *dst = nullptr;
        uint32_t dst_offset = 0;
        uint32_t num_pages_read = 0;
    };
private:
    Soc *m_soc;
    Noc m_noc;
    uint32_t m_max_write_packed_cores;
    ReadBufferDescriptor m_read_buffer_desc;
    const uint8_t *m_cmd_reg;
    const uint8_t *m_cmd_reg_end;
    const uint8_t *m_cmd_ptr;
};

} // namespace dispatch
} // namespace device
} // namespace metal
} // namespace tt

