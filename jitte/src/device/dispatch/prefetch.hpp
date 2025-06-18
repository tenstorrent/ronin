// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "arch/noc_arch.hpp"

#include "core/soc.hpp"

#include "dispatch/noc.hpp"
#include "dispatch/dispatch.hpp"

namespace tt {
namespace metal {
namespace device {
namespace dispatch {

class Prefetch {
public:
    Prefetch(
        Soc *soc, 
        NocArch *noc_arch, 
        Dispatch *dispatch);
    ~Prefetch();
public:
    void run(const uint8_t *cmd_reg, uint32_t cmd_seq_size);
private:
    void process_cmd(uint32_t &stride);
    uint32_t process_relay_linear_cmd();
    uint32_t process_relay_paged_cmd(bool is_dram, uint32_t page_id);
    uint32_t process_relay_paged_packed_cmd();
    void process_relay_paged_packed_sub_cmds(uint32_t total_length);
    uint32_t process_relay_inline_cmd();
    uint32_t process_relay_inline_noflush_cmd();
    uint32_t process_stall();
    uint32_t process_terminate();
    void flush_dispatch_data();
    void check_cmd_reg_limit(uint32_t length);
    void copy_to_local_cache(const uint8_t *src, uint32_t count);
private:
    Soc *m_soc;
    Noc m_noc;
    Dispatch *m_dispatch;
    std::vector<uint8_t> m_dispatch_data;
    const uint8_t *m_cmd_reg;
    const uint8_t *m_cmd_reg_end;
    const uint8_t *m_cmd_ptr;
    std::vector<uint32_t> m_l1_cache;
};

} // namespace dispatch
} // namespace device
} // namespace metal
} // namespace tt

