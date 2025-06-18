// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <array>
#include <memory>

#include "arch/soc_arch.hpp"

#include "core/memory.hpp"

namespace tt {
namespace metal {
namespace device {

class Soc {
public:
    Soc(SocArch *soc_arch);
    ~Soc();
public:
    uint32_t worker_l1_size() const {
        return m_soc_arch->worker_l1_size();
    }
    uint32_t dram_bank_size() const {
        return m_soc_arch->dram_bank_size();
    }
    CoreType core_type(int x, int y) const {
        return m_soc_arch->core_type(x, y);
    }
    WorkerCoreType worker_core_type(int x, int y) const {
        return m_soc_arch->worker_core_type(x, y);
    }
    int core_dram_channel(int x, int y) {
        return m_soc_arch->get_core_dram_channel(x, y);
    }
    int worker_x_size() const {
        return m_worker_x_size;
    }
    int worker_y_size() const {
        return m_worker_y_size;
    }
    void logical_to_routing_coord(int logical_x, int logical_y, int &x, int &y);
    uint32_t sysmem_size();
    uint8_t *map_sysmem_addr(uint32_t addr);
    uint32_t dram_size(int dram_channel);
    uint8_t *map_dram_addr(int dram_channel, uint32_t addr);
    uint32_t l1_size(int x, int y);
    uint8_t *map_l1_addr(int x, int y, uint32_t addr);
    Memory *get_worker_l1(int x, int y);
    void set_worker_l1(int logical_x, int logical_y, Memory *memory);
private:
    void init(SocArch *soc_arch);
    int get_xy(int x, int y) {
        return x * m_y_size + y;
    }
private:
    struct Core {
        CoreType core_type;
        Memory *l1;
    };
private:
    SocArch *m_soc_arch;
    int m_x_size;
    int m_y_size;
    int m_worker_x_size;
    int m_worker_y_size;
    DramBank m_sysmem;
    std::vector<std::unique_ptr<DramBank>> m_dram_banks;
    std::vector<std::unique_ptr<Core>> m_cores;
};

} // namespace device
} // namespace metal
} // namespace tt

