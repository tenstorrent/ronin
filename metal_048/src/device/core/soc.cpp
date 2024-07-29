
#include <cstdint>
#include <cassert>
#include <string>
#include <stdexcept>

#include "arch/soc_arch.hpp"

#include "core/soc.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

std::string xy_to_string(int x, int y) {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

} // namespace

//
//    Soc
//

Soc::Soc(SocArch *soc_arch) {
    init(soc_arch);
}

Soc::~Soc() { }

void Soc::logical_to_routing_coord(int logical_x, int logical_y, int &x, int &y) {
    x = m_soc_arch->worker_logical_to_routing_x(logical_x);
    y = m_soc_arch->worker_logical_to_routing_y(logical_y);
    if (x < 0 || y < 0) {
        throw std::runtime_error(
            "Invalid logical worker core coordinates " + 
                xy_to_string(logical_x, logical_y));
    }
}

uint32_t Soc::sysmem_size() {
    return m_sysmem.size();
}

uint8_t *Soc::map_sysmem_addr(uint32_t addr) {
    return m_sysmem.map_addr(addr);
}

uint32_t Soc::dram_size(int dram_channel) {
    return m_dram_banks[dram_channel]->size();
}

uint8_t *Soc::map_dram_addr(int dram_channel, uint32_t addr) {
    DramBank *dram_bank = m_dram_banks[dram_channel].get();
    return dram_bank->map_addr(addr);
}

uint32_t Soc::l1_size(int x, int y) {
    Memory *l1 = get_worker_l1(x, y);
    return l1->size();
}

uint8_t *Soc::map_l1_addr(int x, int y, uint32_t addr) {
    Memory *l1 = get_worker_l1(x, y);
    return l1->map_addr(addr);
}

Memory *Soc::get_worker_l1(int x, int y) {
    int xy = get_xy(x, y);
    Core *core = m_cores[xy].get();
    if (core == nullptr || core->core_type != CoreType::WORKER) {
        throw std::runtime_error("No worker core at " + xy_to_string(x, y));
    }
    return core->l1;
}

void Soc::set_worker_l1(int logical_x, int logical_y, Memory *memory) {
#if 0 // ACHTUNG: Temporary workaround to allow Whisper memory size adjustments
    assert(memory->size() == m_soc_arch->worker_l1_size());
#else
    assert(memory->size() >= m_soc_arch->worker_l1_size());
#endif
    int x, y;
    logical_to_routing_coord(logical_x, logical_y, x, y);
    CoreType core_type = m_soc_arch->core_type(x, y);
    if (core_type != CoreType::WORKER) {
        throw std::runtime_error("No worker core at " + xy_to_string(x, y));
    }
    int xy = get_xy(x, y);
    m_cores[xy]->l1 = memory;
}

void Soc::init(SocArch *soc_arch) {
    m_soc_arch = soc_arch;
    m_x_size = m_soc_arch->x_size();
    m_y_size = m_soc_arch->y_size();
    m_worker_x_size = m_soc_arch->worker_x_size();
    m_worker_y_size = m_soc_arch->worker_y_size();
    // make 'sysmem_size' configurable?
    uint32_t sysmem_size = 1024 * 1024 * 1024;
    m_sysmem.init(sysmem_size);
    int num_dram_banks = m_soc_arch->num_dram_channels();
    uint32_t dram_bank_size = m_soc_arch->dram_bank_size();
    m_dram_banks.resize(num_dram_banks);
    for (int i = 0; i < num_dram_banks; i++) {
        DramBank *dram_bank = new DramBank();
        dram_bank->init(dram_bank_size);
        m_dram_banks[i].reset(dram_bank);
    }
    uint32_t worker_l1_size = m_soc_arch->worker_l1_size();
    m_cores.resize(m_x_size * m_y_size);
    for (int x = 0; x < m_x_size; x++) {
        for (int y = 0; y < m_y_size; y++) {
            CoreType core_type = m_soc_arch->core_type(x, y);
            // so far, only worker cores are relevant
            if (core_type == CoreType::WORKER) {
                int xy = get_xy(x, y);
                Core *core = new Core();
                core->core_type = core_type;
                // deferred: will be set by 'set_worker_l1'
                core->l1 = nullptr;
                m_cores[xy].reset(core);
            }
        }
    }
}

} // namespace device
} // namespace metal
} // namespace tt

