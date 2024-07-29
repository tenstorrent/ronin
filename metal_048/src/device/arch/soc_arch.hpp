#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <utility>

namespace tt {
namespace metal {
namespace device {

enum class CoreType {
    ARC,
    DRAM,
    ETH,
    PCIE,
    WORKER,
    HARVESTED,
    ROUTER_ONLY,
    INVALID
}; 

enum class WorkerCoreType {
    NONE,
    COMPUTE_AND_STORAGE,
    STORAGE_ONLY,
    DISPATCH
};

class SocArch {
public:
    SocArch();
    ~SocArch();
public:
    void init(
        int x_size,
        int y_size,
        uint32_t worker_l1_size,
        uint32_t storage_core_l1_bank_size,
        uint32_t dram_bank_size,
        uint32_t eth_l1_size,
        int num_dram_channels);
    void set_core_type(CoreType core_type, int x, int y); 
    void set_core_type(CoreType core_type, int x, int y0, int y1); 
    void set_worker_core_type(WorkerCoreType worker_core_type, int x, int y); 
    void set_worker_core_type(WorkerCoreType worker_core_type, int x, int y0, int y1); 
    void set_dram_preferred_worker_endpoint(int dram_channel, int x, int y);
    void finalize();
public:
    int x_size() const {
        return m_x_size;
    }
    int y_size() const {
        return m_y_size;
    }
    uint32_t worker_l1_size() const {
        return m_worker_l1_size;
    }
    uint32_t storage_core_l1_bank_size() const {
        return m_storage_core_l1_bank_size;
    }
    uint32_t dram_bank_size() const {
        return m_dram_bank_size;
    }
    uint32_t eth_l1_size() const {
        return m_eth_l1_size;
    }
    int num_dram_channels() const {
        return m_num_dram_channels;
    }
    CoreType core_type(int x, int y) const;
    WorkerCoreType worker_core_type(int x, int y) const;
    int get_core_dram_channel(int x, int y);
    void get_dram_preferred_worker_endpoint(int dram_channel, int &x, int &y) const;
    int worker_x_size() const {
        return m_worker_x_size;
    }
    int worker_y_size() const {
        return m_worker_y_size;
    }
    int compute_and_storage_x_size() const {
        return m_compute_and_storage_x_size;
    }
    int compute_and_storage_y_size() const {
        return m_compute_and_storage_y_size;
    }
    int worker_logical_to_routing_x(int logical_x);
    int worker_logical_to_routing_y(int logical_y);
    int worker_routing_to_logical_x(int x);
    int worker_routing_to_logical_y(int y);
private:
    int get_xy(int x, int y) const {
        return x * m_y_size + y;
    }
private:
    int m_x_size;
    int m_y_size;
    uint32_t m_worker_l1_size;
    uint32_t m_storage_core_l1_bank_size;
    uint32_t m_dram_bank_size;
    uint32_t m_eth_l1_size;
    int m_num_dram_channels;
    std::vector<CoreType> m_core_types;
    std::vector<WorkerCoreType> m_worker_core_types;
    std::vector<std::pair<int, int>> m_dram_preferred_worker_endpoints;
    int m_worker_x_size;
    int m_worker_y_size;
    int m_compute_and_storage_x_size;
    int m_compute_and_storage_y_size;
    std::vector<int> m_worker_logical_to_routing_x;
    std::vector<int> m_worker_logical_to_routing_y;
    std::vector<int> m_worker_routing_to_logical_x;
    std::vector<int> m_worker_routing_to_logical_y;  
};

SocArch *get_soc_arch_grayskull();
SocArch *get_soc_arch_wormhole_b0();

} // namespace device
} // namespace metal
} // namespace tt

