// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>

#include "arch/soc_arch.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

std::string xy_to_string(int x, int y) {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

void verify(bool cond, const std::string &msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

} // namespace

//
//    SocArch
//

SocArch::SocArch():
        m_x_size(0),
        m_y_size(0),
        m_worker_l1_size(0),
        m_storage_core_l1_bank_size(0),
        m_dram_bank_size(0),
        m_eth_l1_size(0),
        m_num_dram_channels(0),
        m_worker_x_size(0),
        m_worker_y_size(0),
        m_compute_and_storage_x_size(0),
        m_compute_and_storage_y_size(0) { }

SocArch::~SocArch() { }

void SocArch::init(
        int x_size,
        int y_size,
        uint32_t worker_l1_size,
        uint32_t storage_core_l1_bank_size,
        uint32_t dram_bank_size,
        uint32_t eth_l1_size,
        int num_dram_channels) {
    m_x_size = x_size;
    m_y_size = y_size;
    m_worker_l1_size = worker_l1_size;
    m_storage_core_l1_bank_size = storage_core_l1_bank_size;
    m_dram_bank_size = dram_bank_size,
    m_eth_l1_size = eth_l1_size;
    m_num_dram_channels = num_dram_channels;
    m_core_types.resize(m_x_size * m_y_size, CoreType::INVALID);
    m_worker_core_types.resize(m_x_size * m_y_size, WorkerCoreType::NONE);
    m_dram_preferred_worker_endpoints.resize(m_num_dram_channels);
}

void SocArch::set_core_type(CoreType core_type, int x, int y) {
    verify(
        x >= 0 && x < m_x_size && y >= 0 && y < m_y_size,
        "Core coordinates " + xy_to_string(x, y) + " are out of range");
    int xy = get_xy(x, y);
    verify(
        m_core_types[xy] == CoreType::INVALID,
        "Core type at " + xy_to_string(x, y) + " is already set");
    m_core_types[xy] = core_type;
}

void SocArch::set_core_type(CoreType core_type, int x, int y0, int y1) {
    for (int y = y0; y <= y1; y++) {
        set_core_type(core_type, x, y);
    }
}

void SocArch::set_worker_core_type(WorkerCoreType worker_core_type, int x, int y) {
    verify(
        x >= 0 && x < m_x_size && y >= 0 && y < m_y_size,
        "Core coordinates " + xy_to_string(x, y) + " are out of range");
    int xy = get_xy(x, y);
    verify(
        m_core_types[xy] == CoreType::WORKER,
        "Core at " + xy_to_string(x, y) + " is not worker");
    verify(
        m_worker_core_types[xy] == WorkerCoreType::NONE,
        "Worker core type at " + xy_to_string(x, y) + " is already set");
    m_worker_core_types[xy] = worker_core_type;
}

void SocArch::set_worker_core_type(WorkerCoreType worker_core_type, int x, int y0, int y1) {
    for (int y = y0; y <= y1; y++) {
        set_worker_core_type(worker_core_type, x, y);
    }
}

void SocArch::set_dram_preferred_worker_endpoint(int dram_channel, int x, int y) {
    verify(
        dram_channel >= 0 && dram_channel < m_num_dram_channels,
        "DRAM channel " + std::to_string(dram_channel) + " is out of range");
    std::pair<int, int> &coord = m_dram_preferred_worker_endpoints[dram_channel];
    coord.first = x;
    coord.second = y;
}

void SocArch::finalize() {
    std::vector<bool> is_worker_x(m_x_size, false);
    std::vector<bool> is_worker_y(m_y_size, false);
    std::vector<bool> is_cs_x(m_x_size, false);
    std::vector<bool> is_cs_y(m_y_size, false);

    for (int x = 0; x < m_x_size; x++) {
        for (int y = 0; y < m_y_size; y++) {
            int xy = get_xy(x, y);
            if (m_core_types[xy] == CoreType::WORKER) {
                is_worker_x[x] = true;
                is_worker_y[y] = true;
                if (m_worker_core_types[xy] == WorkerCoreType::COMPUTE_AND_STORAGE) {
                    is_cs_x[x] = true;
                    is_cs_y[y] = true;
                }
            }
        }
    }

    m_worker_x_size = 0;
    m_worker_y_size = 0;
    m_compute_and_storage_x_size = 0;
    m_compute_and_storage_y_size = 0;
    m_worker_routing_to_logical_x.resize(m_x_size, -1);
    m_worker_routing_to_logical_y.resize(m_y_size, -1);

    for (int x = 0; x < m_x_size; x++) {
        if (is_worker_x[x]) {
            m_worker_routing_to_logical_x[x] = m_worker_x_size;
            m_worker_x_size++;
            if (is_cs_x[x]) {
                m_compute_and_storage_x_size++;
            }
        }
    }

    for (int y = 0; y < m_y_size; y++) {
        if (is_worker_y[y]) {
            m_worker_routing_to_logical_y[y] = m_worker_y_size;
            m_worker_y_size++;
            if (is_cs_y[y]) {
                m_compute_and_storage_y_size++;
            }
        }
    }

    m_worker_logical_to_routing_x.resize(m_worker_x_size);
    m_worker_logical_to_routing_y.resize(m_worker_y_size);

    for (int x = 0; x < m_x_size; x++) {
        int logical_x = m_worker_routing_to_logical_x[x];
        if (logical_x >= 0) {
            m_worker_logical_to_routing_x[logical_x] = x;
        }
    }

    for (int y = 0; y < m_y_size; y++) {
        int logical_y = m_worker_routing_to_logical_y[y];
        if (logical_y >= 0) {
            m_worker_logical_to_routing_y[logical_y] = y;
        }
    }
}

CoreType SocArch::core_type(int x, int y) const {
    verify(
        x >= 0 && x < m_x_size && y >= 0 && y < m_y_size,
        "Core coordinates " + xy_to_string(x, y) + " are out of range");
    int xy = get_xy(x, y);
    return m_core_types[xy];
}

WorkerCoreType SocArch::worker_core_type(int x, int y) const {
    verify(
        x >= 0 && x < m_x_size && y >= 0 && y < m_y_size,
        "Core coordinates " + xy_to_string(x, y) + " are out of range");
    int xy = get_xy(x, y);
    return m_worker_core_types[xy];
}

int SocArch::get_core_dram_channel(int x, int y) {
    // ACHTUNG: Only preferred workers are considered here
    //     It is assumed that callers always use preferred workers
    //     to access DRAM via NoC API
    int dram_channel = -1;
    int count = int(m_dram_preferred_worker_endpoints.size());
    for (int i = 0; i < count; i++) {
        const std::pair<int, int> &coord = m_dram_preferred_worker_endpoints[i];
        if (coord.first == x && coord.second == y) {
            dram_channel = i;
            break;
        }
    }
    verify(
        dram_channel >= 0,
        "DRAM channel not found for core at " + xy_to_string(x, y));
    return dram_channel;
}

void SocArch::get_dram_preferred_worker_endpoint(int dram_channel, int &x, int &y) const {
    verify(
        dram_channel >= 0 && dram_channel < m_num_dram_channels,
        "DRAM channel " + std::to_string(dram_channel) + " is out of range");
    const std::pair<int, int> &coord = m_dram_preferred_worker_endpoints[dram_channel];
    x = coord.first;
    y = coord.second;
}

int SocArch::worker_logical_to_routing_x(int logical_x) {
    verify(
        logical_x >= 0 && logical_x < m_worker_x_size,
        "Logical core x coordinate " + std::to_string(logical_x) + " is out of range");
    return m_worker_logical_to_routing_x[logical_x];
}

int SocArch::worker_logical_to_routing_y(int logical_y) {
    verify(
        logical_y >= 0 && logical_y < m_worker_y_size,
        "Logical core y coordinate " + std::to_string(logical_y) + " is out of range");
    return m_worker_logical_to_routing_y[logical_y];
}

int SocArch::worker_routing_to_logical_x(int x) {
    verify(
        x >= 0 && x < m_x_size,
        "Core x coordinate " + std::to_string(x) + " is out of range");
    return m_worker_routing_to_logical_x[x];
}

int SocArch::worker_routing_to_logical_y(int y) {
    verify(
        y >= 0 && y < m_y_size,
        "Core y coordinate " + std::to_string(y) + " is out of range");
    return m_worker_routing_to_logical_y[y];
}

} // namespace device
} // namespace metal
} // namespace tt

