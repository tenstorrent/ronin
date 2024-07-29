
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <utility>

#include "tt_metal/third_party/umd/device/command_processor.h"

#include "tt_metal/impl/dispatch/command_queue_manager.hpp"

namespace tt {
namespace tt_metal {

// [RONIN] Replaces original "command_queue_interface" mechanics

//
//    CQManager
//

CQManager::CQManager(CommandProcessor *command_processor, uint32_t num_hw_cqs):
        m_command_processor(command_processor),
        m_bypass_enable(false),
        m_next_event_id(num_hw_cqs, 0) { 
    m_data.reserve(128 * 1024);        
}

CQManager::~CQManager() { }

void *CQManager::reserve(uint32_t size) {
    m_data.resize(size);
    return m_data.data();
}

void CQManager::write(const void *data, uint32_t size, uint32_t write_ptr) {
    assert(write_ptr + size <= m_data.size());
    memcpy(m_data.data() + write_ptr, data, size);
}

void CQManager::push(uint32_t size) {
    assert(size <= m_data.size());
    uint8_t *data = reinterpret_cast<uint8_t *>(m_data.data());
    m_command_processor->run_commands(data, size);
    m_data.clear();
}

void CQManager::config_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) {
    m_command_processor->configure_read_buffer(
        padded_page_size,
        dst,
        dst_offset,
        num_pages_read);
}

void CQManager::launch_kernels() {
    m_command_processor->launch_kernels();
}

std::vector<uint32_t> CQManager::get_bypass_data() { 
    return std::move(m_bypass_buffer); 
}

void CQManager::reset_event_id(uint32_t cq_id) {
    m_next_event_id[cq_id] = 0;
}

uint32_t CQManager::get_next_event(uint32_t cq_id) {
    uint32_t event_id = m_next_event_id[cq_id];
    m_next_event_id[cq_id]++;
    return event_id;
}

} // namespace tt_metal
} // namespace tt

