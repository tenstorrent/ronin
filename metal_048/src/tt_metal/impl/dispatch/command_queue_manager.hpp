#pragma once

#include <cstdint>
#include <vector>

#include "tt_metal/third_party/umd/device/command_processor.h"

namespace tt {
namespace tt_metal {

class CQManager {
public:
    CQManager(CommandProcessor *command_processor, uint32_t num_hw_cqs);
    ~CQManager();
public:
    void *reserve(uint32_t size);
    void write(const void *data, uint32_t size, uint32_t write_ptr);
    void push(uint32_t size);
    void config_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read);
    void launch_kernels();
    void set_bypass_mode(bool enable) {
        m_bypass_enable = enable;
    }
    bool get_bypass_mode() {
        return m_bypass_enable;
    }
    std::vector<uint32_t> get_bypass_data();
    void reset_event_id(uint32_t cq_id);
    uint32_t get_next_event(uint32_t cq_id);
private:
    CommandProcessor *m_command_processor;
    std::vector<char> m_data;
    bool m_bypass_enable;
    std::vector<uint32_t> m_bypass_buffer;
    std::vector<uint32_t> m_next_event_id;
};

} // namespace tt_metal
} // namespace tt

