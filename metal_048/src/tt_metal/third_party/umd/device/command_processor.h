#pragma once

#include <cstdint>

// [RONIN]

class CommandProcessor {
public:
    CommandProcessor() { }
    virtual ~CommandProcessor() { }
public:
    virtual void configure_read_buffer(
        uint32_t padded_page_size,
        void *dst,
        uint32_t dst_offset,
        uint32_t num_pages_read) = 0;
    virtual void run_commands(const uint8_t *cmd_reg, uint32_t cmd_seq_size) = 0;
    virtual void launch_kernels() = 0;
};

