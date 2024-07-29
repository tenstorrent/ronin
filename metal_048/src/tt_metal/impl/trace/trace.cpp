// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/trace/trace.hpp"

#include <memory>
#include <string>

#include "dispatch/device_command.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/trace/trace.hpp"

#ifdef _MSC_VER // [RONIN]
namespace {

inline uint32_t __builtin_clz(uint32_t x) {
    unsigned long y = 0;
    if (_BitScanReverse(&y, (unsigned long)x)) {
        return 32 - uint32_t(y);
    } else {
        return 32;
    }
}

} // namespace
#endif

namespace {
// Labels to make the code more readable
static constexpr bool kBlocking = true;
static constexpr bool kNonBlocking = false;

// Min size is bounded by NOC transfer efficiency
// Max size is bounded by Prefetcher CmdDatQ size
static constexpr uint32_t kExecBufPageMin = 1024;
static constexpr uint32_t kExecBufPageMax = 4096;

// Assumes pages are interleaved across all banks starting at 0
size_t interleaved_page_size(
    const uint32_t buf_size, const uint32_t num_banks, const uint32_t min_size, const uint32_t max_size) {
    // Populate power of 2 numbers within min and max as candidates
    TT_FATAL(min_size > 0 and min_size <= max_size);
    vector<uint32_t> candidates;
    candidates.reserve(__builtin_clz(min_size) - __builtin_clz(max_size) + 1);
    for (uint32_t size = 1; size <= max_size; size <<= 1) {
        if (size >= min_size) {
            candidates.push_back(size);
        }
    }
    uint32_t min_waste = -1;
    uint32_t pick = 0;
    // Pick the largest size that minimizes waste
    for (const uint32_t size : candidates) {
        // Pad data to the next fully banked size
        uint32_t fully_banked = num_banks * size;
        uint32_t padded_size = (buf_size + fully_banked - 1) / fully_banked * fully_banked;
        uint32_t waste = padded_size - buf_size;
        if (waste <= min_waste) {
            min_waste = waste;
            pick = size;
        }
    }
    TT_FATAL(pick >= min_size and pick <= max_size);
    return pick;
}
}  // namespace

namespace tt::tt_metal {

std::atomic<uint32_t> Trace::global_trace_id = 0;

uint32_t Trace::next_id() {
    return global_trace_id++;
}

std::shared_ptr<TraceBuffer> Trace::create_trace_buffer(
    const CommandQueue& cq, shared_ptr<detail::TraceDescriptor> desc, uint32_t unpadded_size) {
    size_t page_size = interleaved_page_size(
        unpadded_size, cq.device()->num_banks(BufferType::DRAM), kExecBufPageMin, kExecBufPageMax);
    uint64_t padded_size = round_up(unpadded_size, page_size);

    // Commit the trace buffer to device DRAM
    return std::make_shared<TraceBuffer>(
        desc,
        std::make_shared<Buffer>(
            cq.device(), padded_size, page_size, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED));
}

void Trace::initialize_buffer(CommandQueue& cq, std::shared_ptr<TraceBuffer> trace_buffer) {
    vector<uint32_t>& data = trace_buffer->desc->data;

    uint64_t unpadded_size = data.size() * sizeof(uint32_t);
    TT_FATAL(
        unpadded_size <= trace_buffer->buffer->size(),
        "Trace data size {} is larger than specified trace buffer size {}. Increase specified buffer size.",
        unpadded_size,
        trace_buffer->buffer->size());
    size_t numel_padding = (trace_buffer->buffer->size() - unpadded_size) / sizeof(uint32_t);
    if (numel_padding > 0) {
        data.resize(data.size() + numel_padding, 0 /*padding value*/);
    }
    uint64_t padded_size = data.size() * sizeof(uint32_t);
    EnqueueWriteBuffer(cq, trace_buffer->buffer, data, kBlocking);

    log_trace(
        LogMetalTrace,
        "Trace issue buffer unpadded size={}, padded size={}, num_pages={}",
        unpadded_size,
        padded_size,
        trace_buffer->buffer->num_pages());
}

// there is a cost to validation, please use it judiciously
void Trace::validate_instance(const TraceBuffer& trace_buffer) {
    vector<uint32_t> backdoor_data;
    detail::ReadFromBuffer(trace_buffer.buffer, backdoor_data);
    if (backdoor_data != trace_buffer.desc->data) {
        log_info(LogMetalTrace, "Trace buffer expected: {}", trace_buffer.desc->data);
        log_info(LogMetalTrace, "Trace buffer observed: {}", backdoor_data);
    }
    // add more checks
}

}  // namespace tt::tt_metal
