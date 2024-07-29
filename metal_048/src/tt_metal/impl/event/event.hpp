// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
//#include <thread> // [RONIN]

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"

namespace tt::tt_metal
{
    class Device;
    struct Event
    {
        Device * device = nullptr;
        uint32_t cq_id = -1;
        uint32_t event_id = -1;
        std::atomic<bool> ready = false; // Event is ready for use.

        void wait_until_ready() {
#if 0 // TODO: Revise this
            while (!ready) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                log_trace(tt::LogMetal, "Waiting for Event to be ready. (ready: {} cq_id: {} event_id: {})", ready, cq_id, event_id);
            }
#endif
            // [RONIN] Events are populated on Enqueue Record Event
            TT_ASSERT(ready, "Event must be ready");

            TT_ASSERT(device != nullptr, "Event must have initialized device ptr");
            TT_ASSERT(event_id != -1, "Event must have initialized event_id");
            TT_ASSERT(cq_id != -1, "Event must have initialized cq_id");
        }
    };
}
