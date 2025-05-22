// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include "test/util/timer.hpp"

namespace ronin {
namespace op {
namespace common {
namespace test {
namespace util {

//
//    Timer
//

Timer::Timer(): 
        m_elapsed(0.0f) { }

Timer::~Timer() { }

void Timer::reset() {
    m_elapsed = 0.0f;
}

void Timer::start() {
    m_start = std::chrono::steady_clock::now();
}

void Timer::stop() {
    m_end = std::chrono::steady_clock::now();
    m_elapsed +=
        std::chrono::duration_cast<
            std::chrono::duration<float, std::milli>>(m_end - m_start).count();
}

float Timer::elapsed() {
    return m_elapsed;
}

} // namespace util
} // namespace test
} // namespace common
} // namespace op
} // namespace ronin

