// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

class Timer {
public:
    Timer();
    ~Timer();
public:
    void reset();
    void start();
    void stop();
    float elapsed();
private:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;
    float m_elapsed;
};

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

