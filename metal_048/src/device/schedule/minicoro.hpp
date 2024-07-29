#pragma once

#include <functional>

#include "vendor/minicoro/minicoro.h"

namespace tt {
namespace metal {
namespace device {
namespace schedule {

class Coro {
public:
    Coro(std::function<void ()> func);
    ~Coro();
public:
    void reset(size_t stack_size);
    void clear();
    void resume();
    void yield();
    bool is_alive();
private:
    static void wrap_func(mco_coro *co);
private:
    std::function<void ()> m_func;
    bool m_valid;
    mco_coro *m_co;
};

} // namespace schedule
} // namespace device
} // namespace metal
} // namespace tt

