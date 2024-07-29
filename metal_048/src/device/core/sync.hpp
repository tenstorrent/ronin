#pragma once

#include <cassert>
#include <functional>

#include "schedule/schedule.hpp"

namespace tt {
namespace metal {
namespace device {

using schedule::Scheduler;

//
//    Synchronization
//

class Sync {
public:
    Sync(Scheduler *scheduler):
        m_scheduler(scheduler) { }
    ~Sync() { }
public:
    void wait(std::function<bool ()> condition) {
        assert(m_scheduler != nullptr);
        m_scheduler->wait(condition);
    }
private:
    Scheduler *m_scheduler;
};

} // namespace device
} // namespace metal
} // namespace tt

