// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Always synchronous mode

#include <functional>

#include "common/env_lib.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

#if defined(TRACY_ENABLE)
#define TracyTTThreadName(name, id)                     \
    std::string tmp = fmt::format("{} : {}", name, id); \
    tracy::SetThreadName(tmp.c_str());
#else
#define TracyTTThreadName(name, id)
#endif

namespace tt {

enum class WorkExecutorMode {
    SYNCHRONOUS = 0,
    ASYNCHRONOUS = 1,
};

enum class WorkerQueueMode {
    LOCKFREE = 0,
    LOCKBASED = 1,
};

enum class WorkerState {
    RUNNING = 0,
    TERMINATE = 1,
    IDLE = 2,
};

// Synchronous mode only
class WorkExecutor {
public:
    WorkExecutor(int cpu_core, int device_id): 
            cpu_core_for_worker(cpu_core), 
            managed_device_id(device_id) { }

    WorkExecutor(WorkExecutor &&other) {
        worker_state = std::move(other.worker_state);
        cpu_core_for_worker = std::move(other.managed_device_id);
        managed_device_id = std::move(other.managed_device_id);
    }

    WorkExecutor& operator=(WorkExecutor &&other) {
        if (this != &other) {
            worker_state = std::move(other.worker_state);
            managed_device_id = std::move(other.managed_device_id);
            cpu_core_for_worker = std::move(other.cpu_core_for_worker);
        }
        return *this;
    }

    ~WorkExecutor() { }

    inline void initialize() {
        this->work_executor_mode = WorkExecutorMode::SYNCHRONOUS;
        this->worker_queue_mode = WorkerQueueMode::LOCKFREE;
        this->worker_state = WorkerState::IDLE;
    }

    inline void reset() {
        this->work_executor_mode = WorkExecutorMode::SYNCHRONOUS;
    }

    inline void run_worker() {
        // Must not be called at all?
    }

    inline void push_work(const std::function<void()> &work_executor, bool blocking = false) {
        work_executor();
    }

    inline void push_work(std::shared_ptr<std::function<void()>> work_executor, bool blocking = false) {
        // Execute work in current thread.
        (*work_executor)();
    }

    inline void synchronize() { }

    inline void set_worker_mode(const WorkExecutorMode &mode) { }

    WorkExecutorMode get_worker_mode() { return work_executor_mode; }

    inline void set_worker_queue_mode(const WorkerQueueMode &mode) { }

    WorkerQueueMode get_worker_queue_mode() { return worker_queue_mode; }

    inline std::size_t get_parent_thread_id() { return 0; }

private:
    WorkerState worker_state = WorkerState::IDLE;
    int cpu_core_for_worker = 0;
    int managed_device_id = 0;

    WorkExecutorMode work_executor_mode = WorkExecutorMode::SYNCHRONOUS;
    WorkerQueueMode worker_queue_mode = WorkerQueueMode::LOCKFREE;
};

}  // namespace tt
