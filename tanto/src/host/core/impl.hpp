// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <variant>
#include <mutex>

#include "core/api.hpp"
#include "core/metal.hpp"

namespace ronin {
namespace tanto {
namespace host {

class PlatformImpl {
public:
    PlatformImpl();
    ~PlatformImpl();
public:
    static std::shared_ptr<PlatformImpl> get_default();
    void add_device(const std::shared_ptr<DeviceImpl> &device);
    std::shared_ptr<DeviceImpl> find_device(uint32_t id);
private:
    static void make_default();
private:
    static std::once_flag m_default_flag; 
    static std::shared_ptr<PlatformImpl> m_default;
    std::vector<std::shared_ptr<DeviceImpl>> m_devices;
};

class DeviceImpl {
public:
    DeviceImpl(const std::shared_ptr<PlatformImpl> &platform, uint32_t id);
    ~DeviceImpl();
public:
    static std::shared_ptr<DeviceImpl> create(
        const std::shared_ptr<PlatformImpl> &platform, uint32_t id);
    std::shared_ptr<PlatformImpl> platform() {
        return m_platform.lock();
    }
    uint32_t id() {
        return m_id;
    }
    void add_global(const std::shared_ptr<GlobalImpl> &global);
    void add_local(const std::shared_ptr<LocalImpl> &local);
    void add_program(const std::shared_ptr<ProgramImpl> &program);
    void add_queue(const std::shared_ptr<QueueImpl> &queue);
    const std::shared_ptr<QueueImpl> find_queue(uint32_t id);
    void dram_grid_size(uint32_t &x, uint32_t &y);
    void worker_grid_size(uint32_t &x, uint32_t &y);
    void worker_core_from_logical_core(
        uint32_t logical_x,
        uint32_t logical_y,
        uint32_t &worker_x,
        uint32_t &worker_y);
    void close();
#ifdef METAL_057
    metal::IDevice *impl() {
        return m_impl;
    }
#else
     metal::Device *impl() {
        return m_impl;
    }
 #endif
 void validate_impl();
private:
    std::weak_ptr<PlatformImpl> m_platform;
    uint32_t m_id;
    std::vector<std::shared_ptr<GlobalImpl>> m_globals;
    std::vector<std::shared_ptr<LocalImpl>> m_locals;
    std::vector<std::shared_ptr<ProgramImpl>> m_programs;
    std::vector<std::shared_ptr<QueueImpl>> m_queues;
    // TODO: Figure out whether smart pointer is needed here
#ifdef METAL_057
    metal::IDevice *m_impl;
#else
     metal::Device *m_impl;
#endif
};

class ProgramImpl {
public:
    ProgramImpl(const std::shared_ptr<DeviceImpl> &device);
    ~ProgramImpl();
public:
    static std::shared_ptr<ProgramImpl> create(const std::shared_ptr<DeviceImpl> &device);
    std::shared_ptr<DeviceImpl> device() {
        return m_device.lock();
    }
    void add_grid(const std::shared_ptr<GridImpl> &grid);
    void add_local(const std::shared_ptr<LocalImpl> &local);
    void add_pipe(const std::shared_ptr<PipeImpl> &pipe);
    void add_semaphore(const std::shared_ptr<SemaphoreImpl> &semaphore);
    void add_kernel(const std::shared_ptr<KernelImpl> &kernel);
    metal::Program &impl() {
        return m_impl;
    }
    void before_enqueue();
    void after_enqueue();
private:
    void create_impl();
private:
    std::weak_ptr<DeviceImpl> m_device;
    std::vector<std::shared_ptr<GridImpl>> m_grids;
    std::vector<std::shared_ptr<LocalImpl>> m_locals;
    std::vector<std::shared_ptr<PipeImpl>> m_pipes;
    std::vector<std::shared_ptr<SemaphoreImpl>> m_semaphores;
    std::vector<std::shared_ptr<KernelImpl>> m_kernels;
    metal::Program m_impl;
};

class GridImpl {
public:
    GridImpl(const std::shared_ptr<ProgramImpl> &program);
    ~GridImpl();
public:
    static const std::shared_ptr<GridImpl> create(
        const std::shared_ptr<ProgramImpl> &program,
        uint32_t x,
        uint32_t y);
    static const std::shared_ptr<GridImpl> create(
        const std::shared_ptr<ProgramImpl> &program,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end);
    static const std::shared_ptr<GridImpl> create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::vector<Range> &ranges);
    std::shared_ptr<ProgramImpl> program() {
        return m_program.lock();
    }
    int range_count();
    Range range_at(int index);
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &impl() const {
        return m_impl;
    }
    uint32_t make_cbid(PipeKind pipe_kind);
private:
    void validate_range(const Range &range);
    void create_impl();
private:
    static constexpr uint32_t
        START_INPUT_CBID = 0,
        END_INPUT_CBID = START_INPUT_CBID + 8,
        START_OUTPUT_CBID = 16,
        END_OUTPUT_CBID = START_OUTPUT_CBID + 8,
        START_INTERMED_CBID = 24,
        END_INTERMED_CBID = START_INTERMED_CBID + 8;
private:
    std::weak_ptr<ProgramImpl> m_program;
    std::vector<Range> m_ranges;
    std::variant<CoreCoord, CoreRange, CoreRangeSet> m_impl;
    uint32_t m_next_input_cbid;
    uint32_t m_next_output_cbid;
    uint32_t m_next_intermed_cbid;
};

class GlobalImpl {
public:
    GlobalImpl(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        GlobalDist dist,
        bool is_dram,
        uint32_t size,
        uint32_t page_size);
    ~GlobalImpl();
public:
    static std::shared_ptr<GlobalImpl> create(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        bool is_dram,
        uint32_t size,
        uint32_t log2_page_size);
    static std::shared_ptr<GlobalImpl> create(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        GlobalDist dist,
        uint32_t size,
        uint32_t page_size);
    std::shared_ptr<DeviceImpl> device() {
        return m_device.lock();
    }
    DataFormat data_format() {
        return m_data_format;
    }
    GlobalDist dist() {
        return m_dist;
    }
    bool is_dram() {
        return m_is_dram;
    }
    uint32_t size() {
        return m_size;
    }
    uint32_t page_size() {
        return m_page_size;
    }
    uint32_t log2_page_size();
    const std::shared_ptr<metal::Buffer> &impl() const {
        return m_impl;
    }
    uint32_t bytes();
    uint32_t page_bytes();
private:
    static void validate_dist_size(
        GlobalDist dist,
        uint32_t size,
        uint32_t page_size);
    void create_impl();
    void create_impl_linear();
    void create_impl_dist();
private:
    std::weak_ptr<DeviceImpl> m_device;
    DataFormat m_data_format;
    GlobalDist m_dist;
    bool m_is_dram;
    uint32_t m_size;
    uint32_t m_page_size;
    std::shared_ptr<metal::Buffer> m_impl;
};

class LocalImpl {
public:
    LocalImpl(
        const std::shared_ptr<DeviceImpl> &device,
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        DataFormat data_format,
        uint32_t size,
        LocalScope scope);
    ~LocalImpl();
public:
    static std::shared_ptr<LocalImpl> create(
        const std::shared_ptr<DeviceImpl> &device,
        DataFormat data_format,
        uint32_t size);
    static std::shared_ptr<LocalImpl> create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        DataFormat data_format,
        uint32_t size);
    std::shared_ptr<DeviceImpl> device() {
        return m_device.lock();
    }
    std::shared_ptr<ProgramImpl> program() {
        return m_program.lock();
    }
    std::shared_ptr<GridImpl> grid() {
        return m_grid;
    }
    DataFormat data_format() {
        return m_data_format;
    }
    uint32_t size() {
        return m_size;
    }
    LocalScope scope() {
        return m_scope;
    }
    const std::shared_ptr<metal::Buffer> &impl() const {
        return m_impl;
    }
    void create_impl();
    void release_impl();
private:
    std::weak_ptr<DeviceImpl> m_device;
    std::weak_ptr<ProgramImpl> m_program;
    std::shared_ptr<GridImpl> m_grid;
    DataFormat m_data_format;
    uint32_t m_size;
    LocalScope m_scope;
    std::shared_ptr<metal::Buffer> m_impl;
};

class PipeImpl {
public:
    PipeImpl(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        PipeKind kind,
        DataFormat data_format,
        uint32_t size,
        uint32_t frame_size);
    ~PipeImpl();
public:
    static std::shared_ptr<PipeImpl> create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        PipeKind kind,
        DataFormat data_format,
        uint32_t size,
        uint32_t frame_size);
    std::shared_ptr<ProgramImpl> program() {
        return m_program.lock();
    }
    std::shared_ptr<GridImpl> grid() {
        return m_grid;
    }
    PipeKind kind() {
        return m_kind;
    }
    DataFormat data_format() {
        return m_data_format;
    }
    uint32_t size() {
        return m_size;
    }
    uint32_t frame_size() {
        return m_frame_size;
    }
    uint32_t cbid() {
        return m_cbid;
    }
    void set_local(const std::shared_ptr<LocalImpl> &local);
    metal::CBHandle impl() {
        return m_impl;
    }
    void update_dynamic_address();
private:
    void create_impl();
private:
    std::weak_ptr<ProgramImpl> m_program;
    std::shared_ptr<GridImpl> m_grid;
    PipeKind m_kind;
    DataFormat m_data_format;
    uint32_t m_size;
    uint32_t m_frame_size;
    uint32_t m_cbid;
    std::shared_ptr<LocalImpl> m_local;
    metal::CBHandle m_impl;
};

class SemaphoreImpl {
public:
    SemaphoreImpl(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        uint32_t init_value);
    ~SemaphoreImpl();
public:
    static std::shared_ptr<SemaphoreImpl> create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        uint32_t init_value);
    std::shared_ptr<ProgramImpl> program() {
        return m_program.lock();
    }
    std::shared_ptr<GridImpl> grid() {
        return m_grid;
    }
    uint32_t init_value() {
        return m_init_value;
    }
    uint32_t impl() {
        return m_impl;
    }
private:
    void create_impl();
private:
    std::weak_ptr<ProgramImpl> m_program;
    std::shared_ptr<GridImpl> m_grid;
    uint32_t m_init_value;
    uint32_t m_impl;
};

class KernelImpl {
public:
    KernelImpl(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        KernelKind kind,
        KernelFormat format,
        const std::string &path,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines);
    ~KernelImpl();
public:
    static std::shared_ptr<KernelImpl> create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        KernelKind kind,
        KernelFormat format,
        const std::string &path,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines);
    std::shared_ptr<ProgramImpl> program() {
        return m_program.lock();
    }
    std::shared_ptr<GridImpl> grid() {
        return m_grid;
    }
    KernelKind kind() {
        return m_kind;
    }
    KernelFormat format() {
        return m_format;
    }
    std::string path() {
        return m_path;
    }
    const std::vector<uint32_t> &compile_args() {
        return m_compile_args;
    }
    const std::map<std::string, std::string> &defines() {
        return m_defines;
    }
    void set_args(
        uint32_t x,
        uint32_t y,
        const std::vector<KernelArg> &args);
    void set_args(
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end,
        const std::vector<KernelArg> &args);
    void set_args(
        const std::shared_ptr<GridImpl> &grid, 
        const std::vector<KernelArg> &args);
    metal::KernelHandle impl() {
        return m_impl;
    }
    void set_args_impl();
private:
    void set_args(const Range &range, const std::vector<KernelArg> &args);
    void validate_range(const Range &range);
    void create_impl();
    static std::shared_ptr<metal::RuntimeArgs> 
        make_args_impl(const std::vector<KernelArg> &args);
    static void update_args_impl(
        const std::vector<KernelArg> &args,
        metal::RuntimeArgs &args_impl);
private:
    struct RangeArgs {
        Range range;
        std::vector<KernelArg> args;
        std::shared_ptr<metal::RuntimeArgs> args_impl;
    };
private:
    std::weak_ptr<ProgramImpl> m_program;
    std::shared_ptr<GridImpl> m_grid;
    KernelKind m_kind;
    KernelFormat m_format;
    std::string m_path;
    std::vector<uint32_t> m_compile_args;
    std::map<std::string, std::string> m_defines;
    std::vector<RangeArgs> m_ranges_args;
    metal::KernelHandle m_impl;
};

class QueueImpl {
public:
    QueueImpl(const std::shared_ptr<DeviceImpl> &device, uint32_t id);
    ~QueueImpl();
public:
    static std::shared_ptr<QueueImpl> create(
        const std::shared_ptr<DeviceImpl> &device, uint32_t id);
    std::shared_ptr<DeviceImpl> device() {
        return m_device.lock();
    }
    uint32_t id() {
        return m_id;
    }
    void enqueue_read(
        const std::shared_ptr<GlobalImpl> &global, 
        void *dst,
        bool blocking);
    void enqueue_write(
        const std::shared_ptr<GlobalImpl> &global, 
        const void *src,
        bool blocking);
    void enqueue_program(const std::shared_ptr<ProgramImpl> &program, bool blocking);
    void finish();
private:
    void create_impl();
private:
    std::weak_ptr<DeviceImpl> m_device;
    uint32_t m_id;
    metal::CommandQueue *m_impl;
};

} // namespace host
} // namespace tanto
} // namespace ronin

