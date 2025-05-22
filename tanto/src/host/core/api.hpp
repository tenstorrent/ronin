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
#include <exception>

namespace ronin {
namespace tanto {
namespace host {

class Platform;
class Device;
class Program;
class Grid;
class Global;
class Local;
class Pipe;
class Semaphore;
class Kernel;
class Queue;

class PlatformImpl;
class DeviceImpl;
class ProgramImpl;
class GridImpl;
class GlobalImpl;
class LocalImpl;
class PipeImpl;
class SemaphoreImpl;
class KernelImpl;
class QueueImpl;

enum class DataFormat {
    UINT8,
    UINT16,
    UINT32,
    FLOAT32,
    BFLOAT16
};

enum class GlobalDist {
    LINEAR,
    BLOCK,
    CYCLIC
};

enum class LocalScope {
    DEVICE,
    PROGRAM
};

enum class PipeKind {
    INPUT,
    OUTPUT,
    INTERMED
};

enum class KernelKind {
    READER,
    WRITER,
    MATH
};

enum class KernelFormat {
    METAL,
    TANTO
};

struct Range {
    uint32_t x_start;
    uint32_t y_start;
    uint32_t x_end;
    uint32_t y_end;
};

class Error: public std::exception {
public:
    Error(const char *msg);
    ~Error();
public:
    const char *what() const noexcept override;
private:
    const char *m_msg;
};

class Platform {
public:
    Platform();
    Platform(const Platform &other);
    Platform(Platform &&other) noexcept;
    explicit Platform(std::shared_ptr<PlatformImpl> &&impl);
    ~Platform();
public:
    Platform &operator=(const Platform &other);
    Platform &operator=(Platform &&other) noexcept;
    static Platform get_default();
    const std::shared_ptr<PlatformImpl> &impl() const {
        return m_impl;
    }
private:
    std::shared_ptr<PlatformImpl> m_impl;
};

class Device {
public:
    Device();
    Device(const Device &other);
    Device(Device &&other) noexcept;
    explicit Device(std::shared_ptr<DeviceImpl> &&impl);
    explicit Device(const Platform &platform, uint32_t id);
    ~Device();
public:
    Device &operator=(const Device &other);
    Device &operator=(Device &&other) noexcept;
    const std::shared_ptr<DeviceImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Platform platform() const;
    uint32_t id() const;
    void dram_grid_size(uint32_t &x, uint32_t &y) const;
    void worker_grid_size(uint32_t &x, uint32_t &y) const;
    void worker_core_from_logical_core(
        uint32_t logical_x,
        uint32_t logical_y,
        uint32_t &worker_x,
        uint32_t &worker_y) const;
    void close();
private:
    std::shared_ptr<DeviceImpl> m_impl;
};

class Program {
public:
    Program();
    Program(const Program &other);
    Program(Program &&other) noexcept;
    explicit Program(std::shared_ptr<ProgramImpl> &&impl);
    explicit Program(const Device &device);
    ~Program();
public:
    Program &operator=(const Program &other);
    Program &operator=(Program &&other) noexcept;
    const std::shared_ptr<ProgramImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Device device() const;
private:
    std::shared_ptr<ProgramImpl> m_impl;
};

class Grid {
public:
    Grid();
    Grid(const Grid &other);
    Grid(Grid &&other) noexcept;
    explicit Grid(std::shared_ptr<GridImpl> &&impl);
    explicit Grid(
        const Program &program, 
        uint32_t x, 
        uint32_t y);
    explicit Grid(
        const Program &program,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end, 
        uint32_t y_end);
    explicit Grid(
        const Program &program,
        const std::vector<Range> &ranges);
    ~Grid();
public:
    Grid &operator=(const Grid &other);
    Grid &operator=(Grid &&other) noexcept;
    const std::shared_ptr<GridImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Program program() const;
    int range_count() const;
    Range range_at(int index) const;
private:
    std::shared_ptr<GridImpl> m_impl;
};

class Global {
public:
    Global();
    Global(const Global &other);
    Global(Global &&other) noexcept;
    explicit Global(std::shared_ptr<GlobalImpl> &&impl);
    explicit Global(
        const Device &device,
        DataFormat data_format,
        uint32_t size,
        uint32_t log2_page_size);
    // DEPRECATED
    explicit Global(
        const Device &device,
        DataFormat data_format,
        bool is_dram,
        uint32_t size,
        uint32_t log2_page_size);
    // EXPERIMENTAL
    explicit Global(
        const Device &device,
        DataFormat data_format,
        GlobalDist dist,
        uint32_t size,
        uint32_t page_size);
    ~Global();
public:
    Global &operator=(const Global &other);
    Global &operator=(Global &&other) noexcept;
    const std::shared_ptr<GlobalImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Device device() const;
    DataFormat data_format() const;
    GlobalDist dist() const;
    bool is_dram() const;
    uint32_t size() const;
    uint32_t page_size() const;
    uint32_t log2_page_size() const;
    uint32_t bytes() const;
    uint32_t page_bytes() const;
private:
    std::shared_ptr<GlobalImpl> m_impl;
};

class Local {
public:
    Local();
    Local(const Local &other);
    Local(Local &&other) noexcept;
    explicit Local(std::shared_ptr<LocalImpl> &&impl);
    explicit Local(
        const Device &device,
        DataFormat data_format,
        uint32_t size);
    explicit Local(
        const Program &program,
        const Grid &grid,
        DataFormat data_format,
        uint32_t size);
    ~Local();
public:
    Local &operator=(const Local &other);
    Local &operator=(Local &&other) noexcept;
    const std::shared_ptr<LocalImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Device device() const;
    Program program() const;
    Grid grid() const;
    DataFormat data_format() const;
    uint32_t size() const;
    LocalScope scope() const;
private:
    std::shared_ptr<LocalImpl> m_impl;
};

class Pipe {
public:
    Pipe();
    Pipe(const Pipe &other);
    Pipe(Pipe &&other) noexcept;
    explicit Pipe(std::shared_ptr<PipeImpl> &&impl);
    explicit Pipe(
        const Program &program,
        const Grid &grid,
        PipeKind kind,
        DataFormat data_format,
        uint32_t size,
        uint32_t frame_size);
    ~Pipe();
public:
    Pipe &operator=(const Pipe &other);
    Pipe &operator=(Pipe &&other) noexcept;
    const std::shared_ptr<PipeImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Program program() const;
    Grid grid() const;
    PipeKind kind() const;
    DataFormat data_format() const;
    uint32_t size() const;
    uint32_t frame_size() const;
    void set_local(const Local &local) const;
private:
    std::shared_ptr<PipeImpl> m_impl;
};

class Semaphore {
public:
    Semaphore();
    Semaphore(const Semaphore &other);
    Semaphore(Semaphore &&other) noexcept;
    explicit Semaphore(std::shared_ptr<SemaphoreImpl> &&impl);
    explicit Semaphore(
        const Program &program,
        const Grid &grid, 
        uint32_t init_value);
    ~Semaphore();
public:
    Semaphore &operator=(const Semaphore &other);
    Semaphore &operator=(Semaphore &&other) noexcept;
    const std::shared_ptr<SemaphoreImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Program program() const;
    Grid grid() const;
    uint32_t init_value() const;
private:
    std::shared_ptr<SemaphoreImpl> m_impl;
};

using KernelArg = std::variant<uint32_t, Global, Local, Pipe, Semaphore>;

class Kernel {
public:
    Kernel();
    Kernel(const Kernel &other);
    Kernel(Kernel &&other) noexcept;
    explicit Kernel(std::shared_ptr<KernelImpl> &&impl);
    explicit Kernel(
        const Program &program,
        const Grid &grid,
        KernelKind kind,
        KernelFormat format,
        const std::string &path,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines);
    ~Kernel();
public:
    Kernel &operator=(const Kernel &other);
    Kernel &operator=(Kernel &&other) noexcept;
    const std::shared_ptr<KernelImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Program program() const;
    Grid grid() const;
    KernelKind kind() const;
    KernelFormat format() const;
    std::string path() const;
    const std::vector<uint32_t> &compile_args() const;
    const std::map<std::string, std::string> &defines() const;
    void set_args(
        uint32_t x,
        uint32_t y,
        const std::vector<KernelArg> &args) const;
    void set_args(
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end,
        const std::vector<KernelArg> &args) const;
    void set_args(const Grid &grid, const std::vector<KernelArg> &args) const;
private:
    std::shared_ptr<KernelImpl> m_impl;
};

class Queue {
public:
    Queue();
    Queue(const Queue &other);
    Queue(Queue &&other) noexcept;
    explicit Queue(std::shared_ptr<QueueImpl> &&impl);
    explicit Queue(const Device &device, uint32_t id);
    ~Queue();
public:
    Queue &operator=(const Queue &other);
    Queue &operator=(Queue &&other) noexcept;
    const std::shared_ptr<QueueImpl> &impl() const {
        return m_impl;
    }
    bool is_null() const {
        return (m_impl == nullptr);
    }
    Device device() const;
    uint32_t id() const;
    void enqueue_read(
        const Global &global, 
        void *dst,
        bool blocking) const;
    void enqueue_write(
        const Global &global, 
        const void *src,
        bool blocking) const;
    void enqueue_program(const Program &program, bool blocking) const;
    void finish() const;
private:
    std::shared_ptr<QueueImpl> m_impl;
};

} // namespace host
} // namespace tanto
} // namespace ronin

