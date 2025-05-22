// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"

namespace ronin {
namespace tanto {
namespace host {

//
//    Error
//

Error::Error(const char *msg):
        m_msg(msg) { }

Error::~Error() { }

const char *Error::what() const noexcept {
    return (m_msg != nullptr) ? m_msg : "<unknown>";
}

//
//    Platform
//

Platform::Platform() { }

Platform::Platform(const Platform &other):
        m_impl(other.m_impl) { }

Platform::Platform(Platform &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Platform::Platform(std::shared_ptr<PlatformImpl> &&impl):
        m_impl(std::move(impl)) { }

Platform::~Platform() { }

Platform &Platform::operator=(const Platform &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Platform &Platform::operator=(Platform &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Platform Platform::get_default() {
    return Platform(PlatformImpl::get_default());
}

//
//    Device
//

Device::Device() { }

Device::Device(const Device &other):
        m_impl(other.m_impl) { }

Device::Device(Device &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Device::Device(std::shared_ptr<DeviceImpl> &&impl):
        m_impl(std::move(impl)) { }

Device::Device(const Platform &platform, uint32_t id):
        m_impl(DeviceImpl::create(platform.impl(), id)) { }

Device::~Device() { }

Device &Device::operator=(const Device &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Device &Device::operator=(Device &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Platform Device::platform() const {
    return Platform(m_impl->platform());
}

uint32_t Device::id() const {
    return m_impl->id();
}

void Device::dram_grid_size(uint32_t &x, uint32_t &y) const {
    m_impl->dram_grid_size(x, y);
}

void Device::worker_grid_size(uint32_t &x, uint32_t &y) const {
    m_impl->worker_grid_size(x, y);
}

void Device::worker_core_from_logical_core(
        uint32_t logical_x,
        uint32_t logical_y,
        uint32_t &worker_x,
        uint32_t &worker_y) const {
    m_impl->worker_core_from_logical_core(logical_x, logical_y, worker_x, worker_y);
}

void Device::close() {
    m_impl->close();
}

//
//    Program
//

Program::Program() { }

Program::Program(const Program &other):
        m_impl(other.m_impl) { }

Program::Program(Program &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Program::Program(std::shared_ptr<ProgramImpl> &&impl):
        m_impl(std::move(impl)) { }

Program::Program(const Device &device):
        m_impl(ProgramImpl::create(device.impl())) { }

Program::~Program() { }

Program &Program::operator=(const Program &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Program &Program::operator=(Program &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Device Program::device() const {
    return Device(m_impl->device());
}

//
//    Grid
//

Grid::Grid() { }

Grid::Grid(const Grid &other):
        m_impl(other.m_impl) { }

Grid::Grid(Grid &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Grid::Grid(std::shared_ptr<GridImpl> &&impl):
        m_impl(std::move(impl)) { }

Grid::Grid(
        const Program &program, 
        uint32_t x, 
        uint32_t y):
            m_impl(GridImpl::create(program.impl(), x, y)) { }

Grid::Grid(
        const Program &program,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end, 
        uint32_t y_end):
            m_impl(GridImpl::create(
                program.impl(),
                x_start, 
                y_start, 
                x_end, 
                y_end)) { }

Grid::Grid(
        const Program &program,
        const std::vector<Range> &ranges):
            m_impl(GridImpl::create(program.impl(), ranges)) { }

Grid::~Grid() { }

Grid &Grid::operator=(const Grid &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Grid &Grid::operator=(Grid &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Program Grid::program() const {
    return Program(m_impl->program());
}

int Grid::range_count() const {
    return m_impl->range_count();
}

Range Grid::range_at(int index) const {
    return m_impl->range_at(index);
}

//
//    Global
//

Global::Global() { }

Global::Global(const Global &other):
        m_impl(other.m_impl) { }

Global::Global(Global &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Global::Global(std::shared_ptr<GlobalImpl> &&impl):
        m_impl(std::move(impl)) { }

Global::Global(
        const Device &device,
        DataFormat data_format,
        uint32_t size,
        uint32_t log2_page_size):
            m_impl(GlobalImpl::create(
                device.impl(),
                data_format,
                true,
                size,
                log2_page_size)) { }

Global::Global(
        const Device &device,
        DataFormat data_format,
        bool is_dram,
        uint32_t size,
        uint32_t log2_page_size):
            m_impl(GlobalImpl::create(
                device.impl(),
                data_format,
                is_dram,
                size,
                log2_page_size)) { }

Global::Global(
        const Device &device,
        DataFormat data_format,
        GlobalDist dist,
        uint32_t size,
        uint32_t page_size):
            m_impl(GlobalImpl::create(
                device.impl(),
                data_format,
                dist,
                size,
                page_size)) { }

Global::~Global() { }

Global &Global::operator=(const Global &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Global &Global::operator=(Global &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Device Global::device() const {
    return Device(m_impl->device());
}

DataFormat Global::data_format() const {
    return m_impl->data_format();
}

GlobalDist Global::dist() const {
    return m_impl->dist();
}

bool Global::is_dram() const {
    return m_impl->is_dram();
}

uint32_t Global::size() const {
    return m_impl->size();
}

uint32_t Global::page_size() const {
    return m_impl->page_size();
}

uint32_t Global::log2_page_size() const {
    return m_impl->log2_page_size();
}

uint32_t Global::bytes() const {
    return m_impl->bytes();
}

uint32_t Global::page_bytes() const {
    return m_impl->page_bytes();
}

//
//    Local
//

Local::Local() { }

Local::Local(const Local &other):
        m_impl(other.m_impl) { }

Local::Local(Local &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Local::Local(std::shared_ptr<LocalImpl> &&impl):
        m_impl(std::move(impl)) { }

Local::Local(
        const Device &device,
        DataFormat data_format,
        uint32_t size):
            m_impl(LocalImpl::create(
                device.impl(),
                data_format,
                size)) { }

Local::Local(
        const Program &program,
        const Grid &grid,
        DataFormat data_format,
        uint32_t size):
            m_impl(LocalImpl::create(
                program.impl(),
                grid.impl(),
                data_format,
                size)) { }

Local::~Local() { }

Local &Local::operator=(const Local &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Local &Local::operator=(Local &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Device Local::device() const {
    return Device(m_impl->device());
}

Program Local::program() const {
    return Program(m_impl->program());
}

Grid Local::grid() const {
    return Grid(m_impl->grid());
}

DataFormat Local::data_format() const {
    return m_impl->data_format();
}

uint32_t Local::size() const {
    return m_impl->size();
}

LocalScope Local::scope() const {
    return m_impl->scope();
}

//
//    Pipe
//

Pipe::Pipe() { }

Pipe::Pipe(const Pipe &other):
        m_impl(other.m_impl) { }

Pipe::Pipe(Pipe &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Pipe::Pipe(std::shared_ptr<PipeImpl> &&impl):
        m_impl(std::move(impl)) { }

Pipe::Pipe(
        const Program &program,
        const Grid &grid,
        PipeKind kind,
        DataFormat data_format,
        uint32_t size,
        uint32_t frame_size):
            m_impl(PipeImpl::create(
                program.impl(),
                grid.impl(),
                kind,
                data_format,
                size,
                frame_size)) { }
                
Pipe::~Pipe() { }

Pipe &Pipe::operator=(const Pipe &other) {
    if (this != &other) {
        m_impl = other.m_impl;    
    }
    return *this;
}

Pipe &Pipe::operator=(Pipe &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Program Pipe::program() const {
    return Program(m_impl->program());
}

Grid Pipe::grid() const {
    return Grid(m_impl->grid());
}

PipeKind Pipe::kind() const {
    return m_impl->kind();
}

DataFormat Pipe::data_format() const {
    return m_impl->data_format();
}

uint32_t Pipe::size() const {
    return m_impl->size();
}

uint32_t Pipe::frame_size() const {
    return m_impl->frame_size();
}

void Pipe::set_local(const Local &local) const {
    m_impl->set_local(local.impl());
}

//
//    Semaphore
//

Semaphore::Semaphore() { }

Semaphore::Semaphore(const Semaphore &other):
        m_impl(other.m_impl) { }

Semaphore::Semaphore(Semaphore &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Semaphore::Semaphore(std::shared_ptr<SemaphoreImpl> &&impl):
        m_impl(std::move(impl)) { }

Semaphore::Semaphore(
        const Program &program,
        const Grid &grid, 
        uint32_t init_value):
            m_impl(SemaphoreImpl::create(
                program.impl(),
                grid.impl(),
                init_value)) { }

Semaphore::~Semaphore() { }

Semaphore &Semaphore::operator=(const Semaphore &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Semaphore &Semaphore::operator=(Semaphore &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Program Semaphore::program() const {
    return Program(m_impl->program());
}

Grid Semaphore::grid() const {
    return Grid(m_impl->grid());
}

uint32_t Semaphore::init_value() const {
    return m_impl->init_value();
}

//
//    Kernel
//

Kernel::Kernel() { }

Kernel::Kernel(const Kernel &other):
        m_impl(other.m_impl) { }

Kernel::Kernel(Kernel &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Kernel::Kernel(std::shared_ptr<KernelImpl> &&impl):
        m_impl(std::move(impl)) { }

Kernel::Kernel(
        const Program &program,
        const Grid &grid,
        KernelKind kind,
        KernelFormat format,
        const std::string &path,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines):
            m_impl(KernelImpl::create(
                program.impl(),
                grid.impl(),
                kind,
                format,
                path,
                compile_args,
                defines)) { }


Kernel::~Kernel() { }

Kernel &Kernel::operator=(const Kernel &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Kernel &Kernel::operator=(Kernel &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Program Kernel::program() const {
    return Program(m_impl->program());
}

Grid Kernel::grid() const {
    return Grid(m_impl->grid());
}

KernelKind Kernel::kind() const {
    return m_impl->kind();
}

KernelFormat Kernel::format() const {
    return m_impl->format();
}

std::string Kernel::path() const {
    return m_impl->path();
}

const std::vector<uint32_t> &Kernel::compile_args() const {
    return m_impl->compile_args();
}

const std::map<std::string, std::string> &Kernel::defines() const {
    return m_impl->defines();
}

void Kernel::set_args(
        uint32_t x,
        uint32_t y,
        const std::vector<KernelArg> &args) const {
    m_impl->set_args(x, y, args);
}

void Kernel::set_args(
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end,
        const std::vector<KernelArg> &args) const {
    m_impl->set_args(x_start, y_start, x_end, y_end, args);
}

void Kernel::set_args(const Grid &grid, const std::vector<KernelArg> &args) const {
    m_impl->set_args(grid.impl(), args);
}

//
//    Queue
//

Queue::Queue() { }

Queue::Queue(const Queue &other):
        m_impl(other.m_impl) { }

Queue::Queue(Queue &&other) noexcept:
        m_impl(std::move(other.m_impl)) { }

Queue::Queue(std::shared_ptr<QueueImpl> &&impl):
        m_impl(std::move(impl)) { }

Queue::Queue(const Device &device, uint32_t id):
        m_impl(QueueImpl::create(device.impl(), id)) { }

Queue::~Queue() { }

Queue &Queue::operator=(const Queue &other) {
    if (this != &other) {
        m_impl = other.m_impl;
    }
    return *this;
}

Queue &Queue::operator=(Queue &&other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Device Queue::device() const {
    return Device(m_impl->device());
}

uint32_t Queue::id() const {
    return m_impl->id();
}

void Queue::enqueue_read(
        const Global &global, 
        void *dst,
        bool blocking) const {
    m_impl->enqueue_read(global.impl(), dst, blocking);
}

void Queue::enqueue_write(
        const Global &global, 
        const void *src,
        bool blocking) const {
    m_impl->enqueue_write(global.impl(), src, blocking);
}

void Queue::enqueue_program(const Program &program, bool blocking) const {
    m_impl->enqueue_program(program.impl(), blocking);
}

void Queue::finish() const {
    m_impl->finish();
}

} // namespace host
} // namespace tanto
} // namespace ronin

