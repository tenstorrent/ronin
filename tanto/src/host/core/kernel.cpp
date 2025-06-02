// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <vector>
#include <map>
#include <memory>
#include <variant>
#include <type_traits>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/util.hpp"
#include "core/metal.hpp"

#if 0 // TODO: Revise this
#ifdef METAL_057

#include "tt-metalium/program.hpp"

#else

#include "tt_metal/detail/program.hpp"

#endif
#endif

namespace ronin {
namespace tanto {
namespace host {

//
//    KernelImpl
//

KernelImpl::KernelImpl(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        KernelKind kind,
        KernelFormat format,
        const std::string &path,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines):
            m_program(program),
            m_grid(grid),
            m_kind(kind),
            m_format(format),
            m_path(path),
            m_compile_args(compile_args),
            m_defines(defines),
            m_impl(0) { }

KernelImpl::~KernelImpl() { }

std::shared_ptr<KernelImpl> KernelImpl::create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        KernelKind kind,
        KernelFormat format,
        const std::string &path,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines) {
    validate_program_grid(program, grid);
    auto kernel =
        std::make_shared<KernelImpl>(
            program,
            grid,
            kind,
            format,
            path,
            compile_args,
            defines);
    program->add_kernel(kernel);
    kernel->create_impl();
    return kernel;
}

void KernelImpl::set_args(
        uint32_t x,
        uint32_t y,
        const std::vector<KernelArg> &args) {
    Range range{x, y, x, y};
    set_args(range, args);
}

void KernelImpl::set_args(
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end,
        const std::vector<KernelArg> &args) {
    Range range{x_start, y_start, x_end, y_end};
    set_args(range, args);
}

void KernelImpl::set_args(
        const std::shared_ptr<GridImpl> &grid, 
        const std::vector<KernelArg> &args) {
    int range_count = grid->range_count();
    for (int i = 0; i < range_count; i++) {
        Range range = grid->range_at(i);
        set_args(range, args);
    }
}

#if 0 // TODO: Revise this
void KernelImpl::set_args(const Range &range, const std::vector<KernelArg> &args) {
    validate_range(range);
    m_ranges_args.emplace_back(RangeArgs{range, args});
}

void KernelImpl::set_args_impl() {
    std::shared_ptr<ProgramImpl> program = m_program.lock();
    std::shared_ptr<DeviceImpl> device = program->device();
// Since 0.52 ?
//    std::shared_ptr<metal::Kernel> kernel = program->impl().get_kernel(m_impl);
    std::shared_ptr<metal::Kernel> kernel = metal::detail::GetKernel(program->impl(), m_impl);
    for (RangeArgs &range_args: m_ranges_args) {
        Range &range = range_args.range;
        std::shared_ptr<metal::RuntimeArgs> args_impl = make_args_impl(range_args.args);
        metal::SetRuntimeArgs(
            device->impl(), 
            kernel,
            CoreRange(
                CoreCoord(range.x_start, range.y_start),
                CoreCoord(range.x_end, range.y_end)),
            args_impl);
    }
}
#endif

void KernelImpl::set_args(const Range &range, const std::vector<KernelArg> &args) {
    validate_range(range);
    std::shared_ptr<metal::RuntimeArgs> args_impl = make_args_impl(args);
    m_ranges_args.emplace_back(RangeArgs{range, args, args_impl});
}

void KernelImpl::set_args_impl() {
    std::shared_ptr<ProgramImpl> program = m_program.lock();
    std::shared_ptr<DeviceImpl> device = program->device();
// Since 0.52 ?
//    std::shared_ptr<metal::Kernel> kernel = program->impl().get_kernel(m_impl);
    std::shared_ptr<metal::Kernel> kernel = metal::detail::GetKernel(program->impl(), m_impl);
    for (RangeArgs &range_args: m_ranges_args) {
        Range &range = range_args.range;
        update_args_impl(range_args.args, *range_args.args_impl);
        metal::SetRuntimeArgs(
            device->impl(), 
            kernel,
            CoreRange(
                CoreCoord(range.x_start, range.y_start),
                CoreCoord(range.x_end, range.y_end)),
            range_args.args_impl);
    }
}

void KernelImpl::validate_range(const Range &range) {
    validate_range_coord(range);
    for (RangeArgs &range_args: m_ranges_args) {
        if (range_overlap(range, range_args.range)) {
            throw Error("Overlapping ranges of argument lists");
        }
    }
}

void KernelImpl::create_impl() {
    // TODO: Compilation of Tanto kernels must happen here
    std::shared_ptr<ProgramImpl> program = m_program.lock();
    switch (m_kind) {
    case KernelKind::READER:
        {
            // or use ReaderDataMovementConfig
            metal::DataMovementConfig config{
                .processor = DataMovementProcessor::RISCV_1, 
                .noc = NOC::RISCV_1_default,
                .compile_args = m_compile_args,
                .defines = m_defines
            };
            m_impl = metal::CreateKernel(program->impl(), m_path, m_grid->impl(), config);
        }
        break;
    case KernelKind::WRITER:
        {
            // or use WriterDataMovementConfig
            metal::DataMovementConfig config{
                .processor = DataMovementProcessor::RISCV_0, 
                .noc = NOC::RISCV_0_default,
                .compile_args = m_compile_args,
                .defines = m_defines
            };
            m_impl = metal::CreateKernel(program->impl(), m_path, m_grid->impl(), config);
        }
        break;
    case KernelKind::MATH:
        {
            // fixed in this implememtation
            MathFidelity math_fidelity = MathFidelity::HiFi4;
            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            metal::ComputeConfig config{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = m_compile_args,
                .defines = m_defines
            };
            m_impl = metal::CreateKernel(program->impl(), m_path, m_grid->impl(), config);
        }
        break;
    default:
        assert(false);
        break;
    }
}

/*
Tanto mapping of standard objects

struct Global {
    uint32_t addr;
    uint32_t log2_page_size;
};
struct Local {
    uint32_t addr;
};
struct Pipe {
    uint32_t cb_id;
    uint32_t frame_size;
};
struct Semaphore {
    uint32_t addr;
};
*/

#if 0 // TODO: Revse this
std::shared_ptr<metal::RuntimeArgs> 
        KernelImpl::make_args_impl(const std::vector<KernelArg> &args) {
    std::shared_ptr<metal::RuntimeArgs> args_impl = std::make_shared<metal::RuntimeArgs>();
    for (const KernelArg &arg: args) {
        std::visit(
            [&args_impl](auto &&v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, uint32_t>) {
                    args_impl->emplace_back(v);
                } else if constexpr (std::is_same_v<T, Global>) {
                    metal::Buffer *buffer = v.impl()->impl().get();
                    args_impl->emplace_back(buffer);
                    if (v.dist() == GlobalDist::LINEAR) {
                        args_impl->emplace_back(u32_log2(buffer->page_size()));
                    } else {
                        args_impl->emplace_back(v.page_bytes());
                    }
                } else if constexpr (std::is_same_v<T, Local>) {
                    args_impl->emplace_back(v.impl()->impl().get());
                } else if constexpr (std::is_same_v<T, Pipe>) {
                    args_impl->emplace_back(v.impl()->cbid());
                    args_impl->emplace_back(v.frame_size());
                } else if constexpr (std::is_same_v<T, Semaphore>) {
                    args_impl->emplace_back(v.impl()->impl());
                }
            }, arg);
    }
    return args_impl;
}
#endif

std::shared_ptr<metal::RuntimeArgs> 
        KernelImpl::make_args_impl(const std::vector<KernelArg> &args) {
    std::shared_ptr<metal::RuntimeArgs> args_impl = std::make_shared<metal::RuntimeArgs>();
    for (const KernelArg &arg: args) {
        std::visit(
            [&args_impl](auto &&v) {
                using T = std::decay_t<decltype(v)>;
                // SKIPPED: Global, Local
                //     (Global can be kept in passthrough queue mode)
                if constexpr (std::is_same_v<T, uint32_t>) {
                    args_impl->emplace_back(v);
                } else if constexpr (std::is_same_v<T, Global>) {
                    // reserve space for buffer pointer and page size
                    args_impl->emplace_back((metal::Buffer *)nullptr);
                    args_impl->emplace_back(uint32_t(0));
                } else if constexpr (std::is_same_v<T, Local>) {
                    // reserve space for buffer pointer
                    args_impl->emplace_back((metal::Buffer *)nullptr);
                } else if constexpr (std::is_same_v<T, Pipe>) {
                    args_impl->emplace_back(v.impl()->cbid());
                    args_impl->emplace_back(v.frame_size());
                } else if constexpr (std::is_same_v<T, Semaphore>) {
                    args_impl->emplace_back(v.impl()->impl());
                }
            }, arg);
    }
    return args_impl;
}

void KernelImpl::update_args_impl(
        const std::vector<KernelArg> &args,
        metal::RuntimeArgs &args_impl) {
    int pos = 0;
    for (const KernelArg &arg: args) {
        std::visit(
            [&args_impl, &pos](auto &&v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, uint32_t>) {
                    pos++;
                } else if constexpr (std::is_same_v<T, Global>) {
                    metal::Buffer *buffer = v.impl()->impl().get();
                    args_impl[pos] = buffer;
                    if (v.dist() == GlobalDist::LINEAR) {
                        args_impl[pos+1] = u32_log2(buffer->page_size());
                    } else {
                        args_impl[pos+1] = v.page_bytes();
                    }
                    pos += 2;
                } else if constexpr (std::is_same_v<T, Local>) {
                    args_impl[pos] = v.impl()->impl().get();
                    pos++;
                } else if constexpr (std::is_same_v<T, Pipe>) {
                    pos += 2;
                } else if constexpr (std::is_same_v<T, Semaphore>) {
                    pos++;
                }
            }, arg);
    }
}

} // namespace host
} // namespace tanto
} // namespace ronin

