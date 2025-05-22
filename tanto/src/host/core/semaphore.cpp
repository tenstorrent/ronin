// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <variant>
#include <type_traits>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/util.hpp"
#include "core/metal.hpp"

namespace ronin {
namespace tanto {
namespace host {

namespace {

std::variant<CoreRange, CoreRangeSet> make_core_spec(const std::shared_ptr<GridImpl> &grid) {
    return std::visit(
        [](auto &&arg) -> std::variant<CoreRange, CoreRangeSet> {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                return CoreRange(arg);
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                return arg;
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                return arg;
            }
        }, 
        grid->impl());
}

} // namespace

//
//    SemaphoreImpl
//

SemaphoreImpl::SemaphoreImpl(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        uint32_t init_value):
            m_program(program),
            m_grid(grid),
            m_init_value(init_value),
            m_impl(0) { }

SemaphoreImpl::~SemaphoreImpl() { }

std::shared_ptr<SemaphoreImpl> SemaphoreImpl::create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        uint32_t init_value) {
    validate_program_grid(program, grid);
    auto semaphore = std::make_shared<SemaphoreImpl>(program, grid, init_value);
    program->add_semaphore(semaphore);
    semaphore->create_impl();
    return semaphore;
}

void SemaphoreImpl::create_impl() {
    std::shared_ptr<ProgramImpl> program = m_program.lock();
    std::variant<CoreRange, CoreRangeSet> core_spec = make_core_spec(m_grid);
    m_impl = metal::CreateSemaphore(program->impl(), core_spec, m_init_value);
}

} // namespace host
} // namespace tanto
} // namespace ronin

