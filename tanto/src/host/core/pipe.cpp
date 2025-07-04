// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/util.hpp"
#include "core/metal.hpp"

namespace ronin {
namespace tanto {
namespace host {

namespace {

tt::DataFormat map_data_format(DataFormat data_format) {
    switch (data_format) {
    case DataFormat::UINT16:
        return tt::DataFormat::UInt16;
    case DataFormat::UINT32:
        return tt::DataFormat::UInt32;
    case DataFormat::FLOAT32:
        return tt::DataFormat::Float32;
    case DataFormat::BFLOAT16:
        return tt::DataFormat::Float16_b;
    default:
        throw Error("Unsupported data format");
        return tt::DataFormat(0);
    }
}

} // namespace

//
//    PipeImpl
//

PipeImpl::PipeImpl(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        PipeKind kind,
        DataFormat data_format,
        uint32_t size,
        uint32_t frame_size):
            m_program(program),
            m_grid(grid),
            m_kind(kind),
            m_data_format(data_format),
            m_size(size),
            m_frame_size(frame_size),
            m_cbid(0),
            m_local(nullptr),
            m_impl(0) { }

PipeImpl::~PipeImpl() { }

std::shared_ptr<PipeImpl> PipeImpl::create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::shared_ptr<GridImpl> &grid,
        PipeKind kind,
        DataFormat data_format,
        uint32_t size,
        uint32_t frame_size) {
    validate_program_grid(program, grid);
    auto pipe =
        std::make_shared<PipeImpl>(
            program,
            grid,
            kind,
            data_format,
            size,
            frame_size);
    program->add_pipe(pipe);
    pipe->create_impl();
    return pipe;
}

void PipeImpl::set_local(const std::shared_ptr<LocalImpl> &local) {
    // to avoid deep greed comparison, require same grid object
    if (local->grid() != m_grid) {
        throw Error("Grid mismatch of pipe and local buffer");
    }
    m_local = local;
}

void PipeImpl::create_impl() {
    // size parameters are in tiles
    uint32_t tile_bytes = 1024 * get_item_bytes(m_data_format);
    uint32_t bytes = m_size * tile_bytes;
    std::shared_ptr<ProgramImpl> program = m_program.lock();
    m_cbid = m_grid->make_cbid(m_kind);
    metal::CircularBufferConfig config(bytes, {{m_cbid, map_data_format(m_data_format)}});
    config.set_page_size(m_cbid, tile_bytes);
    m_impl = metal::CreateCircularBuffer(program->impl(), m_grid->impl(), config);
}

void PipeImpl::update_dynamic_address() {
    if (m_local != nullptr) {
        std::shared_ptr<ProgramImpl> program = m_program.lock();
        metal::UpdateDynamicCircularBufferAddress(program->impl(), m_impl, *m_local->impl());
    }
}

} // namespace host
} // namespace tanto
} // namespace ronin

