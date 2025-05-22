// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>
#include <set>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/util.hpp"
#include "core/metal.hpp"

namespace ronin {
namespace tanto {
namespace host {

//
//    GridImpl
//

GridImpl::GridImpl(const std::shared_ptr<ProgramImpl> &program):
        m_program(program),
        m_next_input_cbid(START_INPUT_CBID),
        m_next_output_cbid(START_OUTPUT_CBID),
        m_next_intermed_cbid(START_INTERMED_CBID) { }

GridImpl::~GridImpl() { }

const std::shared_ptr<GridImpl> GridImpl::create(
        const std::shared_ptr<ProgramImpl> &program,
        uint32_t x,
        uint32_t y) {
    auto grid = std::make_shared<GridImpl>(program);
    grid->m_ranges.emplace_back(Range{x, y, x, y});
    grid->create_impl();
    return grid;
}

const std::shared_ptr<GridImpl> GridImpl::create(
        const std::shared_ptr<ProgramImpl> &program,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end) {
    auto grid = std::make_shared<GridImpl>(program);
    grid->m_ranges.emplace_back(Range{x_start, y_start, x_end, y_end});
    grid->create_impl();
    return grid;
}

const std::shared_ptr<GridImpl> GridImpl::create(
        const std::shared_ptr<ProgramImpl> &program,
        const std::vector<Range> &ranges) {
    auto grid = std::make_shared<GridImpl>(program);
    for (const auto &range: ranges) {
        grid->validate_range(range);
        grid->m_ranges.emplace_back(range);
    }
    program->add_grid(grid);
    grid->create_impl();
    return grid;
}

int GridImpl::range_count() {
    return int(m_ranges.size());
}

Range GridImpl::range_at(int index) {
    return m_ranges[index];
}

uint32_t GridImpl::make_cbid(PipeKind pipe_kind) {
    uint32_t cbid = 0;
    switch (pipe_kind) {
    case PipeKind::INPUT:
        cbid = m_next_input_cbid;
        if (cbid >= END_INPUT_CBID) {
            throw Error("Too many input pipes");
        }
        m_next_input_cbid++;
        break;
    case PipeKind::OUTPUT:
        cbid = m_next_output_cbid;
        if (cbid >= END_OUTPUT_CBID) {
            throw Error("Too many output pipes");
        }
        m_next_output_cbid++;
        break;
    case PipeKind::INTERMED:
        cbid = m_next_intermed_cbid;
        if (cbid >= END_INTERMED_CBID) {
            throw Error("Too many intermediate pipes");
        }
        m_next_intermed_cbid++;
        break;
    default:
        assert(false);
        break;
    }
    return cbid;
}

void GridImpl::validate_range(const Range &range) {
    validate_range_coord(range);
    for (Range &range2: m_ranges) {
        if (range_overlap(range, range2)) {
            throw Error("Overlapping ranges in a grid");
        }
    }
}

void GridImpl::create_impl() {
    if (m_ranges.size() == 1) {
        Range &range = m_ranges[0];
        m_impl = 
            CoreRange(
                CoreCoord(range.x_start, range.y_start), 
                CoreCoord(range.x_end, range.y_end));
    } else {
        std::set<CoreRange> range_set;
        for (Range &range: m_ranges) {
            range_set.emplace(
                CoreRange(
                    CoreCoord(range.x_start, range.y_start), 
                    CoreCoord(range.x_end, range.y_end)));
        }
        m_impl = CoreRangeSet(range_set);
    }
}

} // namespace host
} // namespace tanto
} // namespace ronin

