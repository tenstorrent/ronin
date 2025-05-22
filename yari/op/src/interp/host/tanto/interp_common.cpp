// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "host/tanto/interp_common.hpp"

namespace ronin {
namespace op {
namespace interp {
namespace tanto {

float get_input_coord(
        CoordTransformMode mode,
        int output_coord, 
        float scale, 
        int output_dim, 
        int input_dim) {
    switch (mode) {
    case CoordTransformMode::HALF_PIXEL:
        return (output_coord + 0.5f) / scale - 0.5f;
    case CoordTransformMode::PYTORCH_HALF_PIXEL:
        return (output_dim > 1) ? 
            (output_coord + 0.5f) / scale - 0.5f : 
            0.0f;
    case CoordTransformMode::ASYMMETRIC:
        return output_coord / scale;
    case CoordTransformMode::TF_HALF_PIXEL_FOR_NN:
        return (output_coord + 0.5f) / scale;
    case CoordTransformMode::ALIGN_CORNERS:
        return (output_coord > 1) ? 
            output_coord * float(input_dim - 1) / float(output_dim - 1) : 
            1.0f;
    default:
        assert(false);
        return 0.0f;
    }
}

} // namespace tanto
} // namespace interp
} // namespace op
} // namespace ronin

