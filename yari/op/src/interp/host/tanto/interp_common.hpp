// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace op {
namespace interp {
namespace tanto {

enum class CoordTransformMode {
    HALF_PIXEL,
    PYTORCH_HALF_PIXEL,
    ASYMMETRIC,
    TF_HALF_PIXEL_FOR_NN,
    ALIGN_CORNERS
};

float get_input_coord(
    CoordTransformMode mode,
    int output_coord, 
    float scale, 
    int output_dim, 
    int input_dim);

} // namespace tanto
} // namespace interp
} // namespace op
} // namespace ronin

