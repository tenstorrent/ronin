#pragma once

#include <cstdint>

#include "core/kernel_structs.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

void pack_tile(DataFormat data_format, const float *src, uint8_t *dst);
void unpack_tile(DataFormat data_format, const uint8_t *src, float *dst);

void tile_to_faces(const float *src, float *dst);
void faces_to_tile(const float *src, float *dst);

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

