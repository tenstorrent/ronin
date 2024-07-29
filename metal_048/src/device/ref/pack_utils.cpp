
#include <cstdint>
#include <string>
#include <stdexcept>

#include "core/kernel_structs.hpp"

#include "ref/pack_utils.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

namespace {

union U32 {
    float f;
    uint32_t i;
};

//
//    Pack
//

void pack_tile_float32(const float *src, uint8_t *dst) {
    float *ptr = reinterpret_cast<float *>(dst);
    for (int i = 0; i < 1024; i++) {
        ptr[i] = src[i];
    }
}

void pack_tile_float16(const float *src, uint8_t *dst) {
    // TODO
}

void pack_tile_float16b(const float *src, uint8_t *dst) {
    U32 u32;
    uint16_t *ptr = reinterpret_cast<uint16_t *>(dst);
    for (int i = 0; i < 1024; i++) {
        u32.f = src[i];
        ptr[i] = uint16_t(u32.i >> 16);
    }
}

void pack_tile_bfp8(const float *src, uint8_t *dst) {
    // TODO (Reserved)
}

void pack_tile_bfp8b(const float *src, uint8_t *dst) {
    // TODO (Reserved)
}

//
//    Unpack
//

void unpack_tile_float32(const uint8_t *src, float *dst) {
    const float *ptr = reinterpret_cast<const float *>(src);
    for (int i = 0; i < 1024; i++) {
        dst[i] = ptr[i];
    }
}

void unpack_tile_float16(const uint8_t *src, float *dst) {
    // TODO
}

void unpack_tile_float16b(const uint8_t *src, float *dst) {
    U32 u32;
    const uint16_t *ptr = reinterpret_cast<const uint16_t *>(src);
    for (int i = 0; i < 1024; i++) {
        u32.i = uint32_t(ptr[i]) << 16;
        dst[i] = u32.f;
    }
}

void unpack_tile_bfp8(const uint8_t *src, float *dst) {
    // TODO (Reserved)
}

void unpack_tile_bfp8b(const uint8_t *src, float *dst) {
    // TODO (Reserved)
}

} // namespace

//
//    Public functions
//

void pack_tile(DataFormat data_format, const float *src, uint8_t *dst) {
    switch (data_format) {
    case DataFormat::Float32:
        pack_tile_float32(src, dst);
        break;
    case DataFormat::Float16:
        pack_tile_float16(src, dst);
        break;
    case DataFormat::Float16_b:
        pack_tile_float16b(src, dst);
        break;
    case DataFormat::Bfp8:
        pack_tile_bfp8(src, dst);
        break;
    case DataFormat::Bfp8_b:
        pack_tile_bfp8b(src, dst);
        break;
    default:
        throw std::runtime_error(
            "Unsupported packing for DataFormat value " + 
                std::to_string(int(data_format)));
    }
}

void unpack_tile(DataFormat data_format, const uint8_t *src, float *dst) {
    switch (data_format) {
    case DataFormat::Float32:
        unpack_tile_float32(src, dst);
        break;
    case DataFormat::Float16:
        unpack_tile_float16(src, dst);
        break;
    case DataFormat::Float16_b:
        unpack_tile_float16b(src, dst);
        break;
    case DataFormat::Bfp8:
        unpack_tile_bfp8(src, dst);
        break;
    case DataFormat::Bfp8_b:
        unpack_tile_bfp8b(src, dst);
        break;
    default:
        throw std::runtime_error(
            "Unsupported unpacking for DataFormat value " + 
                std::to_string(int(data_format)));
    }
}

void tile_to_faces(const float *src, float *dst) {
    for (int fh = 0; fh < 2; fh++) {
        for (int ih = 0; ih < 16; ih++) {
            for (int fw = 0; fw < 2; fw++) {
                for (int iw = 0; iw < 16; iw++) {
                    int isrc = fh * 512 + ih * 32 + fw * 16 + iw;
                    int idst = fh * 512 + fw * 256 + ih * 16 + iw; 
                    dst[idst] = src[isrc];
                }
            }
        }
    }
}

void faces_to_tile(const float *src, float *dst) {
    for (int fh = 0; fh < 2; fh++) {
        for (int ih = 0; ih < 16; ih++) {
            for (int fw = 0; fw < 2; fw++) {
                for (int iw = 0; iw < 16; iw++) {
                    int isrc = fh * 512 + fw * 256 + ih * 16 + iw; 
                    int idst = fh * 512 + ih * 32 + fw * 16 + iw;
                    dst[idst] = src[isrc];
                }
            }
        }
    }
}

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

