// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include "core/kernel_structs.hpp"
#include "core/llk_defs.hpp"
#include "core/memory.hpp"
#include "core/cb_api.hpp"

#include "ref/pack_utils.hpp"
#include "ref/llk.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

namespace {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float elu(float x, float slope) {
    return (x <= 0.0) ? slope * (std::exp(x) - 1) : x;
}

inline float gelu(float x) {
    float SQRT_2_DIV_PI = 0.7978845608028654f;  // sqrt(2.0 / PI);
    return 0.5f * x * (1.0f + std::tanh(SQRT_2_DIV_PI * (x + 0.044715f * x * x * x)));
}

inline float max_threshold_relu(float x, float threshold) {
    return (x > threshold) ? threshold : (x < 0.0f) ? 0.0f : x;
}

inline float min_threshold_relu(float x, float threshold) {
    return (x < threshold) ? 0.0f : x;
}

inline float relu(float x) {
    return (x <= 0.0f) ? 0.0f : x;
}

inline float lrelu(float x, float slope) {
    return (x <= 0.0f) ? slope * x : x;
}

inline float pow_u32(float x, uint32_t pow) {
    float y = 1.0f;
    while (pow > 0) {
        if ((pow & 1) != 0) {
            y *= x;
        }
        y *= y;
        pow >>= 1;
    }
    return y;
}

union U32 {
    float f;
    uint32_t i;
};

float u16a_as_float(int x) {
    // u16a: [15] sign [14:10] exponent (bias 15) [9:0] mantissa
    uint32_t m = uint32_t(x & 0x03ff);
    uint32_t e = uint32_t(x & 0x7c00) >> 10;
    uint32_t s = uint32_t(x & 0x8000) >> 15;
    // float: [31] sign [30:23] exponent (bias 127) [22:0] mantissa
    m = m << 13;
    e = (e + (127 - 15)) << 23;
    s = s << 31;
    U32 v;
    v.i = s | e | m;
    return v.f;
}

float u16b_as_float(int x) {
    U32 v;
    v.i = uint32_t(x) << 16;
    return v.f;
}

float u32_as_float(int x) {
    U32 v;
    v.i = uint32_t(x);
    return v.f;
}

uint32_t float_as_u32(float x) {
    U32 v;
    v.f = x;
    return v.i;
}

void diag_tile_stats(const char *tag, const float *tile) {
    int imin = -1;
    int imax = -1;
    float vmin = 0.0f;
    float vmax = 0.0f;
    double vsum = 0.0f;
    for (int i = 0; i < 1024; i++) {
        float v = tile[i];
        if (imin < 0 || v < vmin) {
            imin = i;
            vmin = v;
        }
        if (imax < 0 || v > vmax) {
            imax = i;
            vmax = v;
        }
        vsum += double(v);
    }
    printf("%s: MIN %d (%g) MAX %d (%g) SUM %g\n", tag, imin, vmin, imax, vmax, vsum);
}

} // namespace

//
//    LLK
//

LLK::LLK(Memory *l1, CB *cb) { 
    m_l1 = l1;
    m_cb = cb;
    m_src_a.resize(TILE_SIZE);
    m_src_b.resize(TILE_SIZE);
    m_dst.resize(DST_COUNT * TILE_SIZE); 
    m_tile.resize(TILE_SIZE);
    m_temp.resize(TILE_SIZE);
    m_relu_mode = ReluType::NO_RELU;
    m_relu_threshold = 0.0f;
    m_dst_valid.resize(DST_COUNT, false);
    m_block.resize(32 * TILE_SIZE);
}

LLK::~LLK() { }

void LLK::reset() {
    m_relu_mode = ReluType::NO_RELU;
    m_relu_threshold = 0.0f;
}

void LLK::acquire_dst() {
    memset(m_dst.data(), 0, m_dst.size() * sizeof(float));
    for (int i = 0; i < DST_COUNT; i++) {
        m_dst_valid[i] = false;
    }
}

// pack

void LLK::pack(uint32_t tile_index, uint32_t output) {
    float *dst = get_dst_ptr(tile_index);
    float *from = dst;
    switch (m_relu_mode) {
    case ReluType::NO_RELU:
        // nothing to do
        break;
    case ReluType::MAX_THRESHOLD_RELU:
        from = m_temp.data();
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            from[i] = max_threshold_relu(dst[i], m_relu_threshold);
        }
        break;
    case ReluType::MIN_THRESHOLD_RELU:
        from = m_temp.data();
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            from[i] = min_threshold_relu(dst[i], m_relu_threshold);
        }
        break;
    default:
        assert(false);
        break;
    }
    // currently, always in-order pack
    uint8_t *to = get_cb_write_ptr(output);
    uint32_t offset = m_cb->get_write_tile_ptr(output) * m_cb->get_tile_size(output);
    to += offset;
    m_cb->incr_write_tile_ptr(output, 1);
    float *tile = m_tile.data();
    tile_to_faces(from, tile);
    pack_tile(get_cb_data_format(output), tile, to);
}

void LLK::matmul_pack(uint32_t tile_index, uint32_t output, uint32_t ntiles) {
    for (uint32_t i = 0; i < ntiles; i++) {
        pack(tile_index + i, output);
    }
}

void LLK::pack_relu_config(uint32_t config) {
    uint32_t mode_enc = config & 0xf;
    ReluType mode = 
        (mode_enc == 0) ? ReluType::NO_RELU : 
        (mode_enc == 3) ? ReluType::MAX_THRESHOLD_RELU : 
            ReluType::MIN_THRESHOLD_RELU;
    uint32_t threshold_enc = config >> 16; 
    // TODO: Verify threshold encoding format
    float threshold = u16a_as_float(threshold_enc);
    m_relu_mode = mode;
    m_relu_threshold = threshold;
}

void LLK::pack_block(uint32_t block_index, uint32_t output) {
    // special method: emulator only
    float *from = m_block.data() + block_index * TILE_SIZE;
    float *dst = get_dst_ptr(0);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = from[i];
    }
    pack(0, output);
}

void LLK::pack_block_raw(uint32_t block_index, uint32_t output) {
    // special method: emulator only
    float *from = m_block.data() + block_index * TILE_SIZE;
    uint8_t *to = get_cb_write_ptr(output);
    uint32_t offset = m_cb->get_write_tile_ptr(output) * m_cb->get_tile_size(output);
    to += offset;
    m_cb->incr_write_tile_ptr(output, 1);
    pack_tile(get_cb_data_format(output), from, to);
}

void LLK::pack_untilize(
        uint32_t block_rt_dim, 
        uint32_t output, 
        uint32_t block_c_index,
        uint32_t block_ct_dim, 
        uint32_t full_ct_dim) {
    float *dst = get_dst_ptr(0);
    DataFormat df = get_cb_data_format(output);
    uint8_t *base_ptr = get_cb_write_ptr(output);
    uint32_t offset = block_c_index * 32;
    uint32_t stride_rt = full_ct_dim * 1024;
    uint32_t stride_ct = 32;
    uint32_t stride_r = full_ct_dim * 32;
    for (uint32_t block_rt = 0; block_rt < block_rt_dim; block_rt++) {
        for (uint32_t block_ct = 0; block_ct < block_ct_dim; block_ct++) {
            for (uint32_t r = 0; r < 32; r++) {
                uint32_t index = 
                    offset +
                    block_rt * stride_rt + 
                    block_ct * stride_ct +
                    r * stride_r;
                uint8_t *ptr = base_ptr + get_raw_offset(df, index);
                pack_raw(df, dst, ptr, 32);
                dst += 32;
            }
        }
    }
}

// unpack

void LLK::unpack_A(uint32_t operand, uint32_t tile_index, bool transpose_xy) {
    float *src_a = m_src_a.data();
    float *to = transpose_xy ? m_temp.data() : src_a;
    uint8_t *from = get_cb_read_ptr(operand) + get_cb_tile_offset(operand, tile_index);
    float *tile = m_tile.data();
    unpack_tile(get_cb_data_format(operand), from, tile);
    faces_to_tile(tile, to);
    if (transpose_xy) {
        for (uint32_t h = 0; h < TILE_DIM; h++) {
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                src_a[h * TILE_DIM + w] = to[w * TILE_DIM + h];
            }
        }
    }
}

void LLK::unpack_AB(
        BroadcastType BType,
        uint32_t operandA, 
        uint32_t operandB, 
        uint32_t tile_index_a, 
        uint32_t tile_index_b) {
    // unpack A
    float *src_a = m_src_a.data();
    uint8_t *from_a = get_cb_read_ptr(operandA) + get_cb_tile_offset(operandA, tile_index_a);
    float *tile = m_tile.data();
    unpack_tile(get_cb_data_format(operandA), from_a, tile);
    faces_to_tile(tile, src_a);

    // upack B
    float *src_b = m_src_b.data();
    float *to_b = (BType == BroadcastType::NONE) ? src_b : m_temp.data();
    uint8_t *from_b = get_cb_read_ptr(operandB) + get_cb_tile_offset(operandB, tile_index_b);
    unpack_tile(get_cb_data_format(operandB), from_b, tile);
    faces_to_tile(tile, to_b);
    switch (BType) {
    case BroadcastType::NONE:
        // nothing to do
        break;
    case BroadcastType::COL:
        for (uint32_t h = 0; h < TILE_DIM; h++) {
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                src_b[h * TILE_DIM + w] = to_b[h * TILE_DIM];
            }
        }
        break;
    case BroadcastType::ROW:
        for (uint32_t h = 0; h < TILE_DIM; h++) {
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                src_b[h * TILE_DIM + w] = to_b[w];
            }
        }
        break;
    case BroadcastType::SCALAR:
        for (uint32_t h = 0; h < TILE_DIM; h++) {
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                src_b[h * TILE_DIM + w] = to_b[0];
            }
        }
        break;
    default:
        assert(false);
        break;
    }
}

void LLK::unpack_AB_matmul(
        uint32_t operandA, 
        uint32_t operandB, 
        uint32_t tile_index_a, 
        uint32_t tile_index_b) {
    // same as unpack_AB with no broadcast

    // unpack A
    float *src_a = m_src_a.data();
    uint8_t *from_a = get_cb_read_ptr(operandA) + get_cb_tile_offset(operandA, tile_index_a);
    float *tile = m_tile.data();
    unpack_tile(get_cb_data_format(operandA), from_a, tile);
    faces_to_tile(tile, src_a);

    // upack B
    float *src_b = m_src_b.data();
    uint8_t *from_b = get_cb_read_ptr(operandB) + get_cb_tile_offset(operandB, tile_index_b);
    unpack_tile(get_cb_data_format(operandB), from_b, tile);
    faces_to_tile(tile, src_b);
}

//
//    Dao of tilize/untilize position calculations (without splitting into faces)
//
//    Given untiled array representing band of 'block' 32x32 blocks
//    Consider one data item in this array
//    Let b = block index, i = position inside block
//    Let t = tile index, r = row inside tile, c = column inside tile
//    Item offset in untiled array
//        p = b * 1024 + i
//        p = r * (block * 32) + t * 32 + c
//    Item offset in corresponding tiled array
//        q = t * 1024 + r * 32 + c
//

void LLK::unpack_tilize(uint32_t icb, uint32_t block) {
    reserve_block(block);
    float *dst = m_block.data();
    float *tile = m_tile.data();
    DataFormat df = get_cb_data_format(icb);
    uint32_t w = block * 32;
    // fill DST with 'block' tiles without further splitting into faces
    // subsequent 'pack' will split tiles into faces
    for (uint32_t idst = 0; idst < block; idst++) {
        uint8_t *from = get_cb_read_ptr(icb) + get_cb_tile_offset(icb, idst);
        // unpack 32x32 block of untiled data
        unpack_tile(df, from, tile);
        uint32_t b = idst * 1024;
        for (uint32_t i = 0; i < 1024; i++) {
            // iterate through 32x32 data items
            // 'p' is item offset in untiled array
            // compute t = tile index, r = row inside tile, c = column inside tile
            uint32_t p = b + i;
            uint32_t c = p % 32;
            p /= 32;
            uint32_t t = p % block;
            uint32_t r = p / block;
            // 'q' is item offset in tiled array
            uint32_t q = t * 1024 + r * 32 + c;
            dst[q] = tile[i];
        }
    }
}

void LLK::unpack_untilize(uint32_t icb, uint32_t block) {
    reserve_block(block);
    float *src_a = m_src_a.data();
    float *dst = m_block.data();
    float *tile = m_tile.data();
    DataFormat df = get_cb_data_format(icb);
    for (uint32_t idst = 0; idst < block; idst++) {
        uint8_t *from = get_cb_read_ptr(icb) + get_cb_tile_offset(icb, idst);
        // unpack 32x32 tile, undo splitting into faces
        unpack_tile(df, from, tile);
        faces_to_tile(tile, src_a);
        for (uint32_t i = 0; i < 1024; i++) {
            // iterate through 32x32 tile items
            // 'idst' is tile index
            // compute r = row inside tile, c = column inside tile
            uint32_t r = i / 32;
            uint32_t c = i % 32;
            uint32_t q = r * (block * 32) + idst * 32 + c;
            // 'q' is item offset in untiled array
            dst[q] = src_a[i];
        }
    }
}

// math

void LLK::math_eltwise_binary(
        EltwiseBinaryType eltwise_binary_type,
        BroadcastType src_b_bcast_type,
        uint32_t dst_index) {
    float *src_a = m_src_a.data();
    float *src_b = m_src_b.data();
    float *dst = get_dst_ptr(dst_index);
    // 'src_b_bcast_type' is not used here as 'unpack_AB'
    // is supposed to perform broadcasting already while loading srcB
    switch (eltwise_binary_type) {
    case EltwiseBinaryType::ELWMUL:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst[i] = src_a[i] * src_b[i];
        }
        break;
    case EltwiseBinaryType::ELWADD:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst[i] = src_a[i] + src_b[i];
        }
        break;
    case EltwiseBinaryType::ELWSUB:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst[i] = src_a[i] - src_b[i];
        }
        break;
    default:
        assert(false);
        break;
    }
    m_dst_valid[dst_index] = true;
}

void LLK::math_eltwise_unary_datacopy(uint32_t dst_index) {
    float *src_a = m_src_a.data();
    float *dst = get_dst_ptr(dst_index);
    copy_tile(dst, src_a);
    m_dst_valid[dst_index] = true;
}

void LLK::math_matmul(uint32_t dst_index, bool transpose) {
    float *src_a = m_src_a.data();
    float *src_b = m_src_b.data();
    float *dst = get_dst_ptr(dst_index);
    if (!transpose) {
        for (uint32_t h = 0; h < TILE_DIM; h++) {
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                float acc = 0.0f;
                for (uint32_t d = 0; d < TILE_DIM; d++) {
                    acc += src_a[h * TILE_DIM + d] * src_b[d * TILE_DIM + w];
                }
                dst[h * TILE_DIM + w] += acc;
            }
        }
    } else {
        for (uint32_t h = 0; h < TILE_DIM; h++) {
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                float acc = 0.0f;
                for (uint32_t d = 0; d < TILE_DIM; d++) {
                    acc += src_a[h * TILE_DIM + d] * src_b[w * TILE_DIM + d];
                }
                dst[h * TILE_DIM + w] += acc;
            }
        }
    }
    m_dst_valid[dst_index] = true;
}

void LLK::math_reduce(PoolType type, ReduceDim dim, uint32_t dst_index) {
    float *src_a = m_src_a.data();
    float *src_b = m_src_b.data();
    float *dst = get_dst_ptr(dst_index);
    float scaler = src_b[0];
    switch (type) {
    case PoolType::SUM:
        switch (dim) {
        case ReduceDim::REDUCE_ROW:
            for (uint32_t h = 0; h < TILE_DIM; h++) {
                float acc = 0.0f;
                for (uint32_t w = 0; w < TILE_DIM; w++) {
                    acc += src_a[h * TILE_DIM + w];
                }
                dst[h * TILE_DIM] += scaler * acc;
            }
            break;
        case ReduceDim::REDUCE_COL:
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                float acc = 0.0f;
                for (uint32_t h = 0; h < TILE_DIM; h++) {                
                    acc += src_a[h * TILE_DIM + w];
                }
                dst[w] += scaler * acc;
            }
            break;
        case ReduceDim::REDUCE_SCALAR:
            {
                float acc = 0.0f;
                for (uint32_t i = 0; i < TILE_SIZE; i++) {
                    acc += src_a[i];
                }
                dst[0] += scaler * acc;
            }
            break;
        default:
            assert(false);
            break;
        }
        break;
    case PoolType::MAX:
        if (!m_dst_valid[dst_index]) {
            for (uint32_t i = 0; i < TILE_SIZE; i++) {
                dst[i] = -std::numeric_limits<float>::max();
            }
        }
        switch (dim) {
        case ReduceDim::REDUCE_ROW:
            for (uint32_t h = 0; h < TILE_DIM; h++) {
                float acc = -std::numeric_limits<float>::max();
                for (uint32_t w = 0; w < TILE_DIM; w++) {
                    acc = std::max(acc, src_a[h * TILE_DIM + w]);
                }
                dst[h * TILE_DIM] = std::max(dst[h * TILE_DIM], scaler * acc);
            }
            break;
        case ReduceDim::REDUCE_COL:
            for (uint32_t w = 0; w < TILE_DIM; w++) {
                float acc = -std::numeric_limits<float>::max();
                for (uint32_t h = 0; h < TILE_DIM; h++) {                
                    acc = std::max(acc, src_a[h * TILE_DIM + w]);
                }
                dst[w] = std::max(dst[w], scaler * acc);
            }
            break;
        case ReduceDim::REDUCE_SCALAR:
            {
                float acc = -std::numeric_limits<float>::max();
                for (uint32_t i = 0; i < TILE_SIZE; i++) {
                    acc = std::max(acc, src_a[i]);
                }
                dst[0] = std::max(dst[0], scaler * acc);
            }
            break;
        default:
            assert(false);
            break;
        }
        break;
    default:
        assert(false);
        break;
    }
    m_dst_valid[dst_index] = true;
}

// math/sfpu

void LLK::math_eltwise_binary_sfpu_copy_dest_values(uint32_t dst_index0, uint32_t dst_index1) {
    float *dst0 = get_dst_ptr(dst_index0);
    float *dst1 = get_dst_ptr(dst_index1);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst0[i] = dst1[i];
    }
}

void LLK::math_eltwise_binary_sfpu_binop(
        SfpuBinaryOp binop, 
        uint32_t dst_index0, 
        uint32_t dst_index1) {
    float *dst0 = get_dst_ptr(dst_index0);
    float *dst1 = get_dst_ptr(dst_index1);
    switch (binop) {
    case SfpuBinaryOp::ADD:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst0[i] += dst1[i];
        }
        break;
    case SfpuBinaryOp::SUB:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst0[i] -= dst1[i];
        }
        break;
    case SfpuBinaryOp::MUL:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst0[i] *= dst1[i];
        }
        break;
    case SfpuBinaryOp::DIV:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst0[i] /= dst1[i];
        }
        break;
    case SfpuBinaryOp::RSUB:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst0[i] = dst1[i] - dst0[i];
        }
        break;
    case SfpuBinaryOp::POW:
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst0[i] = std::pow(dst0[i], dst1[i]);
        }
        break;
    default:
        assert(false);
        break;
    }
}

void LLK::math_eltwise_unary_sfpu_rsqrt(uint32_t dst_index, bool approx) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = 1.0f / std::sqrt(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_sigmoid(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = sigmoid(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_log(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::log(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_log_with_base(uint32_t dst_index, uint32_t base_scale) {
    // TODO
}

void LLK::math_eltwise_unary_sfpu_tanh(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::tanh(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_signbit(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::signbit(dst[i]) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_abs(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::abs(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_sign(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        float x = dst[i];
        dst[i] = (x < 0.0f) ? -1.0f : (x == 0.0f) ? 0.0f : 1.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_square(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        float x = dst[i];
        dst[i] = x * x;
    }
}

void LLK::math_eltwise_unary_sfpu_ltz(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = (dst[i] < 0.0f) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_eqz(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = (dst[i] == 0.0f) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_lez(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = (dst[i] <= 0.0f) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_gtz(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = (dst[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_nez(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = (dst[i] != 0.0f) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_gez(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = (dst[i] >= 0.0f) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_max(uint32_t dst0_index, uint32_t dst1_index) {
    float *dst0 = get_dst_ptr(dst0_index);
    float *dst1 = get_dst_ptr(dst1_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst0[i] = std::max(dst0[i], dst1[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_power(uint32_t dst_index, uint32_t pow) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = pow_u32(dst[i], pow);
    }
}

void LLK::math_eltwise_unary_sfpu_exp2(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::exp2(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_heaviside(uint32_t dst_index, uint32_t param0) {
    float step = u32_as_float(param0);
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        float x = dst[i];
        dst[i] = (x < 0.0f) ? 0.0f : (x == 0.0f) ? step : 1.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_expm1(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::expm1(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_asin(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::asin(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_atan(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::atan(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_acos(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::acos(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_add_scalar(uint32_t dst_index, uint32_t param0) {
    float *dst = get_dst_ptr(dst_index);
    float scalar = u32_as_float(param0);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] += scalar;
    }
}

void LLK::math_eltwise_unary_sfpu_sub_scalar(uint32_t dst_index, uint32_t param0) {
    float *dst = get_dst_ptr(dst_index);
    float scalar = u32_as_float(param0);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] -= scalar;
    }
}

void LLK::math_eltwise_unary_sfpu_mul_scalar(uint32_t dst_index, uint32_t param0) {
    float *dst = get_dst_ptr(dst_index);
    float scalar = u32_as_float(param0);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] *= scalar;
    }
}

void LLK::math_eltwise_unary_sfpu_div_scalar(uint32_t dst_index, uint32_t param0) {
    float *dst = get_dst_ptr(dst_index);
    float scalar = u32_as_float(param0);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] /= scalar;
    }
}

void LLK::math_eltwise_unary_sfpu_rsub_scalar(uint32_t dst_index, uint32_t param0) {
    float *dst = get_dst_ptr(dst_index);
    float scalar = u32_as_float(param0);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = scalar - dst[i];
    }
}

void LLK::math_eltwise_unary_sfpu_ceil(uint32_t dst_index) {
    constexpr float I16_MIN = -32767.0f;
    constexpr float I16_MAX = 32767.0f;
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        float x = dst[i];
        // emulate Metal LLK behavior
        if (x >= I16_MIN && x <= I16_MAX) {
            x = std::ceil(x);
        }
        dst[i] = x;
    }
}

void LLK::math_eltwise_unary_sfpu_ceil_float32(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::ceil(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_elu(uint32_t dst_index, uint32_t param0) {
#if 0 // TODO: Revise this
    float slope = u16b_as_float(param0);
#endif
    float slope = u32_as_float(param0);
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = elu(dst[i], slope);
    }
}

void LLK::math_eltwise_unary_sfpu_erf(uint32_t dst_index, bool approx) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::erf(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_erfc(uint32_t dst_index, bool approx) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::erfc(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_erfinv(uint32_t dst_index) {
    // TODO
}

void LLK::math_eltwise_unary_sfpu_exponential(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::exp(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_fill_bitcast(uint32_t dst_index, uint32_t param0) {
    float fill = u32_as_float(param0);
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = fill;
    }
}

void LLK::math_eltwise_unary_sfpu_floor(uint32_t dst_index) {
    constexpr float I16_MIN = -32767.0f;
    constexpr float I16_MAX = 32767.0f;
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        float x = dst[i];
        // emulate Metal LLK behavior
        if (x >= I16_MIN && x <= I16_MAX) {
            x = std::floor(x);
        }
        dst[i] = x;
    }
}

void LLK::math_eltwise_unary_sfpu_floor_float32(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::floor(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_gelu(uint32_t dst_index, bool approx) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = gelu(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_i0(uint32_t dst_index) {
    // TODO
}

void LLK::math_eltwise_unary_sfpu_isinf(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::isinf(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_isposinf(uint32_t dst_index) {
    // TODO
}

void LLK::math_eltwise_unary_sfpu_isneginf(uint32_t dst_index) {
    // TODO
}

void LLK::math_eltwise_unary_sfpu_isnan(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::isnan(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_isfinite(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::isfinite(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_logical_not_unary(uint32_t dst_index) {
    // how is this different from 'eqz'
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = (dst[i] == 0.0f) ? 1.0f : 0.0f;
    }
}

void LLK::math_eltwise_unary_sfpu_reciprocal(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = 1.0f / dst[i];
    }
}

void LLK::math_eltwise_unary_sfpu_relu_max(uint32_t dst_index, uint32_t param0) {
#if 0 // TODO: Revise this
    float threshold = u16b_as_float(param0);
#endif
    float threshold = u32_as_float(param0);
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = max_threshold_relu(dst[i], threshold);
    }
}

void LLK::math_eltwise_unary_sfpu_relu_min(uint32_t dst_index, uint32_t param0) {
#if 0 // TODO: Revise this
    float threshold = u16b_as_float(param0);
#endif
    float threshold = u32_as_float(param0);
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = min_threshold_relu(dst[i], threshold);
    }
}

void LLK::math_eltwise_unary_sfpu_relu(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = relu(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_lrelu(uint32_t dst_index, uint32_t param0) {
#if 0 // TODO: Revise this
    float slope = u16b_as_float(param0);
#endif
    float slope = u32_as_float(param0);
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = lrelu(dst[i], slope);
    }
}

void LLK::math_eltwise_unary_sfpu_sqrt(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::sqrt(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_sine(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::sin(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_cosine(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::cos(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_tan(uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    for (uint32_t i = 0; i < TILE_SIZE; i++) {
        dst[i] = std::tan(dst[i]);
    }
}

void LLK::math_eltwise_unary_sfpu_typecast(
        uint32_t in_dtype, 
        uint32_t out_dtype, 
        uint32_t dst_index) {
    float *dst = get_dst_ptr(dst_index);
    if (in_dtype == int(DataFormat::Float16_b) && out_dtype == int(DataFormat::UInt16)) {
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
#if 0 // ACHTUNG: Temporary workaround - need regular support of uint16 pack/unpack
            dst[i] = u32_as_float(uint16_t(dst[i]));
#else
            dst[i] = u16b_as_float(uint16_t(dst[i]));
#endif
        }
    } else if (in_dtype == int(DataFormat::UInt16) && out_dtype == int(DataFormat::Float16_b)) {
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst[i] = float(float_as_u32(dst[i]));
        }
    } else {
        // not yet implemented
        assert(false);
    }
}

// implementation

void LLK::reserve_block(uint32_t tiles) {
    uint32_t size = tiles * TILE_SIZE;
    if (m_block.size() < size) {
        m_block.resize(size);
    }
}

DataFormat LLK::get_cb_data_format(uint32_t operand) {
    return m_cb->get_unpack_src_format(operand);
}

uint32_t LLK::get_cb_tile_offset(uint32_t operand, uint32_t tile_index) {
    return m_cb->get_tile_size(operand) * tile_index;
}

uint8_t *LLK::get_cb_read_ptr(uint32_t operand) {
    uint32_t addr = m_cb->get_read_ptr(operand);
    return m_l1->map_addr(addr);
}

uint8_t *LLK::get_cb_write_ptr(uint32_t operand) {
    uint32_t addr = m_cb->get_write_ptr(operand);
    return m_l1->map_addr(addr);
}

float *LLK::get_dst_ptr(uint32_t dst_index) {
    assert(dst_index >= 0 && dst_index < DST_COUNT);
    return m_dst.data() + dst_index * TILE_SIZE;
}

void LLK::copy_tile(float *dst, const float *src) {
    memcpy(dst, src, TILE_SIZE * sizeof(float));
}

void LLK::fill_tile(float *dst, float value) {
    if (value == 0.0f) {
        memset(dst, 0, TILE_SIZE * sizeof(float));
    } else {
        for (uint32_t i = 0; i < TILE_SIZE; i++) {
            dst[i] = value;
        }
    }
}

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

