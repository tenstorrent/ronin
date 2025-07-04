// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/conv2d_basic_spatial.hpp"

//
//    Notes on LX (activation cache) size computation
//
//    Basic p -> h mapping formula
//
//    h = p * stride_h - pad_h + r * dilation_h
//
//    Assume p in range [p_start, p_end)
//    Then h is in range [h_start, h_end) where
//
//    h_start = p_start * stride_h - pad_h
//    h_end = (p_end - 1) * stride_h - pad_h + (R - 1) * dilation_h + 1
//
//    Then LX height is
//
//    h_end - h_start = (p_end - p_start - 1) * stride_h + (R - 1) * dilation_h + 1
//
//    For use in reader kernels, [h_start, h_end) are clipped by [0, H)
//

namespace ronin {
namespace op {
namespace conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;
namespace util = ronin::op::common::util;

constexpr bool ENABLE_CACHE_LX = true;
constexpr bool ENABLE_CACHE_LW = true;
constexpr bool ENABLE_MCAST = true;
constexpr bool ENABLE_PWISE = true;

constexpr uint32_t SPLIT_COUNT = 4;

constexpr bool ENABLE_PARAM_KERNELS = false;

//
//    Conv2dBasicSpatial
//

Conv2dBasicSpatial::Conv2dBasicSpatial(
        int N,
        int H,
        int W,
        int C,
        int P,
        int Q,
        int K,
        int R,
        int S,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        const base::PostOpSpec &post_op,
        int batch_size):
            m_N(uint32_t(N)),
            m_H(uint32_t(H)),
            m_W(uint32_t(W)),
            m_C(uint32_t(C)),
            m_P(uint32_t(P)),
            m_Q(uint32_t(Q)),
            m_K(uint32_t(K)),
            m_R(uint32_t(R)),
            m_S(uint32_t(S)),
            m_pad_h(uint32_t(pad_h)),
            m_pad_w(uint32_t(pad_w)),
            m_stride_h(uint32_t(stride_h)),
            m_stride_w(uint32_t(stride_w)),
            m_dilation_h(uint32_t(dilation_h)),
            m_dilation_w(uint32_t(dilation_w)),
            m_post_op(post_op),
            m_batch_size(uint32_t(batch_size)) { 
    // these values are required to compute input/output volumes and shapes
    m_C_arg = m_C;
    m_K_arg = m_K;
    m_C = u32_align(m_C, 32);
    m_K = u32_align(m_K, 32);            
}

Conv2dBasicSpatial::~Conv2dBasicSpatial() { }

void Conv2dBasicSpatial::init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gz,
        const core::Global &gy) {
    m_device = device;
    m_gx = gx;
    m_gw = gw;
    m_gb = gb;
    m_gz = gz;
    m_gy = gy;

    // ACHTUNG: Supported batch sizes are Wormhole-specific
    assert(m_batch_size < 8 || m_batch_size % 8 == 0);
    assert(m_N % m_batch_size == 0);

    assert(m_C % 32 == 0);
    assert(m_K % 32 == 0);

    m_split_count = SPLIT_COUNT;
    m_split_stride = (m_P + m_split_count - 1) / m_split_count;

    // ACHTUNG: Temporary limit 64 is Wormhole-specific
    assert(m_batch_size * m_split_count <= 64);

    init_config();

    m_program = core::Program(m_device);

    compute_grid_dims(m_grid_x, m_grid_y);

    m_grid = core::Grid(m_program, 0, 0, m_grid_x - 1, m_grid_y - 1);

    m_zero_size = m_C;
    m_mask_size = (m_split_stride * m_Q + 31) / 32;
    m_mask_size *= m_R * m_S * m_split_count;

    uint32_t Ct = m_C / 32;
    uint32_t Kt = m_K / 32;
    if (Kt <= DEST_TILES) {
        m_Ki = Kt;
    } else if (Kt % DEST_TILES == 0) {
        m_Ki = DEST_TILES;
    } else {
        // TODO: Implement better rule?
        m_Ki = 1;
    }
    m_Ko = Kt / m_Ki;
    assert(m_Ki * m_Ko == Kt);

    m_px_frame_size = Ct;
    m_pw_frame_size = Ct;
    if (!m_gb.is_null()) {
        m_pb_frame_size = Kt;
    }
    if (!m_gz.is_null()) {
        m_pz_frame_size = Kt;
    }
    m_py_frame_size = Kt;
    m_px_im_frame_size = Ct;
    m_pz_im_frame_size = Kt;
    m_py_im_frame_size = Kt;

    m_start_p = uint32_t(0) - m_pad_h * m_W * m_C;
    m_start_q = uint32_t(0) - m_pad_w * m_C;
    m_delta_p = m_stride_h * m_W * m_C;
    m_delta_q = m_stride_w * m_C;
    m_delta_r = m_dilation_h * m_W * m_C;
    m_delta_s = m_dilation_w * m_C;
    m_end_q = m_start_q + m_Q * m_delta_q;

    m_enable_param_kernels = ENABLE_PARAM_KERNELS;
    m_metal_kernel_base_path = "op/conv/device/metal";
    m_param_kernel_base_path = "op/conv/device/param";
    m_defines = {{"T", "bfloat16"}};

    init_options();
    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_semaphores();
    create_kernels();

    init_locals();
}

void Conv2dBasicSpatial::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int Conv2dBasicSpatial::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * u32_align(m_H * m_W, 32) * m_C;
    case 1:
        return m_R * m_S * m_K * m_C;
    case 2:
        return m_K * 32;
    case 3:
        return m_N * u32_align(m_P * m_Q, 32) * m_K;
    default:
        assert(false);
        return 0;
    }
}

int Conv2dBasicSpatial::output_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_P * m_Q, 32) * m_K;
}

std::vector<float> Conv2dBasicSpatial::transform_input(int index, const std::vector<float> &x) {
    std::vector<float> y;
    switch (index) {
    case 0:
        y = util::pad(x, m_N, m_H * m_W, m_C_arg, m_N, u32_align(m_H * m_W, 32), m_C);
        break;
    case 1:
        y = util::pad(x, m_R * m_S, m_K_arg, m_C_arg, m_R * m_S, m_K, m_C);
        y = util::tilize(y, m_R * m_S * m_K, m_C);
        y = util::make_faces(y);
        break;
    case 2:
        y = util::pad(x, 1, m_K_arg, 32, m_K);
        y = util::tilize(y, 32, m_K);
        y = util::make_faces(y);
        break;
    case 3:
        y = util::pad(x, m_N, m_P * m_Q, m_K_arg, m_N, u32_align(m_P * m_Q, 32), m_K);
        break;
    default:
        assert(false);
        break;
    }
    return y;
}

std::vector<float> Conv2dBasicSpatial::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, u32_align(m_P * m_Q, 32), m_K, m_N, m_P * m_Q, m_K_arg);
}

void Conv2dBasicSpatial::init_config() {
    m_pwise =
        (ENABLE_PWISE &&
            m_R == 1 && m_S == 1 &&
            m_pad_h == 0 && m_pad_w == 0 &&
            m_stride_h == 1 && m_stride_w == 1 &&
            m_dilation_h == 1 && m_dilation_w == 1);
    int volume_x = get_lx_volume();
    int volume_w = input_volume(1);
    // ACHTUNG: Temporary L1 limit for input caching is Wormhole-specific
    int volume_limit = (128 + 256) * 1024;
    m_cache_lx = false;
    m_cache_lw = false;
    bool may_cache_lx = (ENABLE_CACHE_LX && !m_pwise);
    if (may_cache_lx && ENABLE_CACHE_LW && volume_x + volume_w <= volume_limit) {
        m_cache_lx = true;
        m_cache_lw = true;
    } else if (ENABLE_CACHE_LW && volume_w <= volume_limit) {
        m_cache_lw = true;
    } else if (may_cache_lx && volume_x <= volume_limit) {
        m_cache_lx = true;
    }
    m_mcast = ENABLE_MCAST;
#if 0
printf("@@@ CACHE_LX: %d CACHE_LW: %d MCAST %d PWISE %d\n", 
int(m_cache_lx), int(m_cache_lw), int(m_mcast), int(m_pwise));
#endif
}

void Conv2dBasicSpatial::init_options() {
    m_options = 0;
    if (!m_gb.is_null()) {
        m_options |= OPT_BIAS;
    }
    if (!m_gz.is_null()) {
        m_options |= OPT_ADD;
    }
    base::PostOp op = m_post_op.op();
    if (op != base::PostOp::NONE) {
        m_options |= OPT_UNARY;
    }
}

void Conv2dBasicSpatial::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gw.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gw.bytes() == input_volume(1) * item_bytes);
    if (!m_gb.is_null()) {
        assert(m_gb.bytes() == input_volume(2) * item_bytes);
    }
    if (!m_gz.is_null()) {
        assert(m_gz.bytes() >= input_volume(3) * item_bytes);
    }
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void Conv2dBasicSpatial::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 items
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
    m_gmask = core::Global(m_device, core::DataFormat::UINT32, m_mask_size, log2_page_size);
}

void Conv2dBasicSpatial::create_locals() {
    if (m_cache_lx) {
        uint32_t x_size = get_lx_volume();
        m_lx = core::Local(m_program, m_grid, T, x_size);
    }
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
    m_lmask = core::Local(m_program, m_grid, core::DataFormat::UINT32, m_mask_size);
}

void Conv2dBasicSpatial::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    uint32_t pw_size = 
        m_cache_lw ? 
            m_K * m_R * m_S * m_C / 1024 : 
            m_pw_frame_size * 2;
    m_pw =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            pw_size,
            m_pw_frame_size);
    if (!m_gb.is_null()) {
        m_pb =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INPUT,
                T,
                m_pb_frame_size * 2,
                m_pb_frame_size);
    }
    if (!m_gz.is_null()) {
        m_pz =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INPUT,
                T,
                m_pz_frame_size * 2,
                m_pz_frame_size);
    }
    m_py =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_py_frame_size * 2,
            m_py_frame_size);
    m_px_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_px_im_frame_size * 2,
            m_px_im_frame_size);
    if (!m_gz.is_null()) {
        m_pz_im =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INTERMED,
                T,
                m_pz_im_frame_size * 2,
                m_pz_im_frame_size);
    }
    m_py_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_py_im_frame_size * 2,
            m_py_im_frame_size);
}

void Conv2dBasicSpatial::create_semaphores() {
    if (m_mcast) {
        m_sem_send = core::Semaphore(m_program, m_grid, 0);
        m_sem_recv = core::Semaphore(m_program, m_grid, 0);
    }
}

void Conv2dBasicSpatial::create_kernels() {
    uint32_t reader_options = m_options & OPT_BIAS;
    uint32_t writer_options = m_options & OPT_ADD;
    uint32_t math_options = m_options;
    if (reader_options == OPT_BIAS) {
        if (m_pwise) {
            create_pw_bias_reader();
        } else if (m_cache_lx) {
            create_lx_bias_reader();
        } else {
            create_bias_reader();
        }
    } else {
        // not yet implemented
        assert(false);
    }
    if (m_mcast) {
        if (writer_options == OPT_ADD) {
            if (m_cache_lw) {
                create_lw_mcast_add_writer();
            } else {
                create_mcast_add_writer();
            }
        } else {
            if (m_cache_lw) {
                create_lw_mcast_writer();
            } else {
                create_mcast_writer();
            }
        }
    } else {
        // not yet implemented
        assert(false);
    }
    if (math_options == OPT_BIAS) {
        if (m_cache_lw) {
            create_lw_bias_math();
        } else {
            create_bias_math();
        }
    } else if (math_options == (OPT_BIAS | OPT_ADD)) {
        if (m_cache_lw) {
            create_lw_bias_add_math();
        } else {
            create_bias_add_math();
        }
    } else if (math_options == (OPT_BIAS | OPT_UNARY)) {
        if (m_cache_lw) {
            create_lw_bias_unary_math();
        } else {
            create_bias_unary_math();
        }
    } else if (math_options == (OPT_BIAS | OPT_ADD | OPT_UNARY)) {
        if (m_cache_lw) {
            create_lw_bias_add_unary_math();
        } else {
            create_bias_add_unary_math();
        }
    } else {
        // not yet implemented
        assert(false);
    }
}

void Conv2dBasicSpatial::create_bias_reader() {
    if (m_enable_param_kernels) {
        create_param_bias_reader();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_spatial_bias_reader.cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 R,
        uint32 S,
        uint32 PQ,
        uint32 start_p,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 delta_s,
        uint32 end_q,
        uint32 zero_size,
        uint32 mask_size,
        uint32 x_pos,
        uint32 x_stride,
        uint32 mask_pos)
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_gmask,
        m_lzero,
        m_lmask,
        m_px,
        m_pb,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_R,
        m_S,
        m_split_stride * m_Q,
        uint32_t(0), // [14] start_p
        m_start_q,
        m_delta_p,
        m_delta_q,
        m_delta_r,
        m_delta_s,
        m_end_q,
        m_zero_size,
        m_mask_size,
        uint32_t(0), // [23] x_pos
        x_stride,
        uint32_t(0)  // [25] mask_pos
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_split_count;
            uint32_t qy = y % m_split_count;
            uint32_t batch_index = py * m_grid_x + x;
            uint32_t batch_offset = qy * m_split_stride;
            uint32_t start_p = m_start_p + batch_offset * m_delta_p;
            uint32_t x_pos = batch_index * HW_rnd * m_C;
            uint32_t mask_pos = qy * ((m_split_stride * m_Q + 31) / 32) * m_R * m_S;
            args[14] = start_p;
            args[23] = x_pos;
            args[25] = mask_pos;
            m_reader.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSpatial::create_lx_bias_reader() {
    if (m_enable_param_kernels) {
        create_param_lx_bias_reader();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_spatial_lx_bias_reader.cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lx,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 R,
        uint32 S,
        uint32 PQ,
        uint32 start_p,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 delta_s,
        uint32 end_q,
        uint32 zero_size,
        uint32 mask_size,
        uint32 x_pos,
        uint32 x_stride,
        uint32 x_base,
        uint32 x_size,
        uint32 mask_pos)
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_gmask,
        m_lx,
        m_lzero,
        m_lmask,
        m_px,
        m_pb,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_R,
        m_S,
        m_split_stride * m_Q,
        uint32_t(0), // [15] start_p
        m_start_q,
        m_delta_p,
        m_delta_q,
        m_delta_r,
        m_delta_s,
        m_end_q,
        m_zero_size,
        m_mask_size,
        uint32_t(0), // [24] x_pos
        x_stride,
        uint32_t(0), // [26] x_base
        uint32_t(0), // [27] x_size
        uint32_t(0)  // [28] mask_pos
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_split_count;
            uint32_t qy = y % m_split_count;
            uint32_t batch_index = py * m_grid_x + x;
            uint32_t batch_offset = qy * m_split_stride;
            uint32_t start_p = m_start_p + batch_offset * m_delta_p;
            uint32_t x_pos = batch_index * HW_rnd * m_C;
            // h = p * stride_h - pad_h + r * dilation_h
            uint32_t p_start = qy * m_split_stride;
            uint32_t p_end = p_start + m_split_stride;
            uint32_t h_start = p_start * m_stride_h - m_pad_h;
            h_start = std::max(h_start, uint32_t(0));
            uint32_t h_end = (p_end - 1) * m_stride_h - m_pad_h + (m_R - 1) * m_dilation_h + 1;
            h_end = std::min(h_end, m_H);
            uint32_t x_base = h_start * m_W * m_C;
            uint32_t x_size = (h_end - h_start) * m_W * m_C;
            uint32_t mask_pos = qy * ((m_split_stride * m_Q + 31) / 32) * m_R * m_S;
            args[15] = start_p;
            args[24] = x_pos;
            args[26] = x_base;
            args[27] = x_size;
            args[28] = mask_pos;
            m_reader.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSpatial::create_pw_bias_reader() {
    if (m_enable_param_kernels) {
        create_param_pw_bias_reader();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_spatial_pw_bias_reader.cpp";
    m_reader = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::READER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 HW,
        uint32 C,
        uint32 K,
        uint32 HW_upper,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride)
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gzero,
        m_lzero,
        m_px,
        m_pb,
        m_N / m_batch_size,
        m_split_stride * m_W,
        m_C,
        m_K,
        uint32_t(0), // [10] HW_upper
        m_zero_size,
        uint32_t(0), // [12] x_pos
        x_stride
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_split_count;
            uint32_t qy = y % m_split_count;
            uint32_t HW_upper = std::min(m_split_stride, m_H - m_split_stride * qy) * m_W;
            uint32_t batch_index = py * m_grid_x + x;
            uint32_t x_pos = (batch_index * HW_rnd + qy * m_split_stride * m_W) * m_C;
            args[10] = HW_upper;
            args[12] = x_pos;
            m_reader.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSpatial::create_mcast_writer() {
    if (m_enable_param_kernels) {
        create_param_mcast_writer();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_spatial_mcast_writer.cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gw,
        global<T> gy,
        pipe<T> pw,
        pipe<T> py,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 PQ,
        uint32 PQK_tail,
        uint32 RS,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t num_dests = m_grid_y - 1;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t split_PQ = m_split_stride * m_Q;
    uint32_t PQK_tail = (split_PQ % 32) * m_K;
    uint32_t y_stride = (m_batch_size * PQ_rnd - u32_align(split_PQ, 32)) * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_sem_send,
        m_sem_recv,
        uint32_t(0), // [6] send_mode
        uint32_t(0), // [7] x0
        uint32_t(0), // [8] y0
        uint32_t(0), // [9] x1
        uint32_t(0), // [10] y1
        num_dests,
        m_N / m_batch_size,
        m_C,
        m_K,
        split_PQ,
        PQK_tail,
        m_R * m_S,
        uint32_t(0), // [18] y_pos
        y_stride
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_split_count;
            uint32_t qy = y % m_split_count;
            uint32_t send_mode = (y == 0) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(x, 0, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(x, m_grid_y - 1, x1_phy, y1_phy);
            uint32_t batch_index = py * m_grid_x + x;
            uint32_t batch_offset = qy * m_split_stride;
            uint32_t y_pos = (batch_index * PQ_rnd + batch_offset * m_Q) * m_K;
            args[6] = send_mode;
            args[7] = x0_phy;
            args[8] = y0_phy;
            args[9] = x1_phy;
            args[10] = y1_phy;
            args[18] = y_pos; 
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSpatial::create_lw_mcast_writer() {
    if (m_enable_param_kernels) {
        create_param_lw_mcast_writer();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_spatial_lw_mcast_writer.cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gw,
        global<T> gy,
        pipe<T> pw,
        pipe<T> py,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 K,
        uint32 PQ,
        uint32 PQK_tail,
        uint32 RSKC,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t num_dests = m_grid_y - 1;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t split_PQ = m_split_stride * m_Q;
    uint32_t PQK_tail = (split_PQ % 32) * m_K;
    uint32_t y_stride = (m_batch_size * PQ_rnd - u32_align(split_PQ, 32)) * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_sem_send,
        m_sem_recv,
        uint32_t(0), // [6] send_mode
        uint32_t(0), // [7] x0
        uint32_t(0), // [8] y0
        uint32_t(0), // [9] x1
        uint32_t(0), // [10] y1
        num_dests,
        m_N / m_batch_size,
        m_K,
        split_PQ,
        PQK_tail,
        m_R * m_S * m_K * m_C,
        uint32_t(0), // [17] y_pos
        y_stride
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_split_count;
            uint32_t qy = y % m_split_count;
            uint32_t send_mode = (y == 0) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(x, 0, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(x, m_grid_y - 1, x1_phy, y1_phy);
            uint32_t batch_index = py * m_grid_x + x;
            uint32_t batch_offset = qy * m_split_stride;
            uint32_t y_pos = (batch_index * PQ_rnd + batch_offset * m_Q) * m_K;
            args[6] = send_mode;
            args[7] = x0_phy;
            args[8] = y0_phy;
            args[9] = x1_phy;
            args[10] = y1_phy;
            args[17] = y_pos; 
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSpatial::create_mcast_add_writer() {
    if (m_enable_param_kernels) {
        create_param_mcast_add_writer();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_spatial_mcast_add_writer.cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gw,
        global<T> gz,
        global<T> gy,
        pipe<T> pw,
        pipe<T> pz,
        pipe<T> py,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 PQ,
        uint32 PQK_tail,
        uint32 RS,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t num_dests = m_grid_y - 1;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t split_PQ = m_split_stride * m_Q;
    uint32_t PQK_tail = (split_PQ % 32) * m_K;
    uint32_t y_stride = (m_batch_size * PQ_rnd - u32_align(split_PQ, 32)) * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gz,
        m_gy,
        m_pw,
        m_pz,
        m_py,
        m_sem_send,
        m_sem_recv,
        uint32_t(0), // [8] send_mode
        uint32_t(0), // [9] x0
        uint32_t(0), // [10] y0
        uint32_t(0), // [11] x1
        uint32_t(0), // [12] y1
        num_dests,
        m_N / m_batch_size,
        m_C,
        m_K,
        split_PQ,
        PQK_tail,
        m_R * m_S,
        uint32_t(0), // [20] y_pos
        y_stride
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_split_count;
            uint32_t qy = y % m_split_count;
            uint32_t send_mode = (y == 0) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(x, 0, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(x, m_grid_y - 1, x1_phy, y1_phy);
            uint32_t batch_index = py * m_grid_x + x;
            uint32_t batch_offset = qy * m_split_stride;
            uint32_t y_pos = (batch_index * PQ_rnd + batch_offset * m_Q) * m_K;
            args[8] = send_mode;
            args[9] = x0_phy;
            args[10] = y0_phy;
            args[11] = x1_phy;
            args[12] = y1_phy;
            args[20] = y_pos; 
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSpatial::create_lw_mcast_add_writer() {
    if (m_enable_param_kernels) {
        create_param_lw_mcast_add_writer();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_spatial_lw_mcast_add_writer.cpp";
    m_writer = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::WRITER, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        global<T> gw,
        global<T> gz,
        global<T> gy,
        pipe<T> pw,
        pipe<T> pz,
        pipe<T> py,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 K,
        uint32 PQ,
        uint32 PQK_tail,
        uint32 RSKC,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t num_dests = m_grid_y - 1;
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t split_PQ = m_split_stride * m_Q;
    uint32_t PQK_tail = (split_PQ % 32) * m_K;
    uint32_t y_stride = (m_batch_size * PQ_rnd - u32_align(split_PQ, 32)) * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gz,
        m_gy,
        m_pw,
        m_pz,
        m_py,
        m_sem_send,
        m_sem_recv,
        uint32_t(0), // [8] send_mode
        uint32_t(0), // [9] x0
        uint32_t(0), // [10] y0
        uint32_t(0), // [11] x1
        uint32_t(0), // [12] y1
        num_dests,
        m_N / m_batch_size,
        m_K,
        split_PQ,
        PQK_tail,
        m_R * m_S * m_K * m_C,
        uint32_t(0), // [19] y_pos
        y_stride
    };
    for (uint32_t x = 0; x < m_grid_x; x++) {
        for (uint32_t y = 0; y < m_grid_y; y++) {
            uint32_t py = y / m_split_count;
            uint32_t qy = y % m_split_count;
            uint32_t send_mode = (y == 0) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(x, 0, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(x, m_grid_y - 1, x1_phy, y1_phy);
            uint32_t batch_index = py * m_grid_x + x;
            uint32_t batch_offset = qy * m_split_stride;
            uint32_t y_pos = (batch_index * PQ_rnd + batch_offset * m_Q) * m_K;
            args[8] = send_mode;
            args[9] = x0_phy;
            args[10] = y0_phy;
            args[11] = x1_phy;
            args[12] = y1_phy;
            args[19] = y_pos; 
            m_writer.set_args(x, y, args);
        }
    }
}

void Conv2dBasicSpatial::create_bias_math() {
    if (m_enable_param_kernels) {
        create_param_bias_math();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_batch_bias_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::create_bias_add_math() {
    if (m_enable_param_kernels) {
        create_param_bias_add_math();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_batch_bias_add_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::create_bias_unary_math() {
    if (m_enable_param_kernels) {
        create_param_bias_unary_math();
        return;
    }
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_metal_kernel_base_path + "/basic_batch_bias_" + suffix + "_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::create_bias_add_unary_math() {
    if (m_enable_param_kernels) {
        create_param_bias_add_unary_math();
        return;
    }
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_metal_kernel_base_path + "/basic_batch_bias_add_" + suffix + "_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::create_lw_bias_math() {
    if (m_enable_param_kernels) {
        create_param_lw_bias_math();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_batch_lw_bias_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSKC)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::create_lw_bias_add_math() {
    if (m_enable_param_kernels) {
        create_param_lw_bias_add_math();
        return;
    }
    std::string path = m_metal_kernel_base_path + "/basic_batch_lw_bias_add_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSKC)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::create_lw_bias_unary_math() {
    if (m_enable_param_kernels) {
        create_param_lw_bias_unary_math();
        return;
    }
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_metal_kernel_base_path + "/basic_batch_lw_bias_" + suffix + "_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSKC,
        unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::create_lw_bias_add_unary_math() {
    if (m_enable_param_kernels) {
        create_param_lw_bias_add_unary_math();
        return;
    }
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_metal_kernel_base_path + "/basic_batch_lw_bias_add_" + suffix + "_math.cpp";
    m_math = 
        core::Kernel(
            m_program, 
            m_grid, 
            core::KernelKind::MATH, 
            core::KernelFormat::METAL,
            path, 
            {}, 
            m_defines);
/*
void kernel(
        pipe<T> px,
        pipe<T> pw,
        pipe<T> pb,
        pipe<T> pz,
        pipe<T> py,
        pipe<T> px_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSKC,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pz,
        m_py,
        m_px_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Ko,
        m_Ki,
        m_split_stride * m_Q,
        m_R * m_S,
        m_R * m_S * m_K * m_C,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void Conv2dBasicSpatial::init_locals() {
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    std::vector<uint32_t> vmask(m_gmask.bytes() / sizeof(uint32_t), 0);
    compute_mask(vmask);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gzero, vzero.data(), true);
    queue.enqueue_write(m_gmask, vmask.data(), true);
}

void Conv2dBasicSpatial::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = m_split_count;
    } else {
        x = 8;
        y = (m_batch_size / 8) * m_split_count;
    }
}

void Conv2dBasicSpatial::compute_mask(std::vector<uint32_t> &vmask) {
    int PQ = m_split_stride * m_Q;
    int size = (PQ + 31) / 32;
    size *= m_R * m_S;
    assert(size * m_split_count <= vmask.size());
    int k = 0;
    for (int split = 0; split < int(m_split_count); split++) {
        for (int pq_start = 0; pq_start < PQ; pq_start += 32) {
            for (int r = 0; r < int(m_R); r++) {
                for (int s = 0; s < int(m_S); s++) {
                    uint32_t mask = 0;
                    uint32_t flag = 1;
                    for (int i = 0; i < 32; i++) {
                        int pq = pq_start + i;
                        if (pq >= PQ) {
                            break;
                        }
                        int p = pq / m_Q + split * m_split_stride;
                        int q = pq % m_Q;
                        if (p >= int(m_P)) {
                            break;
                        }
                        int h = p * m_stride_h - m_pad_h + r * m_dilation_h;
                        int w = q * m_stride_w - m_pad_w + s * m_dilation_w;
                        if (h >= 0 && h < int(m_H) && w >= 0 && w < int(m_W)) {
                            mask |= flag;
                        }
                        flag <<= 1;
                    }
                    vmask[k] = mask;
                    k++;
                } // s
            } // r
        } // pq_start
    } // split
}

uint32_t Conv2dBasicSpatial::get_lx_volume() {
    // h = p * stride_h - pad_h + r * dilation_h
    return ((m_split_stride - 1) * m_stride_h + (m_R - 1) * m_dilation_h + 1) * m_W * m_C;
}

std::string Conv2dBasicSpatial::get_unary_kernel_suffix() {
    base::PostOp op = m_post_op.op();
    switch (op) {
    case base::PostOp::RELU:
        return "relu";
    case base::PostOp::CLIP:
        if (is_unary_relu6()) {
            return "relu6";
        }
        // generic clip is not yet implemented
        assert(false);
        return "<?>";
    default:
        assert(false);
        return "<?>";
    }
}

uint32_t Conv2dBasicSpatial::get_unary_op_code() {
    // these op codes must be used by all kernels
    static constexpr uint32_t
        UNARY_OP_RELU = 0,
        UNARY_OP_RELU6 = 1;
    base::PostOp op = m_post_op.op();
    switch (op) {
    case base::PostOp::RELU:
        return UNARY_OP_RELU;
    case base::PostOp::CLIP:
        if (is_unary_relu6()) {
            return UNARY_OP_RELU6;
        }
        // generic clip is not yet implemented
        assert(false);
        return 0;
    default:
        assert(false);
        return 0;
    }
}

uint32_t Conv2dBasicSpatial::encode_unary_param0() {
    base::PostOp op = m_post_op.op();
    switch (op) {
    case base::PostOp::RELU:
        return 0;
    case base::PostOp::CLIP:
        if (is_unary_relu6()) {
            // param0 hardcoded in kernels
            return 0;
        }
        // generic clip is not yet implemented
        assert(false);
        return 0;
    default:
        assert(false);
        return 0;
    }
}

bool Conv2dBasicSpatial::is_unary_relu6() {
    return (m_post_op.alpha() == 0.0f && m_post_op.beta() == 6.0f);
}

} // namespace tanto
} // namespace conv
} // namespace op
} // namespace ronin

