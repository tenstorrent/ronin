// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/util/transform.hpp"

#include "host/tanto/ds_conv2d_batch.hpp"

namespace ronin {
namespace op {
namespace group_conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;
namespace util = ronin::op::common::util;

constexpr bool ENABLE_CACHE_LX = true;

namespace {

uint32_t u32_align(uint32_t a, uint32_t b) {
    return ((a + b - 1) / b) * b;
}

uint32_t get_item_bytes(core::DataFormat data_format) {
    switch (data_format) {
    case core::DataFormat::UINT32:
        return 4;
    case core::DataFormat::FLOAT32:
        return 4;
    case core::DataFormat::BFLOAT16:
        return 2;
    default:
        assert(false);
        return 0;
    }
}

std::vector<float> expand_weights_bias(const std::vector<float> &x, int split) {
    size_t size = x.size();
    assert(size % split == 0);
    size_t part = size / split;
    std::vector<float> y(size * 32);
    const float *px = x.data();
    float *py = y.data();
    for (int i = 0; i < split; i++) {
        for (int k = 0; k < 32; k++) {
            memcpy(py, px, part * sizeof(float));
            py += part;
        }
        px += part;
    }
    return y;
}

} // namespace

//
//    DSConv2dBatch
//

DSConv2dBatch::DSConv2dBatch(
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

DSConv2dBatch::~DSConv2dBatch() { }

void DSConv2dBatch::init(
        const core::Device &device,
        const core::Global &gx,
        const core::Global &gw,
        const core::Global &gb,
        const core::Global &gw2,
        const core::Global &gb2,
        const core::Global &gz,
        const core::Global &gy) {
    m_device = device;
    m_gx = gx;
    m_gw = gw;
    m_gb = gb;
    m_gw2 = gw2;
    m_gb2 = gb2;
    m_gz = gz;
    m_gy = gy;

    assert(m_batch_size < 8 || m_batch_size % 8 == 0);
    // ACHTUNG: Temporary limit 64 is Wormhole-specific
    assert(m_batch_size <= 64);
    assert(m_N % m_batch_size == 0);

    assert(m_C % 32 == 0);
    assert(m_K % 32 == 0);

    init_config();

    m_program = core::Program(m_device);

    uint32_t grid_x, grid_y;
    compute_grid_dims(grid_x, grid_y);

    m_x_start = 0;
    m_y_start = 0;
    m_x_end = grid_x - 1;
    m_y_end = grid_y - 1;

    m_grid = core::Grid(m_program, m_x_start, m_y_start, m_x_end, m_y_end);

    m_zero_size = m_C;
    m_mask_size = (m_P * m_Q + 31) / 32;
    m_mask_size *= m_R * m_S;

    uint32_t Ct = m_C / 32;
    if (Ct <= DEST_TILES) {
        m_Ci = Ct;
    } else if (Ct % DEST_TILES == 0) {
        m_Ci = DEST_TILES;
    } else {
        // TODO: Implement better rule?
        m_Ci = 1;
    }
    m_Co = Ct / m_Ci;
    assert(m_Ci * m_Co == Ct);

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
    m_pb_frame_size = Ct;
    m_pw2_frame_size = Ct;
    m_pb2_frame_size = Kt;
    m_pz_frame_size = Kt;
    m_py_frame_size = Kt;
    m_pu_im_frame_size = Ct;
    m_pt_im_frame_size = Ct;
    m_pz_im_frame_size = Kt;
    m_py_im_frame_size = Kt;

    m_start_p = uint32_t(-int32_t(m_pad_h * m_W * m_C));
    m_start_q = uint32_t(-int32_t(m_pad_w * m_C));
    m_delta_p = m_stride_h * m_W * m_C;
    m_delta_q = m_stride_w * m_C;
    m_delta_r = m_dilation_h * m_W * m_C;
    m_delta_s = m_dilation_w * m_C;
    m_end_q = m_start_q + m_Q * m_delta_q;

    m_kernel_base_path = "op/group_conv/device/metal";
    m_defines = {{"T", "bfloat16"}};

    m_opt_add = !m_gz.is_null();

    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_kernels();

    init_locals();
}

void DSConv2dBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int DSConv2dBatch::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * u32_align(m_H * m_W, 32) * m_C;
    case 1:
        return m_R * m_S * m_C * 32;
    case 2:
        return m_C * 32;
    case 3:
        return m_K * m_C;
    case 4:
        return m_K * 32;
    case 5:
        return m_N * u32_align(m_P * m_Q, 32) * m_K;
    default:
        assert(false);
        return 0;
    }
}

int DSConv2dBatch::output_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_P * m_Q, 32) * m_K;
}

std::vector<float> DSConv2dBatch::transform_input(int index, const std::vector<float> &x) {
    std::vector<float> y;
    switch (index) {
    case 0:
        y = util::pad(x, m_N, m_H * m_W, m_C_arg, m_N, u32_align(m_H * m_W, 32), m_C);
        break;
    case 1:
        y = util::pad(x, m_R * m_S, m_C_arg, m_R * m_S, m_C);
        y = expand_weights_bias(y, m_R * m_S);
        break;
    case 2:
        y = util::pad(x, m_C_arg, m_C);
        y = expand_weights_bias(y, 1);
        break;
    case 3:
        y = util::pad(x, m_K_arg, m_C_arg, m_K, m_C);
        y = util::tilize(y, m_K, m_C);
        y = util::make_faces(y);
        break;
    case 4:
        y = util::pad(x, 1, m_K_arg, 32, m_K);
        y = util::tilize(y, 32, m_K);
        y = util::make_faces(y);
        break;
    case 5:
        y = util::pad(x, m_N, m_P * m_Q, m_K_arg, m_N, u32_align(m_P * m_Q, 32), m_K);
        break;
    default:
        assert(false);
        break;
    }
    return y;
}

std::vector<float> DSConv2dBatch::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, u32_align(m_P * m_Q, 32), m_K, m_N, m_P * m_Q, m_K_arg);
}

void DSConv2dBatch::init_config() {
    int volume_x = m_H * m_W * m_C;
    m_cache_lx = ENABLE_CACHE_LX && (volume_x <= 401408);
}

void DSConv2dBatch::validate_globals() {
    uint32_t item_bytes = get_item_bytes(T);
    assert(!m_gx.is_null());
    assert(!m_gw.is_null());
    assert(!m_gb.is_null());
    assert(!m_gw2.is_null());
    assert(!m_gb2.is_null());
    assert(!m_gy.is_null());
    assert(m_gx.bytes() >= input_volume(0) * item_bytes);
    assert(m_gw.bytes() == input_volume(1) * item_bytes);
    assert(m_gb.bytes() == input_volume(2) * item_bytes);
    assert(m_gw2.bytes() == input_volume(3) * item_bytes);
    assert(m_gb2.bytes() == input_volume(4) * item_bytes);
    if (!m_gz.is_null()) {
        assert(m_gz.bytes() >= input_volume(5) * item_bytes);
    }
    assert(m_gy.bytes() >= output_volume(0) * item_bytes);
}

void DSConv2dBatch::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
    m_gmask = core::Global(m_device, core::DataFormat::UINT32, m_mask_size, log2_page_size);
}

void DSConv2dBatch::create_locals() {
    if (m_cache_lx) {
        m_lx = core::Local(m_program, m_grid, T, m_H * m_W * m_C);
    }
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
    m_lmask = core::Local(m_program, m_grid, core::DataFormat::UINT32, m_mask_size);
}

void DSConv2dBatch::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    uint32_t pw_size = m_R * m_S * m_C / 32;
    m_pw =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            pw_size,
            m_pw_frame_size);
    m_pb =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pb_frame_size * 2,
            m_pb_frame_size);
    uint32_t pw2_size = m_K * m_C / 1024;
    m_pw2 =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            pw2_size,
            m_pw2_frame_size);
    m_pb2 =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pb2_frame_size * 2,
            m_pb2_frame_size);
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
    m_pu_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_pu_im_frame_size * 2,
            m_pu_im_frame_size);
    m_pt_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_pt_im_frame_size * 2,
            m_pt_im_frame_size);
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

void DSConv2dBatch::create_kernels() {
    if (m_cache_lx) {
        create_lx_reader();
    } else {
        create_reader();
    }
    if (m_opt_add) {
        create_add_writer();
        create_add_math();
    } else {
        create_writer();
        create_math();
    }
}

void DSConv2dBatch::create_reader() {
    std::string path = m_kernel_base_path + "/dsc_batch_reader.cpp";
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
        global<T> gb2,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        pipe<T> pb2,
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
        uint32 x_stride)
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gb2,
        m_gzero,
        m_gmask,
        m_lzero,
        m_lmask,
        m_px,
        m_pb,
        m_pb2,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_R,
        m_S,
        m_P * m_Q,
        m_start_p,
        m_start_q,
        m_delta_p,
        m_delta_q,
        m_delta_r,
        m_delta_s,
        m_end_q,
        m_zero_size,
        m_mask_size,
        uint32_t(0), // [25] x_pos
        x_stride
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[25] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void DSConv2dBatch::create_lx_reader() {
    std::string path = m_kernel_base_path + "/dsc_batch_lx_reader.cpp";
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
        global<T> gb2,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lx,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        pipe<T> pb2,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 R,
        uint32 S,
        uint32 HWC,
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
        uint32 x_stride)
*/
    uint32_t HW_rnd = u32_align(m_H * m_W, 32);
    uint32_t x_stride = m_batch_size * HW_rnd * m_C;
    std::vector<core::KernelArg> args{
        m_gx,
        m_gb,
        m_gb2,
        m_gzero,
        m_gmask,
        m_lx,
        m_lzero,
        m_lmask,
        m_px,
        m_pb,
        m_pb2,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_R,
        m_S,
        m_H * m_W * m_C,
        m_P * m_Q,
        m_start_p,
        m_start_q,
        m_delta_p,
        m_delta_q,
        m_delta_r,
        m_delta_s,
        m_end_q,
        m_zero_size,
        m_mask_size,
        uint32_t(0), // [27] x_pos
        x_stride
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[27] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void DSConv2dBatch::create_writer() {
    std::string path = m_kernel_base_path + "/dsc_batch_writer.cpp";
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
        global<T> gw2,
        global<T> gy,
        pipe<T> pw,
        pipe<T> pw2,
        pipe<T> py,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 KC,
        uint32 PQ,
        uint32 RSC,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gw2,
        m_gy,
        m_pw,
        m_pw2,
        m_py,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_K * m_C,
        m_P * m_Q,
        m_R * m_S * m_C,
        uint32_t(0), // [12] y_pos
        y_stride
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[12] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void DSConv2dBatch::create_add_writer() {
    std::string path = m_kernel_base_path + "/dsc_batch_add_writer.cpp";
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
        global<T> gw2,
        global<T> gz,
        global<T> gy,
        pipe<T> pw,
        pipe<T> pw2,
        pipe<T> pz,
        pipe<T> py,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 KC,
        uint32 PQ,
        uint32 RSC,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gw2,
        m_gz,
        m_gy,
        m_pw,
        m_pw2,
        m_pz,
        m_py,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_K * m_C,
        m_P * m_Q,
        m_R * m_S * m_C,
        uint32_t(0), // [14] y_pos
        y_stride
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[14] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void DSConv2dBatch::create_math() {
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_kernel_base_path + "/dsc_batch_" + suffix + "_math.cpp";
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
        pipe<T> pw2,
        pipe<T> pb2,
        pipe<T> py,
        pipe<T> pu_im,
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Co,
        uint32 Ci,
        uint32 Ko,
        uint32 Ki,
        uint32 KC,
        uint32 PQ,
        uint32 RS,
        uint32 RSC,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pw2,
        m_pb2,
        m_py,
        m_pu_im,
        m_pt_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Co,
        m_Ci,
        m_Ko,
        m_Ki,
        m_K * m_C,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_C,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void DSConv2dBatch::create_add_math() {
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_kernel_base_path + "/dsc_batch_" + suffix + "_add_math.cpp";
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
        pipe<T> pw2,
        pipe<T> pb2,
        pipe<T> pz,
        pipe<T> py,
        pipe<T> pu_im,
        pipe<T> pt_im,
        pipe<T> pz_im,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 Co,
        uint32 Ci,
        uint32 Ko,
        uint32 Ki,
        uint32 KC,
        uint32 PQ,
        uint32 RS,
        uint32 RSC,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_pw2,
        m_pb2,
        m_pz,
        m_py,
        m_pu_im,
        m_pt_im,
        m_pz_im,
        m_py_im,
        m_N / m_batch_size,
        m_C,
        m_K,
        m_Co,
        m_Ci,
        m_Ko,
        m_Ki,
        m_K * m_C,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_C,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void DSConv2dBatch::init_locals() {
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    std::vector<uint32_t> vmask(m_gmask.bytes() / sizeof(uint32_t), 0);
    compute_mask(vmask);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gzero, vzero.data(), true);
    queue.enqueue_write(m_gmask, vmask.data(), true);
}

void DSConv2dBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = 1;
    } else {
        x = 8;
        y = m_batch_size / 8;
    }
}

void DSConv2dBatch::compute_mask(std::vector<uint32_t> &vmask) {
    int PQ = m_P * m_Q;
    int size = (PQ + 31) / 32;
    size *= m_R * m_S;
    assert(size <= vmask.size());
    int k = 0;
    for (int pq_start = 0; pq_start < PQ; pq_start += 32) {
        for (int r = 0; r < int(m_R); r++) {
            for (int s = 0; s < int(m_S); s++) {
                uint32_t mask = 0;
                uint32_t flag = 1;
                for (int i = 0; i < 32; i++) {
                    bool valid = true;
                    int pq = pq_start + i;
                    if (pq >= PQ) {
                        valid = false;
                    } else {
                        int p = pq / m_Q;
                        int q = pq - p * m_Q;
                        int h = p * m_stride_h - m_pad_h + r * m_dilation_h;
                        int w = q * m_stride_w - m_pad_w + s * m_dilation_w;
                        if (h < 0 || h >= int(m_H) || w < 0 || w >= int(m_W)) {
                            valid = false;
                        }
                    }
                    if (valid) {
                        mask |= flag;
                    }
                    flag <<= 1;
                }
                vmask[k] = mask;
                k++;
            }
        }
    }
}

std::string DSConv2dBatch::get_unary_kernel_suffix() {
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

uint32_t DSConv2dBatch::encode_unary_param0() {
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

bool DSConv2dBatch::is_unary_relu6() {
    return (m_post_op.alpha() == 0.0f && m_post_op.beta() == 6.0f);
}

} // namespace tanto
} // namespace group_conv
} // namespace op
} // namespace ronin

