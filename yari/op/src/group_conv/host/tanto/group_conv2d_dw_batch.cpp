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

#include "host/tanto/util.hpp"
#include "host/tanto/group_conv2d_dw_batch.hpp"

namespace ronin {
namespace op {
namespace group_conv {
namespace tanto {

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;
namespace util = ronin::op::common::util;

// Weights are small in depthwise case
// They always can be cached and multicasted reading them yields no benefits

constexpr bool ENABLE_CACHE_LX = false;
constexpr bool ENABLE_CACHE_LW = true; // reserved (always true)
constexpr bool ENABLE_MCAST = false;
constexpr bool ENABLE_ROW_MAJOR = true;

namespace {

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
//    GroupConv2dDwBatch
//

GroupConv2dDwBatch::GroupConv2dDwBatch(
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
        int groups,
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
            m_groups(uint32_t(groups)),
            m_post_op(post_op),
            m_batch_size(uint32_t(batch_size)) { 
    // these values are required to compute input/output volumes and shapes
    m_C_arg = m_C;
    m_K_arg = m_K;
    m_C = u32_align(m_C, 32);
    m_K = u32_align(m_K, 32);
    m_row_major = ENABLE_ROW_MAJOR;
}

GroupConv2dDwBatch::~GroupConv2dDwBatch() { }

void GroupConv2dDwBatch::init(
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

    assert(m_batch_size < 8 || m_batch_size % 8 == 0);
    // ACHTUNG: Temporary limit 64 is Wormhole-specific
    assert(m_batch_size <= 64);
    assert(m_N % m_batch_size == 0);

    assert(m_C_arg == m_groups);
    assert(m_K_arg == m_groups);

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
    if (!m_row_major) {
        m_px_im_frame_size = Ct;
    }
    m_pt_im_frame_size = Kt;
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

    init_options();
    validate_globals();

    create_globals();
    create_locals();
    create_pipes();
    create_semaphores();
    create_kernels();

    init_locals();
}

void GroupConv2dDwBatch::run() {
    core::Queue queue(m_device, 0);
    queue.enqueue_program(m_program, false);
}

int GroupConv2dDwBatch::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * u32_align(m_H * m_W, 32) * m_C;
    case 1:
        return m_R * m_S * m_K * 32;
    case 2:
        return m_K * 32;
    case 3:
        return m_N * u32_align(m_P * m_Q, 32) * m_K;
    default:
        assert(false);
        return 0;
    }
}

int GroupConv2dDwBatch::output_volume(int index) {
    assert(index == 0);
    return m_N * u32_align(m_P * m_Q, 32) * m_K;
}

std::vector<float> GroupConv2dDwBatch::transform_input(int index, const std::vector<float> &x) {
    std::vector<float> y;
    switch (index) {
    case 0:
        y = util::pad(x, m_N, m_H * m_W, m_C_arg, m_N, u32_align(m_H * m_W, 32), m_C);
        break;
    case 1:
        if (m_row_major) {
            y = util::pad(x, m_R * m_S, m_K_arg, m_R * m_S, m_K);
            y = expand_weights_bias(y, m_R * m_S);
        } else {
            y = util::pad(x, 1, m_R * m_S, m_K_arg, 32, m_R * m_S, m_K);
            y = util::tilize(y, 32, m_R * m_S * m_K);
            y = util::make_faces(y);
        }
        break;
    case 2:
        if (m_row_major) {
            y = util::pad(x, m_K_arg, m_K);
            y = expand_weights_bias(y, 1);
        } else {
            y = util::pad(x, 1, m_K_arg, 32, m_K);
            y = util::tilize(y, 32, m_K);
            y = util::make_faces(y);
        }
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

std::vector<float> GroupConv2dDwBatch::transform_output(int index, const std::vector<float> &x) {
    assert(index == 0);
    return util::unpad(x, m_N, u32_align(m_P * m_Q, 32), m_K, m_N, m_P * m_Q, m_K_arg);
}

void GroupConv2dDwBatch::init_config() {
    int volume_x = m_H * m_W * m_C;
    int volume_w = m_R * m_S * m_K * 32;
    // ACHTUNG: Temporary L1 limit for input caching is Wormhole-specific
    int volume_limit = (128 + 256) * 1024;
    m_cache_lx = false;
    m_cache_lw = true; // reserved (always true)
#if 1 // TODO: Revise this
if (volume_x <= 401408) m_cache_lx = true;
#endif
     if (ENABLE_CACHE_LX && volume_x + volume_w <= volume_limit) {
        m_cache_lx = true;
    }
    m_mcast = ENABLE_MCAST && (m_batch_size >= 16);
    // m_row_major is set at construction
#if 0
printf("@@@ CACHE_LX: %d CACHE_LW: %d MCAST %d ROW_MAJOR %d\n", 
int(m_cache_lx), int(m_cache_lw), int(m_mcast), int(m_row_major));
#endif
}

void GroupConv2dDwBatch::init_options() {
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

void GroupConv2dDwBatch::validate_globals() {
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

void GroupConv2dDwBatch::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    m_gzero = core::Global(m_device, T, m_zero_size, log2_page_size);
    m_gmask = core::Global(m_device, core::DataFormat::UINT32, m_mask_size, log2_page_size);
}

void GroupConv2dDwBatch::create_locals() {
    if (m_cache_lx) {
        m_lx = core::Local(m_program, m_grid, T, m_H * m_W * m_C);
    }
    m_lzero = core::Local(m_program, m_grid, T, m_zero_size);
    m_lmask = core::Local(m_program, m_grid, core::DataFormat::UINT32, m_mask_size);
}

void GroupConv2dDwBatch::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_px_frame_size * 2,
            m_px_frame_size);
    uint32_t pw_size = m_R * m_S * m_K / 32;
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
    if (!m_row_major) {
        m_px_im =
            core::Pipe(
                m_program,
                m_grid,
                core::PipeKind::INTERMED,
                T,
                m_px_im_frame_size * 2,
                m_px_im_frame_size);
    }
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
    m_pt_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_pt_im_frame_size * 2,
            m_pt_im_frame_size);
    m_py_im =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INTERMED,
            T,
            m_py_im_frame_size * 2,
            m_py_im_frame_size);
}

void GroupConv2dDwBatch::create_semaphores() {
    if (m_mcast) {
        m_sem_send = core::Semaphore(m_program, m_grid, 0);
        m_sem_recv = core::Semaphore(m_program, m_grid, 0);
    }
}

void GroupConv2dDwBatch::create_kernels() {
    uint32_t reader_options = m_options & OPT_BIAS;
    uint32_t writer_options = m_options & OPT_ADD;
    uint32_t math_options = m_options;
    if (reader_options == OPT_BIAS) {
        if (m_cache_lx) {
            create_lx_bias_reader();
        } else {
            create_bias_reader();
        }
    } else {
        // not yet implemented
        assert(false);
    }
    if (writer_options != OPT_ADD) {
        if (m_mcast) {
            create_mcast_writer();
        } else {
            create_writer();
        }
    } else {
        // not yet implemented
        assert(false);
    }
    if (math_options == OPT_BIAS) {
        if (m_row_major) {
            create_rm_bias_math();
        } else {
            create_bias_math();
        }
    } else if (math_options == (OPT_BIAS | OPT_UNARY)) {
        if (m_row_major) {
            create_rm_bias_unary_math();
        } else {
            create_bias_unary_math();
        }
    } else {
        // not yet implemented
        assert(false);
    }
}

void GroupConv2dDwBatch::create_bias_reader() {
    std::string path = m_kernel_base_path + "/dw_batch_bias_reader.cpp";
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
        uint32 x_stride)
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
        uint32_t(0), // [23] x_pos
        x_stride
    };
    uint32_t x_inc = HW_rnd * m_C;
    uint32_t x_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[23] = x_pos;
            m_reader.set_args(x, y, args);
            x_pos += x_inc;
        }
    }
}

void GroupConv2dDwBatch::create_lx_bias_reader() {
    std::string path = m_kernel_base_path + "/dw_batch_lx_bias_reader.cpp";
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

void GroupConv2dDwBatch::create_writer() {
    std::string path = m_kernel_base_path + "/dw_batch_writer.cpp";
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
        uint32 N,
        uint32 K,
        uint32 PQ,
        uint32 RSK,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
    std::vector<core::KernelArg> args{
        m_gw,
        m_gy,
        m_pw,
        m_py,
        m_N / m_batch_size,
        m_K,
        m_P * m_Q,
        m_R * m_S * m_K,
        uint32_t(0), // [8] y_pos
        y_stride
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            args[8] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void GroupConv2dDwBatch::create_mcast_writer() {
    std::string path = m_kernel_base_path + "/dw_batch_mcast_writer.cpp";
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
        uint32 RSK,
        uint32 y_pos,
        uint32 y_stride)
*/
    uint32_t num_dests = m_batch_size / 8 - 1; // Wormhole-specific, temporary
    uint32_t PQ_rnd = u32_align(m_P * m_Q, 32);
    uint32_t y_stride = (m_batch_size - 1) * PQ_rnd * m_K;
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
        m_P * m_Q,
        m_R * m_S * m_K,
        uint32_t(0), // [16] y_pos
        y_stride
    };
    uint32_t y_inc = PQ_rnd * m_K;
    uint32_t y_pos = 0;
    for (uint32_t x = m_x_start; x <= m_x_end; x++) {
        for (uint32_t y = m_y_start; y <= m_y_end; y++) {
            uint32_t send_mode = (y == m_y_start) ? 1 : 0;
            uint32_t x0_phy, y0_phy;
            m_device.worker_core_from_logical_core(x, m_y_start, x0_phy, y0_phy);
            uint32_t x1_phy, y1_phy;
            m_device.worker_core_from_logical_core(x, m_y_end, x1_phy, y1_phy);
            args[6] = send_mode;
            args[7] = x0_phy;
            args[8] = y0_phy;
            args[9] = x1_phy;
            args[10] = y1_phy;
            args[16] = y_pos;
            m_writer.set_args(x, y, args);
            y_pos += y_inc;
        }
    }
}

void GroupConv2dDwBatch::create_bias_math() {
    std::string path = m_kernel_base_path + "/dw_batch_bias_math.cpp";
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
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSK)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_pt_im,
        m_py_im,
        m_N / m_batch_size,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K
    };
    m_math.set_args(m_grid, args);
}

void GroupConv2dDwBatch::create_rm_bias_math() {
    std::string path = m_kernel_base_path + "/dw_batch_rm_bias_math.cpp";
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
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSK)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_pt_im,
        m_py_im,
        m_N / m_batch_size,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K
    };
    m_math.set_args(m_grid, args);
}

void GroupConv2dDwBatch::create_bias_unary_math() {
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_kernel_base_path + "/dw_batch_bias_" + suffix + "_math.cpp";
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
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSK,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_px_im,
        m_pt_im,
        m_py_im,
        m_N / m_batch_size,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void GroupConv2dDwBatch::create_rm_bias_unary_math() {
    std::string suffix = get_unary_kernel_suffix();
    std::string path = m_kernel_base_path + "/dw_batch_rm_bias_" + suffix + "_math.cpp";
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
        pipe<T> pt_im,
        pipe<T> py_im,
        uint32 N,
        uint32 K,
        uint32 Ko,
        uint32 Ki,
        uint32 PQ,
        uint32 RS,
        uint32 RSK,
        uint32 unary_param0)
*/
    uint32_t unary_param0 = encode_unary_param0();
    std::vector<core::KernelArg> args{
        m_px,
        m_pw,
        m_pb,
        m_py,
        m_pt_im,
        m_py_im,
        m_N / m_batch_size,
        m_K,
        m_Ko,
        m_Ki,
        m_P * m_Q,
        m_R * m_S,
        m_R * m_S * m_K,
        unary_param0
    };
    m_math.set_args(m_grid, args);
}

void GroupConv2dDwBatch::init_locals() {
    std::vector<uint32_t> vzero(m_gzero.bytes() / sizeof(uint32_t), 0);
    std::vector<uint32_t> vmask(m_gmask.bytes() / sizeof(uint32_t), 0);
    compute_mask(vmask);
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gzero, vzero.data(), true);
    queue.enqueue_write(m_gmask, vmask.data(), true);
}

void GroupConv2dDwBatch::compute_grid_dims(uint32_t &x, uint32_t &y) {
    // ACHTUNG: Temporary limit 8 is Wormhole-specific
    if (m_batch_size <= 8) {
        x = m_batch_size;
        y = 1;
    } else {
        x = 8;
        y = m_batch_size / 8;
    }
}

void GroupConv2dDwBatch::compute_mask(std::vector<uint32_t> &vmask) {
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

std::string GroupConv2dDwBatch::get_unary_kernel_suffix() {
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

uint32_t GroupConv2dDwBatch::encode_unary_param0() {
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

bool GroupConv2dDwBatch::is_unary_relu6() {
    return (m_post_op.alpha() == 0.0f && m_post_op.beta() == 6.0f);
}

} // namespace tanto
} // namespace group_conv
} // namespace op
} // namespace ronin

