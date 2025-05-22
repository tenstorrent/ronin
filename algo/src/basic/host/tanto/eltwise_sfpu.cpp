// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"
#include "host/tanto/eltwise_sfpu.hpp"

//
//    Basic elementwise SFPU operations implemented using Tanto host API.
// 
//    Compilation commands:
// 
//    tanto --mode=read -DT=bfloat16 \
//        tanto/eltwise_sfpu_reader.cpp >metal/eltwise_sfpu_reader.cpp
//    tanto --mode=write -DT=bfloat16 \
//        tanto/eltwise_sfpu_writer.cpp >metal/eltwise_sfpu_writer.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=0 \
//        tanto/eltwise_sfpu_math.cpp >metal/eltwise_abs_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=1 \
//        tanto/eltwise_sfpu_math.cpp >metal/eltwise_acos_math.cpp
//    tanto --mode=compute -DT=bfloat16 -P0=2 \
//        tanto/eltwise_sfpu_math.cpp >metal/eltwise_asin_math.cpp
//    ...
//

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    EltwiseSfpu
//

EltwiseSfpu::EltwiseSfpu() { }

EltwiseSfpu::~EltwiseSfpu() { }

void EltwiseSfpu::init(
        const core::Device &device,
        EltwiseSfpuOp op, 
        uint32_t iparam,
        float fparam,
        int N) {
    assert(N % TILE_SIZE == 0);

    m_device = device;
    m_op = op;
    m_iparam = iparam;
    m_fparam = fparam;
    m_N = uint32_t(N);
    m_pipe_frame_size = 1;
    m_block_tiles = m_pipe_frame_size;
    m_num_blocks = m_N / (m_block_tiles * TILE_SIZE);

    m_program = core::Program(m_device);
    m_grid = core::Grid(m_program, 0, 0);

    m_kernel_base_path = "algo/basic/device/metal";
    m_defines = {{"T", "bfloat16"}};

    create_globals();
    create_pipes();
    create_kernels();
}

void EltwiseSfpu::run(const void *x, void *y) {
    core::Queue queue(m_device, 0);
    queue.enqueue_write(m_gx, x, false);
    queue.enqueue_program(m_program, false);
    queue.enqueue_read(m_gy, y, false);
}

void EltwiseSfpu::create_globals() {
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024 (one tile)
    m_gx = core::Global(m_device, T, m_N, log2_page_size);
    m_gy = core::Global(m_device, T, m_N, log2_page_size);
}

void EltwiseSfpu::create_pipes() {
    m_px =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::INPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
    m_py =
        core::Pipe(
            m_program,
            m_grid,
            core::PipeKind::OUTPUT,
            T,
            m_pipe_frame_size * 2,
            m_pipe_frame_size);
}

void EltwiseSfpu::create_kernels() {
    create_reader();
    create_writer();
    create_math();
}

void EltwiseSfpu::create_reader() {
    std::string path = m_kernel_base_path + "/eltwise_sfpu_reader.cpp";
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
        pipe<T> px,
        uint32 gx_pos,
        uint32 num_blocks,
        uint32 block_tiles)
*/
    std::vector<core::KernelArg> args{
        m_gx,
        m_px,
        uint32_t(0), // gx_pos
        m_num_blocks,
        m_block_tiles
    };
    m_reader.set_args(m_grid, args);
}

void EltwiseSfpu::create_writer() {
    std::string path = m_kernel_base_path + "/eltwise_sfpu_writer.cpp";
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
        global<T> gy, 
        pipe<T> py, 
        uint32 gy_pos,
        uint32 num_blocks,
        uint32 block_tiles)
*/
    std::vector<core::KernelArg> args{
        m_gy,
        m_py,
        uint32_t(0), // gy_pos
        m_num_blocks,
        m_block_tiles
    };
    m_writer.set_args(m_grid, args);
}

void EltwiseSfpu::create_math() {
    std::string path = m_kernel_base_path + "/" + make_math_name() + ".cpp";
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
        pipe<T> py, 
        uint32 num_blocks,
        uint32 block_tiles,
        uint32 iparam, 
        float fparam)
*/
    std::vector<core::KernelArg> args{
        m_px,
        m_py,
        m_num_blocks,
        m_block_tiles,
        m_iparam,
        float_as_u32(m_fparam)
    };
    m_math.set_args(m_grid, args);
}

std::unordered_map<EltwiseSfpuOp, std::string> g_math_name_map = {
    {EltwiseSfpuOp::Abs, "abs"},
    {EltwiseSfpuOp::Acos, "acos"},
    {EltwiseSfpuOp::Asin, "asin"},
    {EltwiseSfpuOp::Atan, "atan"},
    {EltwiseSfpuOp::Cos, "cos"},
    {EltwiseSfpuOp::Elu, "elu"},
    {EltwiseSfpuOp::Eqz, "eqz"},
    {EltwiseSfpuOp::Erf, "erf"},
    {EltwiseSfpuOp::Erfc, "erfc"},
    {EltwiseSfpuOp::Erfinv, "erfinv"},
    {EltwiseSfpuOp::Exp, "exp"},
    {EltwiseSfpuOp::Exp2, "exp2"},
    {EltwiseSfpuOp::Expm1, "expm1"},
    {EltwiseSfpuOp::Gelu, "gelu"},
    {EltwiseSfpuOp::Gez, "gez"},
    {EltwiseSfpuOp::Gtz, "gtz"},
    {EltwiseSfpuOp::Heaviside, "heaviside"},
    {EltwiseSfpuOp::I0, "i0"},
    {EltwiseSfpuOp::Isfinite, "isfinite"},
    {EltwiseSfpuOp::Isinf, "isinf"},
    {EltwiseSfpuOp::Isnan, "isnan"},
    {EltwiseSfpuOp::Isneginf, "isneginf"},
    {EltwiseSfpuOp::Isposinf, "isposinf"},
    {EltwiseSfpuOp::LeakyRelu, "leaky_relu"},
    {EltwiseSfpuOp::Lez, "lez"},
    {EltwiseSfpuOp::Log, "log"},
    {EltwiseSfpuOp::LogWithBase, "log_with_base"},
    {EltwiseSfpuOp::LogicalNot, "logical_not"},
    {EltwiseSfpuOp::Ltz, "ltz"},
    {EltwiseSfpuOp::Nez, "nez"},
    {EltwiseSfpuOp::Power, "power"},
    {EltwiseSfpuOp::Recip, "recip"},
    {EltwiseSfpuOp::Relu, "relu"},
    {EltwiseSfpuOp::ReluMax, "relu_max"},
    {EltwiseSfpuOp::ReluMin, "relu_min"},
    {EltwiseSfpuOp::Rsqrt, "rsqrt"},
    {EltwiseSfpuOp::Sigmoid, "sigmoid"},
    {EltwiseSfpuOp::Sign, "sign"},
    {EltwiseSfpuOp::Signbit, "signbit"},
    {EltwiseSfpuOp::Sin, "sin"},
    {EltwiseSfpuOp::Sqrt, "sqrt"},
    {EltwiseSfpuOp::Square, "square"},
    {EltwiseSfpuOp::Tan, "tan"},
    {EltwiseSfpuOp::Tanh, "tanh"}
};

std::string EltwiseSfpu::make_math_name() {
    // makes name of precompiled kernel
    auto it = g_math_name_map.find(m_op);
    if (it == g_math_name_map.end()) {
        assert(false);
        return "invalid";
    }
    return "eltwise_" + it->second + "_math";
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

