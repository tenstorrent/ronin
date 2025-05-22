// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "host/core/api.hpp"

#include "host/tanto/util.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

namespace core = ronin::tanto::host;

enum class EltwiseSfpuOp {
    Abs,
    Acos,
    Asin,
    Atan,
    Cos,
    Elu,
    Eqz,
    Erf,
    Erfc,
    Erfinv,
    Exp,
    Exp2,
    Expm1,
    Gelu,
    Gez,
    Gtz,
    Heaviside,
    I0,
    Isfinite,
    Isinf,
    Isnan,
    Isneginf,
    Isposinf,
    LeakyRelu,
    Lez,
    Log,
    LogWithBase,
    LogicalNot,
    Ltz,
    Nez,
    Power,
    Recip,
    Relu,
    ReluMax,
    ReluMin,
    Rsqrt,
    Sigmoid,
    Sign,
    Signbit,
    Sin,
    Sqrt,
    Square,
    Tan,
    Tanh
};

class EltwiseSfpu {
public:
    EltwiseSfpu();
    ~EltwiseSfpu();
public:
    void init(
        const core::Device &device,
        EltwiseSfpuOp op, 
        uint32_t iparam,
        float fparam,
        int N);
    void run(const void *x, void *y);
private:
    void create_globals();
    void create_pipes();
    void create_kernels();
    void create_reader();
    void create_writer();
    void create_math();
    std::string make_math_name();
private:
    static const core::DataFormat T = core::DataFormat::BFLOAT16;
    static const uint32_t TILE_SIZE = 1024;
private:
    core::Device m_device;
    EltwiseSfpuOp m_op = EltwiseSfpuOp(0);
    uint32_t m_iparam = 0;
    float m_fparam = 0.0f;
    uint32_t m_N = 0;
    uint32_t m_pipe_frame_size = 0;
    uint32_t m_block_tiles = 0;
    uint32_t m_num_blocks = 0;
    core::Program m_program;
    core::Grid m_grid;
    core::Global m_gx;
    core::Global m_gy;
    core::Pipe m_px;
    core::Pipe m_py;
    core::Kernel m_reader;
    core::Kernel m_writer;
    core::Kernel m_math;
    std::string m_kernel_base_path;
    std::map<std::string, std::string> m_defines;
};

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

