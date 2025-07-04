// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "core/graph_builder.hpp"

namespace ronin {
namespace tanto {
namespace front {

enum class MathBuiltinId {
    NONE = 0,
    PACK,
    PACK_ROW,
    PACK_COL,
    PACK_SCALAR,
    COPY,
    ADD,
    SUB,
    MUL,
    ADD_BCAST_ROWS,
    SUB_BCAST_ROWS,
    MUL_BCAST_ROWS,
    ADD_BCAST_COLS,
    SUB_BCAST_COLS,
    MUL_BCAST_COLS,
    ADD_BCAST_SCALAR,
    SUB_BCAST_SCALAR,
    MUL_BCAST_SCALAR,
    MATMUL,
    REDUCE_MAX_ROWS,
    REDUCE_MAX_COLS,
    REDUCE_MAX_SCALAR,
    REDUCE_SUM_ROWS,
    REDUCE_SUM_COLS,
    REDUCE_SUM_SCALAR,
    TRANSPOSE,
    TILIZE_BLOCK,
    UNTILIZE_BLOCK,
    COPY_DST,
    ADD_DST,
    SUB_DST,
    RSUB_DST,
    MUL_DST,
    DIV_DST,
    POWER_DST,
    ABS,
    ACOS,
    ADD_SCALAR,
    ASIN,
    ATAN,
    CAST_BF16_U16,
    CAST_U16_BF16,
    CEIL,
    COS,
    DIV_SCALAR,
    ELU,
    EQZ,
    ERF,
    ERFC,
    ERFINV,
    EXP,
    EXP2,
    EXPM1,
    FILL,
    FLOOR,
    GELU,
    GEZ,
    GTZ,
    HEAVISIDE,
    I0,
    ISFINITE,
    ISINF,
    ISNAN,
    ISNEGINF,
    ISPOSINF,
    LEAKY_RELU,
    LEZ,
    LOG,
    LOG_WITH_BASE,
    LOGICAL_NOT,
    LTZ,
    MAX,
    MUL_SCALAR,
    NEZ,
    POWER, 
    RECIP,
    RELU,
    RELU_MAX,
    RELU_MIN,
    RSQRT,
    RSUB_SCALAR,
    SIGMOID,
    SIGN,
    SIGNBIT,
    SIN,
    SQRT,
    SQUARE,
    SUB_SCALAR,
    TAN,
    TANH,
    // count
    __COUNT
};

static constexpr int MathBuiltinIdCount = int(MathBuiltinId::__COUNT);

struct MathInitFuncGroup {
    static constexpr int UNPACK = 0;
    static constexpr int MATH = 1;
    static constexpr int PACK = 2;
    static constexpr int SFPU = 3;
    static constexpr int COUNT = 4;
};

enum class MathInitFunc {
    // special
    NONE,
    UNDEF,
    // unpack
    UNPACK_BINARY,
    UNPACK_BCAST_ROWS,
    UNPACK_BCAST_COLS,
    UNPACK_BCAST_SCALAR,
    UNPACK_MATMUL,
    UNPACK_UNARY,
    UNPACK_REDUCE_ROWS,
    UNPACK_REDUCE_COLS,
    UNPACK_REDUCE_SCALAR,
    UNPACK_TRANSPOSE,
    UNPACK_TILIZE_BLOCK,
    UNPACK_UNTILIZE_BLOCK,
    // pack
    PACK,
    PACK_ROW,
    PACK_COL,
    PACK_SCALAR,
    // math
    COPY,
    ADD,
    SUB,
    MUL,
    ADD_BCAST_ROWS,
    SUB_BCAST_ROWS,
    MUL_BCAST_ROWS,
    ADD_BCAST_COLS,
    SUB_BCAST_COLS,
    MUL_BCAST_COLS,
    ADD_BCAST_SCALAR,
    SUB_BCAST_SCALAR,
    MUL_BCAST_SCALAR,
    MATMUL,
    REDUCE_MAX_ROWS,
    REDUCE_MAX_COLS,
    REDUCE_MAX_SCALAR,
    REDUCE_SUM_ROWS,
    REDUCE_SUM_COLS,
    REDUCE_SUM_SCALAR,
    TRANSPOSE,
    // sfpu
    COPY_DST,
    ADD_DST,
    SUB_DST,
    RSUB_DST,
    MUL_DST,
    DIV_DST,
    POWER_DST,
    ABS,
    ACOS,
    ASIN,
    ATAN,
    BINARY_SCALAR,
    CAST,
    CEIL,
    COS,
    ELU,
    EQZ,
    ERF,
    ERFC,
    ERFINV,
    EXP,
    EXP2,
    EXPM1,
    FILL,
    FLOOR,
    GELU,
    GEZ,
    GTZ,
    HEAVISIDE,
    I0,
    ISFINITE,
    ISINF,
    ISNAN,
    ISNEGINF,
    ISPOSINF,
    LEAKY_RELU,
    LEZ,
    LOG,
    LOG_WITH_BASE,
    LOGICAL_NOT,
    LTZ,
    MAX,
    NEZ,
    POWER,
    RECIP,
    RELU,
    RELU_MAX,
    RELU_MIN,
    RSQRT,
    SIGMOID,
    SIGN,
    SIGNBIT,
    SIN,
    SQRT,
    SQUARE,
    TAN,
    TANH,
    // count
    __COUNT
};

static constexpr int MathInitFuncCount = int(MathInitFunc::__COUNT);

class MathInitBuiltinHandler {
public:
    MathInitBuiltinHandler();
    ~MathInitBuiltinHandler();
public:
    MathBuiltinId map(const std::string &class_name, const std::string &method_name);
};

class MathInitFuncHandler {
public:
    MathInitFuncHandler();
    ~MathInitFuncHandler();
public:
    MathInitFunc map(MathBuiltinId id, int group);
private:
    void init();
    void enter_unpack(MathBuiltinId id, MathInitFunc func) {
        enter(id, MathInitFuncGroup::UNPACK, func);
    }
    void enter_math(MathBuiltinId id, MathInitFunc func) {
        enter(id, MathInitFuncGroup::MATH, func);
    }
    void enter_pack(MathBuiltinId id, MathInitFunc func) {
        enter(id, MathInitFuncGroup::PACK, func);
    }
    void enter_sfpu(MathBuiltinId id, MathInitFunc func) {
        enter(id, MathInitFuncGroup::SFPU, func);
    }
    void enter(MathBuiltinId id, int group, MathInitFunc func);
private:
    MathInitFunc m_map[MathBuiltinIdCount][MathInitFuncGroup::COUNT];
};

// public functions

std::string get_math_init_func_name(MathInitFunc func);

} // namespace front
} // namespace tanto
} // namespace ronin

