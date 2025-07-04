// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_map>

#include "core/graph_builder.hpp"
#include "core/math_init_builtin.hpp"

namespace ronin {
namespace tanto {
namespace front {

namespace {

std::unordered_map<std::string, MathBuiltinId> g_math_builtin_id_map = {
    {"pack", MathBuiltinId::PACK},
    {"pack_row", MathBuiltinId::PACK_ROW},
    {"pack_col", MathBuiltinId::PACK_COL},
    {"pack_scalar", MathBuiltinId::PACK_SCALAR},
    {"copy", MathBuiltinId::COPY},
    {"add", MathBuiltinId::ADD},
    {"sub", MathBuiltinId::SUB},
    {"mul", MathBuiltinId::MUL},
    {"add_bcast_rows", MathBuiltinId::ADD_BCAST_ROWS},
    {"sub_bcast_rows", MathBuiltinId::SUB_BCAST_ROWS},
    {"mul_bcast_rows", MathBuiltinId::MUL_BCAST_ROWS},
    {"add_bcast_cols", MathBuiltinId::ADD_BCAST_COLS},
    {"sub_bcast_cols", MathBuiltinId::SUB_BCAST_COLS},
    {"mul_bcast_cols", MathBuiltinId::MUL_BCAST_COLS},
    {"add_bcast_scalar", MathBuiltinId::ADD_BCAST_SCALAR},
    {"sub_bcast_scalar", MathBuiltinId::SUB_BCAST_SCALAR},
    {"mul_bcast_scalar", MathBuiltinId::MUL_BCAST_SCALAR},
    {"matmul", MathBuiltinId::MATMUL},
    {"reduce_max_rows", MathBuiltinId::REDUCE_MAX_ROWS},
    {"reduce_max_cols", MathBuiltinId::REDUCE_MAX_COLS},
    {"reduce_max_scalar", MathBuiltinId::REDUCE_MAX_SCALAR},
    {"reduce_sum_rows", MathBuiltinId::REDUCE_SUM_ROWS},
    {"reduce_sum_cols", MathBuiltinId::REDUCE_SUM_COLS},
    {"reduce_sum_scalar", MathBuiltinId::REDUCE_SUM_SCALAR},
    {"transpose", MathBuiltinId::TRANSPOSE},
    {"tilize_block", MathBuiltinId::TILIZE_BLOCK},
    {"untilize_block", MathBuiltinId::UNTILIZE_BLOCK},
    {"copy_dst", MathBuiltinId::COPY_DST},
    {"add_dst", MathBuiltinId::ADD_DST},
    {"sub_dst", MathBuiltinId::SUB_DST},
    {"rsub_dst", MathBuiltinId::RSUB_DST},
    {"mul_dst", MathBuiltinId::MUL_DST},
    {"div_dst", MathBuiltinId::DIV_DST},
    {"power_dst", MathBuiltinId::POWER_DST},
    {"abs", MathBuiltinId::ABS},
    {"acos", MathBuiltinId::ACOS},
    {"add_scalar", MathBuiltinId::ADD_SCALAR},
    {"asin", MathBuiltinId::ASIN},
    {"atan", MathBuiltinId::ATAN},
    {"cast_bf16_u16", MathBuiltinId::CAST_BF16_U16},
    {"cast_u16_bf16", MathBuiltinId::CAST_U16_BF16},
    {"ceil", MathBuiltinId::CEIL},
    {"cos", MathBuiltinId::COS},
    {"div_scalar", MathBuiltinId::DIV_SCALAR},
    {"elu", MathBuiltinId::ELU},
    {"eqz", MathBuiltinId::EQZ},
    {"erf", MathBuiltinId::ERF},
    {"erfc", MathBuiltinId::ERFC},
    {"erfinv", MathBuiltinId::ERFINV},
    {"exp", MathBuiltinId::EXP},
    {"exp2", MathBuiltinId::EXP2},
    {"expm1", MathBuiltinId::EXPM1},
    {"fill", MathBuiltinId::FILL},
    {"floor", MathBuiltinId::FLOOR},
    {"gelu", MathBuiltinId::GELU},
    {"gez", MathBuiltinId::GEZ},
    {"gtz", MathBuiltinId::GTZ},
    {"heaviside", MathBuiltinId::HEAVISIDE},
    {"i0", MathBuiltinId::I0},
    {"isfinite", MathBuiltinId::ISFINITE},
    {"isinf", MathBuiltinId::ISINF},
    {"isnan", MathBuiltinId::ISNAN},
    {"isneginf", MathBuiltinId::ISNEGINF},
    {"isposinf", MathBuiltinId::ISPOSINF},
    {"leaky_relu", MathBuiltinId::LEAKY_RELU},
    {"lez", MathBuiltinId::LEZ},
    {"log", MathBuiltinId::LOG},
    {"log_with_base", MathBuiltinId::LOG_WITH_BASE},
    {"logical_not", MathBuiltinId::LOGICAL_NOT},
    {"ltz", MathBuiltinId::LTZ},
    {"max", MathBuiltinId::MAX},
    {"mul_scalar", MathBuiltinId::MUL_SCALAR},
    {"nez", MathBuiltinId::NEZ},
    {"power", MathBuiltinId::POWER},
    {"recip", MathBuiltinId::RECIP},
    {"relu", MathBuiltinId::RELU},
    {"relu_max", MathBuiltinId::RELU_MAX},
    {"relu_min", MathBuiltinId::RELU_MIN},
    {"rsqrt", MathBuiltinId::RSQRT},
    {"rsub_scalar", MathBuiltinId::RSUB_SCALAR},
    {"sigmoid", MathBuiltinId::SIGMOID},
    {"sign", MathBuiltinId::SIGN},
    {"signbit", MathBuiltinId::SIGNBIT},
    {"sin", MathBuiltinId::SIN},
    {"sqrt", MathBuiltinId::SQRT},
    {"square", MathBuiltinId::SQUARE},
    {"sub_scalar", MathBuiltinId::SUB_SCALAR},
    {"tan", MathBuiltinId::TAN},
    {"tanh", MathBuiltinId::TANH}
};

} // namespace

//
//     MathInitBuiltinHandler
//

MathInitBuiltinHandler::MathInitBuiltinHandler() { }

MathInitBuiltinHandler::~MathInitBuiltinHandler() { }

MathBuiltinId MathInitBuiltinHandler::map(
        const std::string &class_name, const std::string &method_name) {
    if (class_name.empty()) {
        if (method_name == "tilize_block") {
            return MathBuiltinId::TILIZE_BLOCK;
        }
        if (method_name == "untilize_block") {
            return MathBuiltinId::UNTILIZE_BLOCK;
        }
        return MathBuiltinId::NONE;
    }
    if (class_name == "math") {
        auto it = g_math_builtin_id_map.find(method_name);
        if (it != g_math_builtin_id_map.end()) {
            return it->second;
        }
        return MathBuiltinId::NONE;
    }
    return MathBuiltinId::NONE;
}

//
//    MathInitFuncHandler
//

MathInitFuncHandler::MathInitFuncHandler() { 
    init();
}

MathInitFuncHandler::~MathInitFuncHandler() { }

MathInitFunc MathInitFuncHandler::map(MathBuiltinId id, int group) {
    int index = int(id);
    assert(index >= 0 && index < MathBuiltinIdCount);
    assert(group >= 0 && group < MathInitFuncGroup::COUNT);
    return m_map[index][group];
}

void MathInitFuncHandler::init() {
    for (int i = 0; i < MathBuiltinIdCount; i++) {
        for (int k = 0; k < MathInitFuncGroup::COUNT; k++) {
            m_map[i][k] = MathInitFunc::NONE;
        }
    }
    // unpack
    enter_unpack(MathBuiltinId::COPY, MathInitFunc::UNPACK_UNARY);
    enter_unpack(MathBuiltinId::ADD, MathInitFunc::UNPACK_BINARY);
    enter_unpack(MathBuiltinId::SUB, MathInitFunc::UNPACK_BINARY);
    enter_unpack(MathBuiltinId::MUL, MathInitFunc::UNPACK_BINARY);
    enter_unpack(MathBuiltinId::ADD_BCAST_ROWS, MathInitFunc::UNPACK_BCAST_ROWS);
    enter_unpack(MathBuiltinId::SUB_BCAST_ROWS, MathInitFunc::UNPACK_BCAST_ROWS);
    enter_unpack(MathBuiltinId::MUL_BCAST_ROWS, MathInitFunc::UNPACK_BCAST_ROWS);
    enter_unpack(MathBuiltinId::ADD_BCAST_COLS, MathInitFunc::UNPACK_BCAST_COLS);
    enter_unpack(MathBuiltinId::SUB_BCAST_COLS, MathInitFunc::UNPACK_BCAST_COLS);
    enter_unpack(MathBuiltinId::MUL_BCAST_COLS, MathInitFunc::UNPACK_BCAST_COLS);
    enter_unpack(MathBuiltinId::ADD_BCAST_SCALAR, MathInitFunc::UNPACK_BCAST_SCALAR);
    enter_unpack(MathBuiltinId::SUB_BCAST_SCALAR, MathInitFunc::UNPACK_BCAST_SCALAR);
    enter_unpack(MathBuiltinId::MUL_BCAST_SCALAR, MathInitFunc::UNPACK_BCAST_SCALAR);
    enter_unpack(MathBuiltinId::MATMUL, MathInitFunc::UNPACK_MATMUL);
    enter_unpack(MathBuiltinId::REDUCE_MAX_ROWS, MathInitFunc::UNPACK_REDUCE_ROWS);
    enter_unpack(MathBuiltinId::REDUCE_MAX_COLS, MathInitFunc::UNPACK_REDUCE_COLS);
    enter_unpack(MathBuiltinId::REDUCE_MAX_SCALAR, MathInitFunc::UNPACK_REDUCE_SCALAR);
    enter_unpack(MathBuiltinId::REDUCE_SUM_ROWS, MathInitFunc::UNPACK_REDUCE_ROWS);
    enter_unpack(MathBuiltinId::REDUCE_SUM_COLS, MathInitFunc::UNPACK_REDUCE_COLS);
    enter_unpack(MathBuiltinId::REDUCE_SUM_SCALAR, MathInitFunc::UNPACK_REDUCE_SCALAR);
    enter_unpack(MathBuiltinId::TRANSPOSE, MathInitFunc::UNPACK_TRANSPOSE);
    enter_unpack(MathBuiltinId::TILIZE_BLOCK, MathInitFunc::UNPACK_TILIZE_BLOCK);
    enter_unpack(MathBuiltinId::UNTILIZE_BLOCK, MathInitFunc::UNPACK_UNTILIZE_BLOCK);
    // math
    enter_math(MathBuiltinId::COPY, MathInitFunc::COPY);
    enter_math(MathBuiltinId::ADD, MathInitFunc::ADD);
    enter_math(MathBuiltinId::SUB, MathInitFunc::SUB);
    enter_math(MathBuiltinId::MUL, MathInitFunc::MUL);
    enter_math(MathBuiltinId::ADD_BCAST_ROWS, MathInitFunc::ADD_BCAST_ROWS);
    enter_math(MathBuiltinId::SUB_BCAST_ROWS, MathInitFunc::SUB_BCAST_ROWS);
    enter_math(MathBuiltinId::MUL_BCAST_ROWS, MathInitFunc::MUL_BCAST_ROWS);
    enter_math(MathBuiltinId::ADD_BCAST_COLS, MathInitFunc::ADD_BCAST_COLS);
    enter_math(MathBuiltinId::SUB_BCAST_COLS, MathInitFunc::SUB_BCAST_COLS);
    enter_math(MathBuiltinId::MUL_BCAST_COLS, MathInitFunc::MUL_BCAST_COLS);
    enter_math(MathBuiltinId::ADD_BCAST_SCALAR, MathInitFunc::ADD_BCAST_SCALAR);
    enter_math(MathBuiltinId::SUB_BCAST_SCALAR, MathInitFunc::SUB_BCAST_SCALAR);
    enter_math(MathBuiltinId::MUL_BCAST_SCALAR, MathInitFunc::MUL_BCAST_SCALAR);
    enter_math(MathBuiltinId::MATMUL, MathInitFunc::MATMUL);
    enter_math(MathBuiltinId::REDUCE_MAX_ROWS, MathInitFunc::REDUCE_MAX_ROWS);
    enter_math(MathBuiltinId::REDUCE_MAX_COLS, MathInitFunc::REDUCE_MAX_COLS);
    enter_math(MathBuiltinId::REDUCE_MAX_SCALAR, MathInitFunc::REDUCE_MAX_SCALAR);
    enter_math(MathBuiltinId::REDUCE_SUM_ROWS, MathInitFunc::REDUCE_SUM_ROWS);
    enter_math(MathBuiltinId::REDUCE_SUM_COLS, MathInitFunc::REDUCE_SUM_COLS);
    enter_math(MathBuiltinId::REDUCE_SUM_SCALAR, MathInitFunc::REDUCE_SUM_SCALAR);
    enter_math(MathBuiltinId::TRANSPOSE, MathInitFunc::TRANSPOSE);
    enter_math(MathBuiltinId::TILIZE_BLOCK, MathInitFunc::COPY);
    enter_math(MathBuiltinId::UNTILIZE_BLOCK, MathInitFunc::COPY);
    // sfpu
    enter_sfpu(MathBuiltinId::COPY_DST, MathInitFunc::COPY_DST);
    enter_sfpu(MathBuiltinId::ADD_DST, MathInitFunc::ADD_DST);
    enter_sfpu(MathBuiltinId::SUB_DST, MathInitFunc::SUB_DST);
    enter_sfpu(MathBuiltinId::RSUB_DST, MathInitFunc::RSUB_DST);
    enter_sfpu(MathBuiltinId::MUL_DST, MathInitFunc::MUL_DST);
    enter_sfpu(MathBuiltinId::DIV_DST, MathInitFunc::DIV_DST);
    enter_sfpu(MathBuiltinId::POWER_DST, MathInitFunc::POWER_DST);
    enter_sfpu(MathBuiltinId::ABS, MathInitFunc::ABS);
    enter_sfpu(MathBuiltinId::ACOS, MathInitFunc::ACOS);
    enter_sfpu(MathBuiltinId::ADD_SCALAR, MathInitFunc::BINARY_SCALAR);
    enter_sfpu(MathBuiltinId::ASIN, MathInitFunc::ASIN);
    enter_sfpu(MathBuiltinId::ATAN, MathInitFunc::ATAN);
    enter_sfpu(MathBuiltinId::CAST_BF16_U16, MathInitFunc::CAST);
    enter_sfpu(MathBuiltinId::CAST_U16_BF16, MathInitFunc::CAST);
    enter_sfpu(MathBuiltinId::CEIL, MathInitFunc::CEIL);
    enter_sfpu(MathBuiltinId::COS, MathInitFunc::COS);
    enter_sfpu(MathBuiltinId::DIV_SCALAR, MathInitFunc::BINARY_SCALAR);
    enter_sfpu(MathBuiltinId::ELU, MathInitFunc::ELU);
    enter_sfpu(MathBuiltinId::EQZ, MathInitFunc::EQZ);
    enter_sfpu(MathBuiltinId::ERF, MathInitFunc::ERF);
    enter_sfpu(MathBuiltinId::ERFC, MathInitFunc::ERFC);
    enter_sfpu(MathBuiltinId::ERFINV, MathInitFunc::ERFINV);
    enter_sfpu(MathBuiltinId::EXP, MathInitFunc::EXP);
    enter_sfpu(MathBuiltinId::EXP2, MathInitFunc::EXP2);
    enter_sfpu(MathBuiltinId::EXPM1, MathInitFunc::EXPM1);
    enter_sfpu(MathBuiltinId::FILL, MathInitFunc::FILL);
    enter_sfpu(MathBuiltinId::FLOOR, MathInitFunc::FLOOR);
    enter_sfpu(MathBuiltinId::GELU, MathInitFunc::GELU);
    enter_sfpu(MathBuiltinId::GEZ, MathInitFunc::GEZ);
    enter_sfpu(MathBuiltinId::GTZ, MathInitFunc::GTZ);
    enter_sfpu(MathBuiltinId::HEAVISIDE, MathInitFunc::HEAVISIDE);
    enter_sfpu(MathBuiltinId::I0, MathInitFunc::I0);
    enter_sfpu(MathBuiltinId::ISFINITE, MathInitFunc::ISFINITE);
    enter_sfpu(MathBuiltinId::ISINF, MathInitFunc::ISINF);
    enter_sfpu(MathBuiltinId::ISNAN, MathInitFunc::ISNAN);
    enter_sfpu(MathBuiltinId::ISNEGINF, MathInitFunc::ISNEGINF);
    enter_sfpu(MathBuiltinId::ISPOSINF, MathInitFunc::ISPOSINF);
    enter_sfpu(MathBuiltinId::LEAKY_RELU, MathInitFunc::LEAKY_RELU);
    enter_sfpu(MathBuiltinId::LEZ, MathInitFunc::LEZ);
    enter_sfpu(MathBuiltinId::LOG, MathInitFunc::LOG);
    enter_sfpu(MathBuiltinId::LOG_WITH_BASE, MathInitFunc::LOG_WITH_BASE);
    enter_sfpu(MathBuiltinId::LOGICAL_NOT, MathInitFunc::LOGICAL_NOT);
    enter_sfpu(MathBuiltinId::LTZ, MathInitFunc::LTZ);
    enter_sfpu(MathBuiltinId::MAX, MathInitFunc::MAX);
    enter_sfpu(MathBuiltinId::MUL_SCALAR, MathInitFunc::BINARY_SCALAR);
    enter_sfpu(MathBuiltinId::NEZ, MathInitFunc::NEZ);
    enter_sfpu(MathBuiltinId::POWER, MathInitFunc::POWER); 
    enter_sfpu(MathBuiltinId::RECIP, MathInitFunc::RECIP);
    enter_sfpu(MathBuiltinId::RELU, MathInitFunc::RELU);
    enter_sfpu(MathBuiltinId::RELU_MAX, MathInitFunc::RELU_MAX);
    enter_sfpu(MathBuiltinId::RELU_MIN, MathInitFunc::RELU_MIN);
    enter_sfpu(MathBuiltinId::RSQRT, MathInitFunc::RSQRT);
    enter_sfpu(MathBuiltinId::RSUB_SCALAR, MathInitFunc::BINARY_SCALAR);
    enter_sfpu(MathBuiltinId::SIGMOID, MathInitFunc::SIGMOID);
    enter_sfpu(MathBuiltinId::SIGN, MathInitFunc::SIGN);
    enter_sfpu(MathBuiltinId::SIGNBIT, MathInitFunc::SIGNBIT);
    enter_sfpu(MathBuiltinId::SIN, MathInitFunc::SIN);
    enter_sfpu(MathBuiltinId::SQRT, MathInitFunc::SQRT);
    enter_sfpu(MathBuiltinId::SQUARE, MathInitFunc::SQUARE);
    enter_sfpu(MathBuiltinId::SUB_SCALAR, MathInitFunc::BINARY_SCALAR);
    enter_sfpu(MathBuiltinId::TAN, MathInitFunc::TAN);
    enter_sfpu(MathBuiltinId::TANH, MathInitFunc::TANH);
    // pack
    enter_pack(MathBuiltinId::PACK, MathInitFunc::PACK);
    enter_pack(MathBuiltinId::PACK_ROW, MathInitFunc::PACK_ROW);
    enter_pack(MathBuiltinId::PACK_COL, MathInitFunc::PACK_COL);
    enter_pack(MathBuiltinId::PACK_SCALAR, MathInitFunc::PACK_SCALAR);
    enter_pack(MathBuiltinId::TILIZE_BLOCK, MathInitFunc::PACK);
    enter_pack(MathBuiltinId::UNTILIZE_BLOCK, MathInitFunc::PACK);
}

void MathInitFuncHandler::enter(MathBuiltinId id, int group, MathInitFunc func) {
    int index = int(id);
    assert(index >= 0 && index < MathBuiltinIdCount);
    assert(group >= 0 && group < MathInitFuncGroup::COUNT);
    m_map[index][group] = func;
}

//
//    Public functions
//

namespace {

std::unordered_map<MathInitFunc, std::string> g_math_init_func_name_map = {
    // special
    {MathInitFunc::NONE, "[none]"},
    {MathInitFunc::UNDEF, "[undef]"},
    // unpack
    {MathInitFunc::UNPACK_BINARY, "unpack_binary"},
    {MathInitFunc::UNPACK_BCAST_ROWS, "unpack_bcast_rows"},
    {MathInitFunc::UNPACK_BCAST_COLS, "unpack_bcast_cols"},
    {MathInitFunc::UNPACK_BCAST_SCALAR, "unpack_bcast_scalar"},
    {MathInitFunc::UNPACK_MATMUL, "unpack_matmul"},
    {MathInitFunc::UNPACK_UNARY, "unpack_unary"},
    {MathInitFunc::UNPACK_REDUCE_ROWS, "unpack_reduce_rows"},
    {MathInitFunc::UNPACK_REDUCE_COLS, "unpack_reduce_cols"},
    {MathInitFunc::UNPACK_REDUCE_SCALAR, "unpack_reduce_scalar"},
    {MathInitFunc::UNPACK_TRANSPOSE, "unpack_transpose"},
    {MathInitFunc::UNPACK_TILIZE_BLOCK, "unpack_tilize_block"},
    {MathInitFunc::UNPACK_UNTILIZE_BLOCK, "unpack_untilize_block"},
    // pack
    {MathInitFunc::PACK, "pack"},
    {MathInitFunc::PACK_ROW, "pack_row"},
    {MathInitFunc::PACK_COL, "pack_col"},
    {MathInitFunc::PACK_SCALAR, "pack_scalar"},
    // math
    {MathInitFunc::COPY, "copy"},
    {MathInitFunc::ADD, "add"},
    {MathInitFunc::SUB, "sub"},
    {MathInitFunc::MUL, "mul"},
    {MathInitFunc::ADD_BCAST_ROWS, "add_bcast_rows"},
    {MathInitFunc::SUB_BCAST_ROWS, "sub_bcast_rows"},
    {MathInitFunc::MUL_BCAST_ROWS, "mul_bcast_rows"},
    {MathInitFunc::ADD_BCAST_COLS, "add_bcast_cols"},
    {MathInitFunc::SUB_BCAST_COLS, "sub_bcast_cols"},
    {MathInitFunc::MUL_BCAST_COLS, "mul_bcast_cols"},
    {MathInitFunc::ADD_BCAST_SCALAR, "add_bcast_scalar"},
    {MathInitFunc::SUB_BCAST_SCALAR, "sub_bcast_scalar"},
    {MathInitFunc::MUL_BCAST_SCALAR, "mul_bcast_scalar"},
    {MathInitFunc::MATMUL, "matmul"},
    {MathInitFunc::REDUCE_MAX_ROWS, "reduce_max_rows"},
    {MathInitFunc::REDUCE_MAX_COLS, "reduce_max_cols"},
    {MathInitFunc::REDUCE_MAX_SCALAR, "reduce_max_scalar"},
    {MathInitFunc::REDUCE_SUM_ROWS, "reduce_sum_rows"},
    {MathInitFunc::REDUCE_SUM_COLS, "reduce_sum_cols"},
    {MathInitFunc::REDUCE_SUM_SCALAR, "reduce_sum_scalar"},
    {MathInitFunc::TRANSPOSE, "transpose"},
    {MathInitFunc::COPY_DST, "copy_dst"},
    {MathInitFunc::ADD_DST, "add_dst"},
    {MathInitFunc::SUB_DST, "sub_dst"},
    {MathInitFunc::RSUB_DST, "rsub_dst"},
    {MathInitFunc::MUL_DST, "mul_dst"},
    {MathInitFunc::DIV_DST, "div_dst"},
    {MathInitFunc::POWER_DST, "power_dst"},
    {MathInitFunc::ABS, "abs"},
    {MathInitFunc::ACOS, "acos"},
    {MathInitFunc::ASIN, "asin"},
    {MathInitFunc::ATAN, "atan"},
    {MathInitFunc::BINARY_SCALAR, "binary_scalar"},
    {MathInitFunc::CAST, "cast"},
    {MathInitFunc::CEIL, "ceil"},
    {MathInitFunc::COS, "cos"},
    {MathInitFunc::ELU, "elu"},
    {MathInitFunc::EQZ, "eqz"},
    {MathInitFunc::ERF, "erf"},
    {MathInitFunc::ERFC, "erfc"},
    {MathInitFunc::ERFINV, "erfinv"},
    {MathInitFunc::EXP, "exp"},
    {MathInitFunc::EXP2, "exp2"},
    {MathInitFunc::EXPM1, "expm1"},
    {MathInitFunc::FILL, "fill"},
    {MathInitFunc::FLOOR, "floor"},
    {MathInitFunc::GELU, "gelu"},
    {MathInitFunc::GEZ, "gez"},
    {MathInitFunc::GTZ, "gtz"},
    {MathInitFunc::HEAVISIDE, "heaviside"},
    {MathInitFunc::I0, "i0"},
    {MathInitFunc::ISFINITE, "isfinite"},
    {MathInitFunc::ISINF, "isinf"},
    {MathInitFunc::ISNAN, "isnan"},
    {MathInitFunc::ISNEGINF, "isneginf"},
    {MathInitFunc::ISPOSINF, "isposinf"},
    {MathInitFunc::LEAKY_RELU, "leaky_relu"},
    {MathInitFunc::LEZ, "lez"},
    {MathInitFunc::LOG, "log"},
    {MathInitFunc::LOG_WITH_BASE, "log_with_base"},
    {MathInitFunc::LOGICAL_NOT, "logical_not"},
    {MathInitFunc::LTZ, "ltz"},
    {MathInitFunc::MAX, "max"},
    {MathInitFunc::NEZ, "nez"},
    {MathInitFunc::POWER, "power"},
    {MathInitFunc::RECIP, "recip"},
    {MathInitFunc::RELU, "relu"},
    {MathInitFunc::RELU_MAX, "relu_max"},
    {MathInitFunc::RELU_MIN, "relu_min"},
    {MathInitFunc::RSQRT, "rsqrt"},
    {MathInitFunc::SIGMOID, "sigmoid"},
    {MathInitFunc::SIGN, "sign"},
    {MathInitFunc::SIGNBIT, "signbit"},
    {MathInitFunc::SIN, "sin"},
    {MathInitFunc::SQRT, "sqrt"},
    {MathInitFunc::SQUARE, "square"},
    {MathInitFunc::TAN, "tan"},
    {MathInitFunc::TANH, "tanh"}
};

} // namespace

std::string get_math_init_func_name(MathInitFunc func) {
    auto it = g_math_init_func_name_map.find(func);
    return (it != g_math_init_func_name_map.end()) ? it->second : "[?]";
}

} // namespace front
} // namespace tanto
} // namespace ronin

