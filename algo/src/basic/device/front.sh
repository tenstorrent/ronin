#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/eltwise_binary_reader.cpp >$METAL/eltwise_binary_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/eltwise_binary_writer.cpp >$METAL/eltwise_binary_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 \
    $TANTO/eltwise_binary_math.cpp >$METAL/eltwise_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 \
    $TANTO/eltwise_binary_math.cpp >$METAL/eltwise_sub_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=2 \
    $TANTO/eltwise_binary_math.cpp >$METAL/eltwise_mul_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/eltwise_sfpu_reader.cpp >$METAL/eltwise_sfpu_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/eltwise_sfpu_writer.cpp >$METAL/eltwise_sfpu_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=10 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_exp_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=13 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_gelu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=25 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_log_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=31 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_recip_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=32 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=36 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_sigmoid_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=40 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_sqrt_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=43 \
    $TANTO/eltwise_sfpu_math.cpp >$METAL/eltwise_tanh_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/bcast_rows_reader.cpp >$METAL/bcast_rows_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/bcast_rows_writer.cpp >$METAL/bcast_rows_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 \
    $TANTO/bcast_rows_math.cpp >$METAL/bcast_rows_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 \
    $TANTO/bcast_rows_math.cpp >$METAL/bcast_rows_sub_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=2 \
    $TANTO/bcast_rows_math.cpp >$METAL/bcast_rows_mul_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/bcast_cols_reader.cpp >$METAL/bcast_cols_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/bcast_cols_writer.cpp >$METAL/bcast_cols_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 \
    $TANTO/bcast_cols_math.cpp >$METAL/bcast_cols_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 \
    $TANTO/bcast_cols_math.cpp >$METAL/bcast_cols_sub_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=2 \
    $TANTO/bcast_cols_math.cpp >$METAL/bcast_cols_mul_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/bcast_scalar_reader.cpp >$METAL/bcast_scalar_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/bcast_scalar_writer.cpp >$METAL/bcast_scalar_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 \
    $TANTO/bcast_scalar_math.cpp >$METAL/bcast_scalar_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 \
    $TANTO/bcast_scalar_math.cpp >$METAL/bcast_scalar_sub_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=2 \
    $TANTO/bcast_scalar_math.cpp >$METAL/bcast_scalar_mul_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/matmul_multi_reader.cpp >$METAL//matmul_multi_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/matmul_multi_writer.cpp >$METAL//matmul_multi_writer.cpp
$FRONT --mode=compute -DT=bfloat16 \
    $TANTO/matmul_multi_math.cpp >$METAL//matmul_multi_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/matmul_single_reader.cpp >$METAL//matmul_single_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/matmul_single_writer.cpp >$METAL//matmul_single_writer.cpp
$FRONT --mode=compute -DT=bfloat16 \
    $TANTO/matmul_single_math.cpp >$METAL//matmul_single_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/reduce_rows_reader.cpp >$METAL/reduce_rows_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/reduce_rows_writer.cpp >$METAL/reduce_rows_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 \
    $TANTO/reduce_rows_math.cpp >$METAL/reduce_rows_max_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 \
    $TANTO/reduce_rows_math.cpp >$METAL/reduce_rows_sum_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/reduce_cols_reader.cpp >$METAL/reduce_cols_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/reduce_cols_writer.cpp >$METAL/reduce_cols_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 \
    $TANTO/reduce_cols_math.cpp >$METAL/reduce_cols_max_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 \
    $TANTO/reduce_cols_math.cpp >$METAL/reduce_cols_sum_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/reduce_scalar_reader.cpp >$METAL/reduce_scalar_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/reduce_scalar_writer.cpp >$METAL/reduce_scalar_writer.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 \
    $TANTO/reduce_scalar_math.cpp >$METAL/reduce_scalar_max_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 \
    $TANTO/reduce_scalar_math.cpp >$METAL/reduce_scalar_sum_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/transpose_wh_reader.cpp >$METAL/transpose_wh_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/transpose_wh_writer.cpp >$METAL/transpose_wh_writer.cpp
$FRONT --mode=compute -DT=bfloat16 \
    $TANTO/transpose_wh_math.cpp >$METAL/transpose_wh_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/unpack_tilize_reader.cpp >$METAL/unpack_tilize_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/unpack_tilize_writer.cpp >$METAL/unpack_tilize_writer.cpp
$FRONT --mode=compute -DT=bfloat16 \
    $TANTO/unpack_tilize_math.cpp >$METAL/unpack_tilize_math.cpp

$FRONT --mode=read -DT=bfloat16 \
    $TANTO/unpack_untilize_reader.cpp >$METAL/unpack_untilize_reader.cpp
$FRONT --mode=write -DT=bfloat16 \
    $TANTO/unpack_untilize_writer.cpp >$METAL/unpack_untilize_writer.cpp
$FRONT --mode=compute -DT=bfloat16 \
    $TANTO/unpack_untilize_math.cpp >$METAL/unpack_untilize_math.cpp


