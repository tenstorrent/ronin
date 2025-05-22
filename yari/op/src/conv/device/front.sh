#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 $TANTO/basic_batch_bias_reader.cpp >$METAL/basic_batch_bias_reader.cpp
$FRONT --mode=read -DT=bfloat16 $TANTO/basic_batch_lx_bias_reader.cpp >$METAL/basic_batch_lx_bias_reader.cpp
$FRONT --mode=read -DT=bfloat16 $TANTO/basic_batch_pw_bias_reader.cpp >$METAL/basic_batch_pw_bias_reader.cpp

$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_writer.cpp >$METAL/basic_batch_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_mcast_writer.cpp >$METAL/basic_batch_mcast_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_lw_writer.cpp >$METAL/basic_batch_lw_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_lw_mcast_writer.cpp >$METAL/basic_batch_lw_mcast_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_add_writer.cpp >$METAL/basic_batch_add_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_mcast_add_writer.cpp >$METAL/basic_batch_mcast_add_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_lw_add_writer.cpp >$METAL/basic_batch_lw_add_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_batch_lw_mcast_add_writer.cpp >$METAL/basic_batch_lw_mcast_add_writer.cpp

$FRONT --mode=compute -DT=bfloat16 $TANTO/basic_batch_bias_math.cpp >$METAL/basic_batch_bias_math.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/basic_batch_bias_add_math.cpp >$METAL/basic_batch_bias_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/basic_batch_bias_unary_math.cpp >$METAL/basic_batch_bias_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/basic_batch_bias_unary_math.cpp >$METAL/basic_batch_bias_relu6_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/basic_batch_bias_add_unary_math.cpp >$METAL/basic_batch_bias_add_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/basic_batch_bias_add_unary_math.cpp >$METAL/basic_batch_bias_add_relu6_math.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/basic_batch_lw_bias_math.cpp >$METAL/basic_batch_lw_bias_math.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/basic_batch_lw_bias_add_math.cpp >$METAL/basic_batch_lw_bias_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/basic_batch_lw_bias_unary_math.cpp >$METAL/basic_batch_lw_bias_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/basic_batch_lw_bias_unary_math.cpp >$METAL/basic_batch_lw_bias_relu6_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/basic_batch_lw_bias_add_unary_math.cpp >$METAL/basic_batch_lw_bias_add_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/basic_batch_lw_bias_add_unary_math.cpp >$METAL/basic_batch_lw_bias_add_relu6_math.cpp

$FRONT --mode=read -DT=bfloat16 $TANTO/basic_spatial_bias_reader.cpp >$METAL/basic_spatial_bias_reader.cpp
$FRONT --mode=read -DT=bfloat16 $TANTO/basic_spatial_lx_bias_reader.cpp >$METAL/basic_spatial_lx_bias_reader.cpp
$FRONT --mode=read -DT=bfloat16 $TANTO/basic_spatial_pw_bias_reader.cpp >$METAL/basic_spatial_pw_bias_reader.cpp

$FRONT --mode=write -DT=bfloat16 $TANTO/basic_spatial_mcast_writer.cpp >$METAL/basic_spatial_mcast_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_spatial_lw_mcast_writer.cpp >$METAL/basic_spatial_lw_mcast_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_spatial_mcast_add_writer.cpp >$METAL/basic_spatial_mcast_add_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_spatial_lw_mcast_add_writer.cpp >$METAL/basic_spatial_lw_mcast_add_writer.cpp

$FRONT --mode=read -DT=bfloat16 $TANTO/basic_split_bias_reader.cpp >$METAL/basic_split_bias_reader.cpp

$FRONT --mode=write -DT=bfloat16 $TANTO/basic_split_writer.cpp >$METAL/basic_split_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/basic_split_add_writer.cpp >$METAL/basic_split_add_writer.cpp

$FRONT --mode=compute -DT=bfloat16 $TANTO/basic_split_bias_math.cpp >$METAL/basic_split_bias_math.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/basic_split_bias_add_math.cpp >$METAL/basic_split_bias_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/basic_split_bias_unary_math.cpp >$METAL/basic_split_bias_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/basic_split_bias_unary_math.cpp >$METAL/basic_split_bias_relu6_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/basic_split_bias_add_unary_math.cpp >$METAL/basic_split_bias_add_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/basic_split_bias_add_unary_math.cpp >$METAL/basic_split_bias_add_relu6_math.cpp

$FRONT --mode=read -DT=bfloat16 $TANTO/image_batch_bias_reader.cpp >$METAL/image_batch_bias_reader.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/image_batch_writer.cpp >$METAL/image_batch_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/image_batch_mcast_writer.cpp >$METAL/image_batch_mcast_writer.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/image_batch_bias_math.cpp >$METAL/image_batch_bias_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/image_batch_bias_unary_math.cpp >$METAL/image_batch_bias_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/image_batch_bias_unary_math.cpp >$METAL/image_batch_bias_relu6_math.cpp


