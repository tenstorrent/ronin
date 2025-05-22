#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 $TANTO/dw_batch_bias_reader.cpp >$METAL/dw_batch_bias_reader.cpp
$FRONT --mode=read -DT=bfloat16 $TANTO/dw_batch_lx_bias_reader.cpp >$METAL/dw_batch_lx_bias_reader.cpp

$FRONT --mode=write -DT=bfloat16 $TANTO/dw_batch_writer.cpp >$METAL/dw_batch_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/dw_batch_mcast_writer.cpp >$METAL/dw_batch_mcast_writer.cpp

$FRONT --mode=compute -DT=bfloat16 $TANTO/dw_batch_bias_math.cpp >$METAL/dw_batch_bias_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/dw_batch_bias_unary_math.cpp >$METAL/dw_batch_bias_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/dw_batch_bias_unary_math.cpp >$METAL/dw_batch_bias_relu6_math.cpp

$FRONT --mode=compute -DT=bfloat16 $TANTO/dw_batch_rm_bias_math.cpp >$METAL/dw_batch_rm_bias_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/dw_batch_rm_bias_unary_math.cpp >$METAL/dw_batch_rm_bias_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/dw_batch_rm_bias_unary_math.cpp >$METAL/dw_batch_rm_bias_relu6_math.cpp

$FRONT --mode=read -DT=bfloat16 $TANTO/dw_spatial_bias_reader.cpp >$METAL/dw_spatial_bias_reader.cpp
$FRONT --mode=read -DT=bfloat16 $TANTO/dw_spatial_lx_bias_reader.cpp >$METAL/dw_spatial_lx_bias_reader.cpp

$FRONT --mode=write -DT=bfloat16 $TANTO/dw_spatial_writer.cpp >$METAL/dw_spatial_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/dw_spatial_mcast_writer.cpp >$METAL/dw_spatial_mcast_writer.cpp

$FRONT --mode=read -DT=bfloat16 $TANTO/dsc_batch_reader.cpp >$METAL/dsc_batch_reader.cpp
$FRONT --mode=read -DT=bfloat16 $TANTO/dsc_batch_lx_reader.cpp >$METAL/dsc_batch_lx_reader.cpp

$FRONT --mode=write -DT=bfloat16 $TANTO/dsc_batch_writer.cpp >$METAL/dsc_batch_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/dsc_batch_add_writer.cpp >$METAL/dsc_batch_add_writer.cpp

$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/dsc_batch_math.cpp >$METAL/dsc_batch_relu_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/dsc_batch_math.cpp >$METAL/dsc_batch_relu6_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=0 $TANTO/dsc_batch_add_math.cpp >$METAL/dsc_batch_relu_add_math.cpp
$FRONT --mode=compute -DT=bfloat16 -P0=1 $TANTO/dsc_batch_add_math.cpp >$METAL/dsc_batch_relu6_add_math.cpp


