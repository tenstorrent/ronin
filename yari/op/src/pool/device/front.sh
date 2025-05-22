#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 $TANTO/pool2d_batch_reader.cpp >$METAL/pool2d_batch_reader.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/pool2d_batch_writer.cpp >$METAL/pool2d_batch_writer.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/pool2d_batch_max_math.cpp >$METAL/pool2d_batch_max_math.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/pool2d_batch_avg_math.cpp >$METAL/pool2d_batch_avg_math.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/pool2d_batch_1x1_math.cpp >$METAL/pool2d_batch_1x1_math.cpp


