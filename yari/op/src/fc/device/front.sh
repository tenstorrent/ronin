#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 $TANTO/fc_batch_bias_reader.cpp >$METAL/fc_batch_bias_reader.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/fc_batch_writer.cpp >$METAL/fc_batch_writer.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/fc_batch_mcast_writer.cpp >$METAL/fc_batch_mcast_writer.cpp
$FRONT --mode=compute -DT=bfloat16 $TANTO/fc_batch_bias_math.cpp >$METAL/fc_batch_bias_math.cpp


