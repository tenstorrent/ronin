#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

FRONT=../../../../../tanto/bin/front/tanto

TANTO=./tanto
METAL=./metal

mkdir -p $METAL

$FRONT --mode=read -DT=bfloat16 $TANTO/load_dist_reader.cpp >$METAL/load_dist_reader.cpp
$FRONT --mode=write -DT=bfloat16 $TANTO/store_dist_writer.cpp >$METAL/store_dist_writer.cpp


