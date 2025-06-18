#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

./build_eltwise_binary.sh
./build_eltwise_sfpu.sh
./build_matmul_multi_core.sh
./build_matmul_single_core.sh


