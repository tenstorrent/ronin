// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum class Algo {
    BASIC_BATCH,
    DW_BATCH,
    DW_SPATIAL,
    DSC_BATCH
};

void run_group(Algo algo, int N, int batch_size, int repeat);
void run_dsc(Algo algo, int N, int batch_size, int repeat);

