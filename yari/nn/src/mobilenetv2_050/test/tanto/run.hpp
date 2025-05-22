// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "test/util/net.hpp"

using namespace ronin::nn::common::test;

void run_global(const util::NetCmdArgs &args);
void run_global_dsc(const util::NetCmdArgs &args);

