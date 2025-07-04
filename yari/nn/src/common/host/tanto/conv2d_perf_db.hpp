// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "host/tanto/layer_base.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace tanto {

enum class Conv2dAlgo {
    NONE,
    BASIC_BATCH,
    BASIC_SPLIT_8,
    BASIC_SPLIT_16,
    BASIC_SPATIAL
};

struct Conv2dPerfEntry {
    int N;
    int H;
    int W;
    int C;
    int P;
    int Q;
    int K;
    int R;
    int S;
    bool fuse_add;
    Conv2dAlgo algo;
};

class Conv2dPerfDb {
public:
    Conv2dPerfDb();
    ~Conv2dPerfDb();
public:
    bool select_algo(
        int N,
        const Conv2dParam &param,
        bool fuse_add,
        Conv2dAlgo &algo,
        int &batch_size);
private:
    void init();
    void add_entry(const Conv2dPerfEntry &entry);
    void finalize();
    static int infer_batch_size(int N, Conv2dAlgo algo);
private:
    struct PerfRange {
        int N;
        int start;
        int end;
    };
private:
    std::vector<Conv2dPerfEntry> m_entries;
    std::vector<PerfRange> m_ranges;
};

} // namespace tanto
} // namespace common
} // namespace nn
} // namespace ronin

