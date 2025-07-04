// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <algorithm>

#include "host/tanto/layer_base.hpp"
#include "host/tanto/conv2d_perf_db.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace tanto {

namespace {

constexpr Conv2dAlgo
    A = Conv2dAlgo::BASIC_BATCH,
    B = Conv2dAlgo::BASIC_SPLIT_16,
    C = Conv2dAlgo::BASIC_SPLIT_8,
    D = Conv2dAlgo::BASIC_SPATIAL;

const Conv2dPerfEntry g_entries[] = {
    // ResNet50

    // N = 16, fuse_add = false
    {16, 56, 56, 64, 56, 56, 64, 1, 1, false, D},       // kernel03
    {16, 56, 56, 64, 56, 56, 64, 3, 3, false, D},       // kernel04
    {16, 56, 56, 64, 56, 56, 256, 1, 1, false, D},      // kernel05
    {16, 56, 56, 256, 56, 56, 64, 1, 1, false, D},      // kernel07
    {16, 56, 56, 64, 28, 28, 64, 3, 3, false, D},       // kernel09
    {16, 28, 28, 64, 28, 28, 256, 1, 1, false, D},      // kernel10
    {16, 28, 28, 256, 28, 28, 128, 1, 1, false, D},     // kernel13
    {16, 28, 28, 128, 28, 28, 128, 3, 3, false, D},     // kernel14
    {16, 28, 28, 128, 28, 28, 512, 1, 1, false, D},     // kernel15
    {16, 28, 28, 256, 28, 28, 512, 1, 1, false, D},     // kernel16
    {16, 28, 28, 512, 28, 28, 128, 1, 1, false, D},     // kernel17
    {16, 28, 28, 128, 14, 14, 128, 3, 3, false, D},     // kernel19
    {16, 14, 14, 128, 14, 14, 512, 1, 1, false, A},     // kernel20
    {16, 14, 14, 512, 14, 14, 256, 1, 1, false, D},     // kernel23
    {16, 14, 14, 256, 14, 14, 256, 3, 3, false, D},     // kernel24
    {16, 14, 14, 256, 14, 14, 1024, 1, 1, false, C},    // kernel25
    {16, 14, 14, 512, 14, 14, 1024, 1, 1, false, C},    // kernel26
    {16, 14, 14, 1024, 14, 14, 256, 1, 1, false, D},    // kernel27
    {16, 14, 14, 256, 7, 7, 256, 3, 3, false, B},       // kernel29
    {16, 7, 7, 256, 7, 7, 1024, 1, 1, false, C},        // kernel30
    {16, 7, 7, 1024, 7, 7, 512, 1, 1, false, C},        // kernel33
    {16, 7, 7, 512, 7, 7, 512, 3, 3, false, C},         // kernel34
    {16, 7, 7, 512, 7, 7, 2048, 1, 1, false, C},        // kernel35
    {16, 7, 7, 1024, 7, 7, 2048, 1, 1, false, C},       // kernel36
    {16, 7, 7, 2048, 7, 7, 512, 1, 1, false, C},        // kernel37
    {16, 7, 7, 512, 7, 7, 2048, 1, 1, false, C},        // kernel38

    // N = 16, fuse_add = true
    {16, 56, 56, 64, 56, 56, 64, 1, 1, true, D},       // kernel03
    {16, 56, 56, 64, 56, 56, 64, 3, 3, true, D},       // kernel04
    {16, 56, 56, 64, 56, 56, 256, 1, 1, true, D},      // kernel05
    {16, 56, 56, 256, 56, 56, 64, 1, 1, true, D},      // kernel07
    {16, 56, 56, 64, 28, 28, 64, 3, 3, true, D},       // kernel09
    {16, 28, 28, 64, 28, 28, 256, 1, 1, true, D},      // kernel10
    {16, 28, 28, 256, 28, 28, 128, 1, 1, true, D},     // kernel13
    {16, 28, 28, 128, 28, 28, 128, 3, 3, true, D},     // kernel14
    {16, 28, 28, 128, 28, 28, 512, 1, 1, true, D},     // kernel15
    {16, 28, 28, 256, 28, 28, 512, 1, 1, true, D},     // kernel16
    {16, 28, 28, 512, 28, 28, 128, 1, 1, true, D},     // kernel17
    {16, 28, 28, 128, 14, 14, 128, 3, 3, true, D},     // kernel19
    {16, 14, 14, 128, 14, 14, 512, 1, 1, true, D},     // kernel20
    {16, 14, 14, 512, 14, 14, 256, 1, 1, true, D},     // kernel23
    {16, 14, 14, 256, 14, 14, 256, 3, 3, true, B},     // kernel24
    {16, 14, 14, 256, 14, 14, 1024, 1, 1, true, D},    // kernel25
    {16, 14, 14, 512, 14, 14, 1024, 1, 1, true, B},    // kernel26
    {16, 14, 14, 1024, 14, 14, 256, 1, 1, true, D},    // kernel27
    {16, 14, 14, 256, 7, 7, 256, 3, 3, true, B},       // kernel29
    {16, 7, 7, 256, 7, 7, 1024, 1, 1, true, B},        // kernel30
    {16, 7, 7, 1024, 7, 7, 512, 1, 1, true, C},        // kernel33
    {16, 7, 7, 512, 7, 7, 512, 3, 3, true, C},         // kernel34
    {16, 7, 7, 512, 7, 7, 2048, 1, 1, true, C},        // kernel35
    {16, 7, 7, 1024, 7, 7, 2048, 1, 1, true, C},       // kernel36
    {16, 7, 7, 2048, 7, 7, 512, 1, 1, true, C},        // kernel37
    {16, 7, 7, 512, 7, 7, 2048, 1, 1, true, C},        // kernel38

    // N = 64, fuse_add = false
    {64, 56, 56, 64, 56, 56, 64, 1, 1, false, D},       // kernel03
    {64, 56, 56, 64, 56, 56, 64, 3, 3, false, D},       // kernel04
    {64, 56, 56, 64, 56, 56, 256, 1, 1, false, D},      // kernel05
    {64, 56, 56, 256, 56, 56, 64, 1, 1, false, D},      // kernel07
    {64, 56, 56, 64, 28, 28, 64, 3, 3, false, D},       // kernel09
    {64, 28, 28, 64, 28, 28, 256, 1, 1, false, D},      // kernel10
    {64, 28, 28, 256, 28, 28, 128, 1, 1, false, A},     // kernel13
    {64, 28, 28, 128, 28, 28, 128, 3, 3, false, D},     // kernel14
    {64, 28, 28, 128, 28, 28, 512, 1, 1, false, D},     // kernel15
    {64, 28, 28, 256, 28, 28, 512, 1, 1, false, D},     // kernel16
    {64, 28, 28, 512, 28, 28, 128, 1, 1, false, D},     // kernel17
    {64, 28, 28, 128, 14, 14, 128, 3, 3, false, D},     // kernel19
    {64, 14, 14, 128, 14, 14, 512, 1, 1, false, B},     // kernel20
    {64, 14, 14, 512, 14, 14, 256, 1, 1, false, D},     // kernel23
    {64, 14, 14, 256, 14, 14, 256, 3, 3, false, A},     // kernel24
    {64, 14, 14, 256, 14, 14, 1024, 1, 1, false, C},    // kernel25
    {64, 14, 14, 512, 14, 14, 1024, 1, 1, false, C},    // kernel26
    {64, 14, 14, 1024, 14, 14, 256, 1, 1, false, D},    // kernel27
    {64, 14, 14, 256, 7, 7, 256, 3, 3, false, B},       // kernel29
    {64, 7, 7, 256, 7, 7, 1024, 1, 1, false, C},        // kernel30
    {64, 7, 7, 1024, 7, 7, 512, 1, 1, false, B},        // kernel33
    {64, 7, 7, 512, 7, 7, 512, 3, 3, false, A},         // kernel34
    {64, 7, 7, 512, 7, 7, 2048, 1, 1, false, C},        // kernel35
    {64, 7, 7, 1024, 7, 7, 2048, 1, 1, false, C},       // kernel36
    {64, 7, 7, 2048, 7, 7, 512, 1, 1, false, C},        // kernel37
    {64, 7, 7, 512, 7, 7, 2048, 1, 1, false, C},        // kernel38

    // N = 64, fuse_add = true
    {64, 56, 56, 64, 56, 56, 64, 1, 1, true, D},       // kernel03
    {64, 56, 56, 64, 56, 56, 64, 3, 3, true, D},       // kernel04
    {64, 56, 56, 64, 56, 56, 256, 1, 1, true, A},      // kernel05
    {64, 56, 56, 256, 56, 56, 64, 1, 1, true, D},      // kernel07
    {64, 56, 56, 64, 28, 28, 64, 3, 3, true, D},       // kernel09
    {64, 28, 28, 64, 28, 28, 256, 1, 1, true, A},      // kernel10
    {64, 28, 28, 256, 28, 28, 128, 1, 1, true, A},     // kernel13
    {64, 28, 28, 128, 28, 28, 128, 3, 3, true, D},     // kernel14
    {64, 28, 28, 128, 28, 28, 512, 1, 1, true, D},     // kernel15
    {64, 28, 28, 256, 28, 28, 512, 1, 1, true, A},     // kernel16
    {64, 28, 28, 512, 28, 28, 128, 1, 1, true, A},     // kernel17
    {64, 28, 28, 128, 14, 14, 128, 3, 3, true, D},     // kernel19
    {64, 14, 14, 128, 14, 14, 512, 1, 1, true, D},     // kernel20
    {64, 14, 14, 512, 14, 14, 256, 1, 1, true, D},     // kernel23
    {64, 14, 14, 256, 14, 14, 256, 3, 3, true, A},     // kernel24
    {64, 14, 14, 256, 14, 14, 1024, 1, 1, true, A},    // kernel25
    {64, 14, 14, 512, 14, 14, 1024, 1, 1, true, B},    // kernel26
    {64, 14, 14, 1024, 14, 14, 256, 1, 1, true, D},    // kernel27
    {64, 14, 14, 256, 7, 7, 256, 3, 3, true, B},       // kernel29
    {64, 7, 7, 256, 7, 7, 1024, 1, 1, true, D},        // kernel30
    {64, 7, 7, 1024, 7, 7, 512, 1, 1, true, B},        // kernel33
    {64, 7, 7, 512, 7, 7, 512, 3, 3, true, A},         // kernel34
    {64, 7, 7, 512, 7, 7, 2048, 1, 1, true, B},        // kernel35
    {64, 7, 7, 1024, 7, 7, 2048, 1, 1, true, C},       // kernel36
    {64, 7, 7, 2048, 7, 7, 512, 1, 1, true, B},        // kernel37
    {64, 7, 7, 512, 7, 7, 2048, 1, 1, true, B},        // kernel38

    // end of table
    {0}
};

} // namespace

//
//    Conv2dPerfDb
//

Conv2dPerfDb::Conv2dPerfDb() {
    init();
}

Conv2dPerfDb::~Conv2dPerfDb() { }

bool Conv2dPerfDb::select_algo(
        int N,
        const Conv2dParam &param,
        bool fuse_add,
        Conv2dAlgo &algo,
        int &batch_size) {
//printf("@@@ Conv2dPerfDb::select_algo N %d H %d W %d C %d P %d Q %d K %d R %d S %d add %d\n",
//N, param.H, param.W, param.C, param.P, param.Q, param.K, param.R, param.S, int(fuse_add));
    algo = Conv2dAlgo::NONE;
    batch_size = 0;
    int num_ranges = int(m_ranges.size());
    int start = -1;
    int end = -1;
    for (int i = num_ranges - 1; i >= 0; i--) {
        PerfRange &range = m_ranges[i];
        if (range.N <= N) {
            start = range.start;
            end = range.end;
            break;
        }
    }
    if (start < 0) {
        return false;
    }
    for (int i = start; i < end; i++) {
        Conv2dPerfEntry &entry = m_entries[i];
        if (entry.H == param.H &&
                entry.W == param.W &&
                entry.C == param.C &&
                entry.P == param.P &&
                entry.Q == param.Q &&
                entry.K == param.K &&
                entry.R == param.R &&
                entry.S == param.S &&
                entry.fuse_add == fuse_add) {
            algo = entry.algo;
            break;
        }
    }
    if (algo == Conv2dAlgo::NONE) {
        return false;
    }
    batch_size = infer_batch_size(N, algo);
//printf("@@@   => algo %d batch_size %d\n", int(algo), batch_size);
    return true;
}

void Conv2dPerfDb::init() {
    for (int i = 0; g_entries[i].N != 0; i++) {
        add_entry(g_entries[i]);
    }
    finalize();
}

void Conv2dPerfDb::add_entry(const Conv2dPerfEntry &entry) {
    m_entries.push_back(entry);
}

void Conv2dPerfDb::finalize() {
    std::sort(
        m_entries.begin(),
        m_entries.end(), 
        [](const Conv2dPerfEntry &a, const Conv2dPerfEntry &b) -> bool {
            return (a.N < b.N);
        });
    int num_entries = int(m_entries.size());
    int N = 0;
    int start = 0;
    for (int i = 0; i < num_entries; i++) {
        int curr_N = m_entries[i].N;
        if (curr_N != N) {
            if (N > 0) {
                m_ranges.push_back({N, start, i});
            }
            N = curr_N;
            start = i;
        }
    }
    if (N > 0) {
        m_ranges.push_back({N, start, num_entries});
    }
}

int Conv2dPerfDb::infer_batch_size(int N, Conv2dAlgo algo) {
    // ACHTUNG: Temporarily Wormhole-specific
    int batch_size = 0;
    if (algo == Conv2dAlgo::BASIC_SPLIT_8) {
        batch_size = 8;
    } else if (algo == Conv2dAlgo::BASIC_SPLIT_16) {
        batch_size = 16;
    } else if (algo == Conv2dAlgo::BASIC_SPATIAL) {
        batch_size = 16;
    } else {
        if (N >= 64) {
            batch_size = 64;
        } else if (N >= 32) {
            batch_size = 32;
        } else if (N >= 16) {
            batch_size = 16;
        } else if (N >= 8) {
            batch_size = 8;
        } else {
            batch_size = N;
        }
    }
    return batch_size;
}

} // namespace tanto
} // namespace common
} // namespace nn
} // namespace ronin

