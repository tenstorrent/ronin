// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "host/util/diag.hpp"

namespace ronin {
namespace op {
namespace common {
namespace util {

std::string diag_data_stats(const std::vector<float> &x) {
    int volume = int(x.size());
    if (volume == 0) {
        return "volume 0";
    }
    double xsum = 0.0;
    float xmin = x[0];
    float xmax = x[0];
    int imin = 0;
    int imax = 0;
    for (int i = 0; i < volume; i++) {
        float v = x[i];
        xsum += double(v);
        if (v < xmin) {
            xmin = v;
            imin = i;
        }
        if (v > xmax) {
            xmax = v;
            imax = i;
        }
    }
    return "volume " + std::to_string(volume) +
        " sum " + std::to_string(xsum) + 
        " min " + std::to_string(xmin) + " [" + std::to_string(imin) + "]" +
        " max " + std::to_string(xmax) + " [" + std::to_string(imax) + "]";
}

} // namespace util
} // namespace common
} // namespace op
} // namespace ronin

