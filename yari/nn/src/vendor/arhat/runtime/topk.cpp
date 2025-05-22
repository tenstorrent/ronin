//
// Copyright (c) 2019-2025 FRAGATA COMPUTER SYSTEMS AG
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

#include "runtime/arhat.hpp"

namespace arhat {

void TopK(int count, const float *data, int k, int *pos, float *val) {
    for (int i = 0; i < k; i++) {
        pos[i] = -1;
        val[i] = 0.0f;
    }
    for (int p = 0; p < count; p++) {
        float v = data[p];
        int j = -1;
        for (int i = 0; i < k; i++) {
            if (pos[i] < 0 || val[i] < v) {
                j = i;
                break;
            }
        }
        if (j >= 0) {
            for (int i = k - 1; i > j; i--) {
                pos[i] = pos[i-1];
                val[i] = val[i-1];
            }
            pos[j] = p;
            val[j] = v;
        }
    }
}

} // arhat

