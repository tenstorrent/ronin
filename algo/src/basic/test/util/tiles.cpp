// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <vector>

#include "test/util/tiles.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace util {

namespace {

void tile_to_faces(float *dst, const float *src) {
    for (int fh = 0; fh < 2; fh++) {
        for (int ih = 0; ih < 16; ih++) {
            for (int fw = 0; fw < 2; fw++) {
                for (int iw = 0; iw < 16; iw++) {
                    int isrc = fh * 512 + ih * 32 + fw * 16 + iw;
                    int idst = fh * 512 + fw * 256 + ih * 16 + iw; 
                    dst[idst] = src[isrc];
                }
            }
        }
    }
}

void faces_to_tile(float *dst, const float *src) {
    for (int fh = 0; fh < 2; fh++) {
        for (int ih = 0; ih < 16; ih++) {
            for (int fw = 0; fw < 2; fw++) {
                for (int iw = 0; iw < 16; iw++) {
                    int isrc = fh * 512 + fw * 256 + ih * 16 + iw; 
                    int idst = fh * 512 + ih * 32 + fw * 16 + iw;
                    dst[idst] = src[isrc];
                }
            }
        }
    }
}

void tilize_block(float *dst, const float *src, int tiles) {
    int dst_pos = 0;
    for (int t = 0; t < tiles; t++) {
        for (int i = 0; i < 32; i++) {
            for (int k = 0; k < 32; k++) {
                int src_pos = (i * tiles + t) * 32 + k;
                dst[dst_pos] = src[src_pos];
                dst_pos++;
            }
        }
    }
}

void untilize_block(float *dst, const float *src, int tiles) {
    int src_pos = 0;
    for (int t = 0; t < tiles; t++) {
        for (int i = 0; i < 32; i++) {
            for (int k = 0; k < 32; k++) {
                int dst_pos = (i * tiles + t) * 32 + k;
                dst[dst_pos] = src[src_pos];
                src_pos++;
            }
        }
    }
}

} // namespace

void tilize(const float *px, float *py, int H, int W) {
    assert(W % 32 == 0);
    assert(H % 32 == 0);
    int block = W / 32;
    std::vector<float> t(W * 32);
    float *pt = t.data();
    int ix = 0;
    int iy = 0;
    for (int h = 0; h < H; h += 32) {
        tilize_block(pt, px + ix, block);
        ix += block * 1024;
        int it = 0;
        for (int w = 0; w < W; w += 32) {
            tile_to_faces(py + iy, pt + it);
            iy += 1024;
            it += 1024;
        }
    }
}

void untilize(const float *px, float *py, int H, int W) {
    assert(W % 32 == 0);
    assert(H % 32 == 0);
    int block = W / 32;
    std::vector<float> t(W * 32);
    float *pt = t.data();
    int ix = 0;
    int iy = 0;
    for (int h = 0; h < H; h += 32) {
        int it = 0;
        for (int w = 0; w < W; w += 32) {
            faces_to_tile(pt + it, px + ix);
            ix += 1024;
            it += 1024;
        }
        untilize_block(py + iy, pt, block);
        iy += block * 1024;
    }
}

std::vector<float> tilize(const std::vector<float> &x, int block) {
    int size = int(x.size());
    int W = block * 32;
    assert(size % W == 0);
    int H = size / W;
    assert(H % 32 == 0);
    std::vector<float> y(size);
    std::vector<float> t(W * 32);
    const float *px = x.data();
    float *py = y.data();
    float *pt = t.data();
    int ix = 0;
    int iy = 0;
    for (int h = 0; h < H; h += 32) {
        tilize_block(pt, px + ix, block);
        ix += block * 1024;
        int it = 0;
        for (int w = 0; w < W; w += 32) {
            tile_to_faces(py + iy, pt + it);
            iy += 1024;
            it += 1024;
        }
    }
    return y;
}

std::vector<float> untilize(const std::vector<float> &x, int block) {
    int size = int(x.size());
    int W = block * 32;
    assert(size % W == 0);
    int H = size / W;
    assert(H % 32 == 0);
    std::vector<float> y(size);
    std::vector<float> t(W * 32);
    const float *px = x.data();
    float *py = y.data();
    float *pt = t.data();
    int ix = 0;
    int iy = 0;
    for (int h = 0; h < H; h += 32) {
        int it = 0;
        for (int w = 0; w < W; w += 32) {
            faces_to_tile(pt + it, px + ix);
            ix += 1024;
            it += 1024;
        }
        untilize_block(py + iy, pt, block);
        iy += block * 1024;
    }
    return y;
}

} // namespace util
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

