// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>

#include "host/util/transform.hpp"

namespace ronin {
namespace op {
namespace common {
namespace util {

namespace {

int round_up(int a, int b) {
    return ((a + b - 1) / b) * b;
}

void copy(float *dst, const float *src, int count) {
    memcpy(dst, src, count * sizeof(float));
}

} // namespace

//
//    float32 <-> bfloat16 conversions
//

namespace {

union U32 {
    float f;
    uint32_t i;
};

} // namespace

std::vector<uint16_t> float_to_u16b(const std::vector<float> &x) {
    U32 u32;
    size_t n = x.size();
    std::vector<uint16_t> y(n);
    for (size_t i = 0; i < n; i++) {
        u32.f = x[i];
        y[i] = uint16_t(u32.i >> 16);
    }
    return y;
}

std::vector<float> u16b_to_float(const std::vector<uint16_t> &x) {
    U32 u32;
    size_t n = x.size();
    std::vector<float> y(n);
    for (size_t i = 0; i < n; i++) {
        u32.i = uint32_t(x[i]) << 16;
        y[i] = u32.f;
    }
    return y;
}

//
//    Tilize / untilize
//

namespace {

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

void tile_to_faces(const float *src, float *dst) {
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

void faces_to_tile(const float *src, float *dst) {
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

} // namespace

std::vector<float> tilize(const std::vector<float> &x, int H, int W) {
    assert(x.size() == H * W);
    assert(H % 32 == 0);
    assert(W % 32 == 0);
    std::vector<float> y(x.size());
    int pos = 0;
    for (int h = 0; h < H; h += 32) {
        tilize_block(y.data() + pos, x.data() + pos, W / 32);
        pos += W * 32;
    }
    return y;
}

std::vector<float> untilize(const std::vector<float> &x, int H, int W) {
    assert(x.size() == H * W);
    assert(H % 32 == 0);
    assert(W % 32 == 0);
    std::vector<float> y(x.size());
    int pos = 0;
    for (int h = 0; h < H; h += 32) {
        untilize_block(y.data() + pos, x.data() + pos, W / 32);
        pos += W * 32;
    }
    return y;
}

std::vector<float> make_faces(const std::vector<float> &x) {
    size_t size = x.size();
    assert(size % 1024 == 0);
    std::vector<float> y(size);
    const float *px = x.data();
    float *py = y.data();
    for (size_t i = 0; i < size; i += 1024) {
        tile_to_faces(px + i, py + i);
    }
    return y;
}

std::vector<float> make_tiles(const std::vector<float> &x) {
    size_t size = x.size();
    assert(size % 1024 == 0);
    std::vector<float> y(size);
    const float *px = x.data();
    float *py = y.data();
    for (size_t i = 0; i < size; i += 1024) {
        faces_to_tile(px + i, py + i);
    }
    return y;
}

//
//    Pad / unpad HW
//

std::vector<float> pad_hw(
        const std::vector<float> &x, 
        int N, 
        int H, 
        int W,
        int C) {
    int HW = H * W;
    int HW_adj = round_up(HW, 32);
    if (HW_adj == HW) {
        return x;
    }
    int xstride = HW * C;
    int ystride = HW_adj * C;
    std::vector<float> y(N * ystride, 0.0f);
    const float *xptr = x.data();
    float *yptr = y.data();
    for (int n = 0; n < N; n++) {
        copy(yptr, xptr, xstride); 
        xptr += xstride;
        yptr += ystride;
    }
    return y;
}

std::vector<float> unpad_hw(
        const std::vector<float> &x, 
        int N, 
        int H, 
        int W,
        int C) {
    int HW = H * W;
    int HW_adj = round_up(HW, 32);
    if (HW_adj == HW) {
        return x;
    }
    int xstride = HW_adj * C;
    int ystride = HW * C;
    std::vector<float> y(N * ystride);
    const float *xptr = x.data();
    float *yptr = y.data();
    for (int n = 0; n < N; n++) {
        copy(yptr, xptr, ystride); 
        xptr += xstride;
        yptr += ystride;
    }
    return y;
}

std::vector<float> pad(const std::vector<float> &x, int nx0, int ny0) {
    assert(nx0 <= ny0);
    if (nx0 == ny0) {
        return x;
    }
    std::vector<float> y(ny0, 0.0f);
    copy(y.data(), x.data(), nx0);
    return y;
}

std::vector<float> pad(
        const std::vector<float> &x,
        int nx0,
        int nx1,
        int ny0,
        int ny1) {
    assert(nx0 <= ny0 && nx1 <= ny1);
    if (nx0 == ny0 && nx1 == ny1) {
        return x;
    }
    std::vector<float> y(ny0 * ny1, 0.0f);
    const float *xptr = x.data();
    float *yptr = y.data();
    int sy0 = ny1;
    for (int ix0 = 0; ix0 < nx0; ix0++) { 
        copy(yptr + ix0 * sy0, xptr, nx1);
        xptr += nx1;
    }
    return y;
}

std::vector<float> pad(
        const std::vector<float> &x,
        int nx0,
        int nx1,
        int nx2,
        int ny0,
        int ny1,
        int ny2) {
    assert(nx0 <= ny0 && nx1 <= ny1 && nx2 <= ny2);
    if (nx0 == ny0 && nx1 == ny1 && nx2 == ny2) {
        return x;
    }
    std::vector<float> y(ny0 * ny1 * ny2, 0.0f);
    const float *xptr = x.data();
    float *yptr = y.data();
    int sy1 = ny2;
    int sy0 = ny1 * sy1;
    for (int ix0 = 0; ix0 < nx0; ix0++) { 
        for (int ix1 = 0; ix1 < nx1; ix1++) { 
            copy(yptr + ix0 * sy0 + ix1 * sy1, xptr, nx2);
            xptr += nx2;
        }
    }
    return y;
}

std::vector<float> pad(
        const std::vector<float> &x,
        int nx0,
        int nx1,
        int nx2,
        int nx3,
        int ny0,
        int ny1,
        int ny2,
        int ny3) {
    assert(nx0 <= ny0 && nx1 <= ny1 && nx2 <= ny2 && nx3 <= ny3);
    if (nx0 == ny0 && nx1 == ny1 && nx2 == ny2 && nx3 == ny3) {
        return x;
    }
    std::vector<float> y(ny0 * ny1 * ny2 * ny3, 0.0f);
    const float *xptr = x.data();
    float *yptr = y.data();
    int sy2 = ny3;
    int sy1 = ny2 * sy2;
    int sy0 = ny1 * sy1;
    for (int ix0 = 0; ix0 < nx0; ix0++) { 
        for (int ix1 = 0; ix1 < nx1; ix1++) { 
            for (int ix2 = 0; ix2 < nx2; ix2++) {
                copy(yptr + ix0 * sy0 + ix1 * sy1 + ix2 * sy2, xptr, nx3);
                xptr += nx3;
            }
        }
    }
    return y;
}

std::vector<float> unpad(const std::vector<float> &x, int nx0, int ny0) {
    assert(ny0 <= nx0);
    if (nx0 == ny0) {
        return x;
    }
    std::vector<float> y(ny0);
    copy(y.data(), x.data(), ny0);
    return y;
}

std::vector<float> unpad(
        const std::vector<float> &x,
        int nx0,
        int nx1,
        int ny0,
        int ny1) {
    assert(ny0 <= nx0 && ny1 <= nx1);
    if (nx0 == ny0 && nx1 == ny1) {
        return x;
    }
    std::vector<float> y(ny0 * ny1);
    const float *xptr = x.data();
    float *yptr = y.data();
    int sx0 = nx1;
    for (int iy0 = 0; iy0 < ny0; iy0++) { 
        copy(yptr, xptr + iy0 * sx0, ny1);
        yptr += ny1;
    }
    return y;
}

std::vector<float> unpad(
        const std::vector<float> &x,
        int nx0,
        int nx1,
        int nx2,
        int ny0,
        int ny1,
        int ny2) {
    assert(ny0 <= nx0 && ny1 <= nx1 && ny2 <= nx2);
    if (nx0 == ny0 && nx1 == ny1 && nx2 == ny2) {
        return x;
    }
    std::vector<float> y(ny0 * ny1 * ny2);
    const float *xptr = x.data();
    float *yptr = y.data();
    int sx1 = nx2;
    int sx0 = nx1 * sx1;
    for (int iy0 = 0; iy0 < ny0; iy0++) { 
        for (int iy1 = 0; iy1 < ny1; iy1++) { 
            copy(yptr, xptr + iy0 * sx0 + iy1 * sx1, ny2);
            yptr += ny2;
        }
    }
    return y;
}

std::vector<float> unpad(
        const std::vector<float> &x,
        int nx0,
        int nx1,
        int nx2,
        int nx3,
        int ny0,
        int ny1,
        int ny2,
        int ny3) {
    assert(ny0 <= nx0 && ny1 <= nx1 && ny2 <= nx2 && ny3 <= nx3);
    if (nx0 == ny0 && nx1 == ny1 && nx2 == ny2 && nx3 == ny3) {
        return x;
    }
    std::vector<float> y(ny0 * ny1 * ny2 * ny3);
    const float *xptr = x.data();
    float *yptr = y.data();
    int sx2 = nx3;
    int sx1 = nx2 * sx2;
    int sx0 = nx1 * sx1;
    for (int iy0 = 0; iy0 < ny0; iy0++) { 
        for (int iy1 = 0; iy1 < ny1; iy1++) { 
            for (int iy2 = 0; iy2 < ny2; iy2++) {
                copy(yptr, xptr + iy0 * sx0 + iy1 * sx1 + iy2 * sx2, ny3);
                yptr += ny3;
            }
        }
    }
    return y;
}

} // namespace util
} // namespace common
} // namespace op
} // namespace ronin

