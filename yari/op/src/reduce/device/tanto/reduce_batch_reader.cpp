// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void read_px(
        global<T> gx,
        local<T> lzero,
        pipe<T> px,
        uint32 H,
        uint32 W,
        uint32 x_start,
        uint32 h_start) {
    uint32 src_pos = x_start;
    uint32 dst_pos = 0;
    px.reserve_back();
    for (uint32 i = 0; i < 32; i++) {
        if (h_start + i >= H) {
            px.read(dst_pos, lzero, 0, W);
        } else {
            px.read(dst_pos, gx, src_pos, W);
        }
        src_pos += W;
        dst_pos += W;
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gs,
        global<T> gzero,
        local<T> lzero,
        pipe<T> px,
        pipe<T> ps,
        uint32 N,
        uint32 H,
        uint32 W,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride) {
    lzero.read(0, gzero, 0, zero_size);
    // read_barrier is below
    px.set_frame(W / 32);
    ps.set_frame(1);
    ps.reserve_back();
    ps.read(0, gs, 0, 1024);
    read_barrier();
    ps.push_back();
    uint32 x_start = x_pos;
    for (uint32 n = 0; n < N; n++) {
        for (uint32 h_start = 0; h_start < H; h_start += 32) {
              read_px(
                  gx,
                  lzero,
                  px,
                  H,
                  W,
                  x_start,
                  h_start);
            x_start += W * 32;
        }
        x_start += x_stride;
    }
}

