// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void read_px(
        global<T> gx,
        local<T> lzero,
        pipe<T> px,
        uint32 H,
        uint32 C,
        uint32 x_start,
        uint32 h_start) {
    uint32 src_pos = x_start;
    uint32 dst_pos = 0;
    px.reserve_back();
    for (uint32 i = 0; i < 32; i++) {
        if (h_start + i >= H) {
            px.read(dst_pos, lzero, 0, C);
        } else {
            px.read(dst_pos, gx, src_pos, C);
        }
        src_pos += C;
        dst_pos += C;
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 H,
        uint32 C,
        uint32 K,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride) {
    lzero.read(0, gzero, 0, zero_size);
    // read_barrier is below
    px.set_frame(C / 32);
    pb.set_frame(K / 32);
    pb.reserve_back();
    pb.read(0, gb, 0, K * 32);
    read_barrier();
    pb.push_back();
    uint32 x_batch = x_pos;
    for (uint32 n = 0; n < N; n++) {
        uint32 x_start = x_batch;
        for (uint32 h_start = 0; h_start < H; h_start += 32) {
            read_px(
                gx, 
                lzero,
                px,
                H,
                C,
                x_start,
                h_start);
            x_start += C * 32;
        } // hw_start
        x_batch += x_stride;
    } // n
}

