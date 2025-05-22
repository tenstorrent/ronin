// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void read_px(
        local<T> lx,
        local<T> lzero,
        pipe<T> px,
        uint32 C,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 end_q,
        uint32 &p_term,
        uint32 &q_term,
        uint32 r_term, 
        uint32 s_term,
        uint32 mask) {
    uint32 dst_pos = 0;
    px.reserve_back();
    px.move_init(C);
    for (uint32 i = 0; i < 32; i++) {
        if ((mask & 1) == 0) {
            px.move(dst_pos, lzero, 0);
        } else {
            uint32 src_pos = p_term + q_term + r_term + s_term;
            px.move(dst_pos, lx, src_pos);
        }
        q_term += delta_q;
        if (q_term >= end_q) {
            q_term = start_q;
            p_term += delta_p;
        }
        mask >>= 1;
        dst_pos += C;
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gb2,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lx,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        pipe<T> pb2,
        uint32 N,
        uint32 C,
        uint32 K,
        uint32 R,
        uint32 S,
        uint32 HWC,
        uint32 PQ,
        uint32 start_p,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 delta_s,
        uint32 end_q,
        uint32 zero_size,
        uint32 mask_size,
        uint32 x_pos,
        uint32 x_stride) {
    lzero.read(0, gzero, 0, zero_size);
    lmask.read(0, gmask, 0, mask_size);
    // read_barrier is below
    px.set_frame(C / 32);
    pb.set_frame(C / 32);
    pb2.set_frame(K / 32);
    pb.reserve_back();
    pb2.reserve_back();
    pb.read(0, gb, 0, C * 32);
    pb2.read(0, gb2, 0, K * 32);
    read_barrier();
    pb.push_back();
    pb2.push_back();
    uint32 x_start = x_pos;
    for (uint32 n = 0; n < N; n++) {
        lx.read(0, gx, x_start, HWC);
        read_barrier();
        uint32 p_start = start_p;
        uint32 q_start = start_q;
        uint32 p_term = 0;
        uint32 q_term = 0;
        uint32 mask_pos = 0;
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            uint32 r_term = 0;
            for (uint32 r = 0; r < R; r++) {
                uint32 s_term = 0;
                for (uint32 s = 0; s < S; s++) {
                    p_term = p_start;
                    q_term = q_start;
                    uint32 mask = lmask.get(mask_pos);
                    read_px(
                        lx, 
                        lzero,
                        px,
                        C,
                        start_q,
                        delta_p,
                        delta_q,
                        end_q,
                        p_term, 
                        q_term, 
                        r_term, 
                        s_term, 
                        mask);
                    mask_pos++;
                    s_term += delta_s;
                } // s
                r_term += delta_r;
            } // r
            p_start = p_term;
            q_start = q_term;
        } // pq_start
        x_start += x_stride;
    } // n
}

