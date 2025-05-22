// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void clear_lx(
        local<T> lx, 
        local<T>lzero, 
        uint32 H, 
        uint32 WC, 
        uint32 before_h, 
        uint32 after_h, 
        uint32 before_wc, 
        uint32 after_wc) {
    uint32 WC_full = before_wc + WC + after_wc;
    uint32 pos = 0;
    for (uint32 i = 0; i < before_h; i++) {
        lx.read(pos, lzero, 0, WC_full);
        pos += WC_full;
    }
    for (uint32 i = 0; i < H; i++) {
        if (before_wc != 0) {
            lx.read(pos, lzero, 0, before_wc);
        }
        pos += before_wc + WC;
        if (after_wc != 0) {
            lx.read(pos, lzero, 0, after_wc);
        }
        pos += after_wc;
    }
    for (uint32 i = 0; i < after_h; i++) {
        lx.read(pos, lzero, 0, WC_full);
        pos += WC_full;
    }
    read_barrier();
}

void load_lx(
        global<T> gx,
        local<T> lx,
        uint32 H,
        uint32 WC,
        uint32 before_wc,
        uint32 after_wc,
        uint32 before_hwc,
        uint32 x_start) {
    uint32 pos = before_hwc + before_wc;
    uint32 stride = WC + after_wc + before_wc;
    for (uint32 i = 0; i < H; i++) {
        lx.read(pos, gx, x_start, WC);
        x_start += WC;
        pos += stride;
    }
    read_barrier();
}

void read_px(
        local <T> lx,
        pipe<T> px,
        uint32 R,
        uint32 SC,
        uint32 RSC_rnd,
        uint32 offset_wc,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 end_q,
        uint32 &p_term,
        uint32 &q_term) {
    uint32 dst_start = 0;
    px.reserve_back();
    px.move_init(SC);
    for (uint32 i = 0; i < 32; i++) {
        uint32 dst_pos = dst_start;
        uint32 r_term = 0;
        for (uint32 r = 0; r < R; r++) {
            uint32 src_pos = p_term + q_term + r_term + offset_wc;
            px.move(dst_pos, lx, src_pos);
            dst_pos += SC;
            r_term += delta_r;
        }
        dst_start += RSC_rnd;
        q_term += delta_q;
        if (q_term >= end_q) {
            q_term = 0;
            p_term += delta_p;
        }
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        local<T> lx,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 H,
        uint32 K,
        uint32 R,
        uint32 WC,
        uint32 PQ,
        uint32 SC,
        uint32 RSC_rnd,
        uint32 before_h,
        uint32 after_h,
        uint32 before_wc,
        uint32 after_wc,
        uint32 offset_wc,
        uint32 before_hwc,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 end_q,
        uint32 x_pos,
        uint32 x_stride,
        uint32 zero_size) {
    lzero.read(0, gzero, 0, zero_size);
    // read_barrier is below
    px.set_frame(RSC_rnd / 32);
    pb.set_frame(K / 32);
    pb.reserve_back();
    pb.read(0, gb, 0, K * 32);
    read_barrier();
    pb.push_back();
    clear_lx(
        lx, 
        lzero, 
        H, 
        WC, 
        before_h, 
        after_h, 
        before_wc, 
        after_wc);
    uint32 x_start = x_pos;
    uint32 p_term = 0;
    uint32 q_term = 0;
    for (uint32 n = 0; n < N; n++) {
        load_lx(
            gx, 
            lx, 
            H, 
            WC, 
            before_wc, 
            after_wc, 
            before_hwc, 
            x_start);
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            read_px(
                lx, 
                px, 
                R, 
                SC, 
                RSC_rnd, 
                offset_wc,
                delta_p, 
                delta_q, 
                delta_r, 
                end_q, 
                p_term, 
                q_term);
        } // pq_start
        x_start += x_stride;
    } // n
}

