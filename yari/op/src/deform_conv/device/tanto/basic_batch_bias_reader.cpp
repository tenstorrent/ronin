// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void make_pc1_im(
        pipe<T> ppc_im,
        pipe<T> pc1_im,
        uint32 rs2) {
    pc1_im.reserve_back();
    // lh and lw occupy consecutive rows
    // hh and hw occupy consecutive rows
    // (lh, lw) and (hh, hw) occupy consecutive tiles
    uint32 p = rs2 / 32;
    uint32 q = rs2 % 32;
    uint32 lh = (p * 64 + q) * 32;
    uint32 lw = lh + 32;
    uint32 hh = lh + 1024;
    uint32 hw = hh + 32;
    // assemble first row of untilized [32, 32 * 4] frame in pc1_im
    // other rows are not significant
    // order is [hh, lh, hw, lw]
    pc1_im.move_init(32);
    pc1_im.move(0, ppc_im, hh);
    pc1_im.move(32, ppc_im, lh);
    pc1_im.move(32 * 2, ppc_im, hw);
    pc1_im.move(32 * 3, ppc_im, lw);
    read_barrier();
    pc1_im.push_back();
}

void read_px(
        global<T> gx,
        local<T> lzero,
        pipe<T> px,
        pipe<uint16> ppi_im,
        uint32 C,
        uint32 H,
        uint32 W,
        uint32 D,
        uint32 rs2,
        uint32 th,
        uint32 tw,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 end_q,
        uint32 x_start,
        uint32 &p_term,
        uint32 &q_term,
        uint32 r_term, 
        uint32 s_term) {
    uint32 dst_pos = 0;
    px.reserve_back();
    for (uint32 i = 0; i < 32; i++) {
        uint32 ih = uint32(ppi_im.get(rs2));
        uint32 iw = uint32(ppi_im.get(rs2 + 1));
#if 1
        // enable for Metal / disable for Jitte (temporary patch)
        ih = ((0x80 | (ih & 0x7f)) << ((ih >> 7) - 127)) >> 7;
        iw = ((0x80 | (iw & 0x7f)) << ((iw >> 7) - 127)) >> 7;
#endif
        ih -= 63;
        iw -= 63;
        uint32 h = p_term + r_term + ih + th;
        uint32 w = q_term + s_term + iw + tw;
        if (h >= H || w >= W) {
            px.read(dst_pos, lzero, 0, C);
        } else {
            // ACHTUNG: int multiplications here
            uint32 src_pos = x_start + (h * W + w) * C;
            px.read(dst_pos, gx, src_pos, C);
        }
        q_term += delta_q;
        if (q_term >= end_q) {
            q_term = start_q;
            p_term += delta_p;
        }
        dst_pos += C;
        rs2 += D;
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gd,
        global<T> gb,
        global<T> gzero,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pd,
        pipe<T> pb,
        pipe<uint16> ppi_im,
        pipe<T> ppc_im,
        pipe<T> pc1_im,
        uint32 N,
        uint32 H,
        uint32 W,
        uint32 C,
        uint32 K,
        uint32 R,
        uint32 S,
        uint32 D,
        uint32 PQ,
        uint32 start_p,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 delta_r,
        uint32 delta_s,
        uint32 end_q,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride,
        uint32 d_pos,
        uint32 d_stride) {
    lzero.read(0, gzero, 0, zero_size);
    // read_barrier is below
    px.set_frame(C / 32);
    pd.set_frame(D / 32);
    pb.set_frame(K / 32);
    ppi_im.set_frame(D / 32);
    ppc_im.set_frame((D / 32) * 2);
    pc1_im.set_frame(4);
    pb.reserve_back();
    pb.read(0, gb, 0, K * 32);
    read_barrier();
    pb.push_back();
    uint32 x_start = x_pos;
    uint32 d_start = d_pos;
    for (uint32 n = 0; n < N; n++) {
        uint32 d_curr = d_start;
        uint32 p_start = start_p;
        uint32 q_start = start_q;
        uint32 p_term = 0;
        uint32 q_term = 0;
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            pd.reserve_back();
            pd.read(0, gd, d_curr, D * 32);
            read_barrier();
            pd.push_back();
            // math kernel transforms pd -> (ppi_im, ppc_im)
            ppi_im.wait_front();
            ppc_im.wait_front();
            uint32 rs2 = 0;
            uint32 r_term = 0;
            for (uint32 r = 0; r < R; r++) {
                uint32 s_term = 0;
                for (uint32 s = 0; s < S; s++) {
                    make_pc1_im(ppc_im, pc1_im, rs2);
                    for (uint32 i = 0; i < 4; i++) {
                        p_term = p_start;
                        q_term = q_start;
                        uint32 th = i >> 1;
                        uint32 tw = i & 1;
                        read_px(
                            gx, 
                            lzero,
                            px,
                            ppi_im,
                            C,
                            H,
                            W,
                            D,
                            rs2,
                            th,
                            tw,
                            start_q,
                            delta_p,
                            delta_q,
                            end_q,
                            x_start, 
                            p_term, 
                            q_term, 
                            r_term, 
                            s_term);
                    } // i
                    s_term += delta_s;
                    rs2 += 2;
                } // s
                r_term += delta_r;
            } // r
            ppi_im.pop_front();
            ppc_im.pop_front();
            p_start = p_term;
            q_start = q_term;
            d_curr += D * 32;
        } // pq_start
        x_start += x_stride;
        d_start += d_stride;
    } // n
}

