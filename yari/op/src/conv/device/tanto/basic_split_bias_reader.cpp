// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void read_px(
        global<T> gx,
        local<T> lzero,
        pipe<T> px,
        uint32 C,
        uint32 start_q,
        uint32 delta_p,
        uint32 delta_q,
        uint32 end_q,
        uint32 start,
        uint32 &p_term,
        uint32 &q_term,
        uint32 r_term, 
        uint32 s_term,
        uint32 mask) {
    uint32 dst_pos = 0;
    for (uint32 i = 0; i < 32; i++) {
        if ((mask & 1) == 0) {
            px.read(dst_pos, lzero, 0, C);
        } else {
            uint32 src_pos = start + p_term + q_term + r_term + s_term;
            px.read(dst_pos, gx, src_pos, C);
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
}

void kernel_send(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 C,
        uint32 R,
        uint32 S,
        uint32 PQ,
        uint32 Kb,
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
        uint32 x_stride,
        uint32 b_pos) {
    sem_recv.set(1);
    lzero.read(0, gzero, 0, zero_size);
    lmask.read(0, gmask, 0, mask_size);
    // read_barrier is below
    px.set_frame(C / 32);
    pb.set_frame(Kb / 32);
    pb.reserve_back();
    pb.read(0, gb, b_pos, Kb * 32);
    read_barrier();
    pb.push_back();
    uint32 x_start = x_pos;
    for (uint32 n = 0; n < N; n++) {
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
                    px.reserve_back();
                    read_px(
                        gx, 
                        lzero,
                        px,
                        C,
                        start_q,
                        delta_p,
                        delta_q,
                        end_q,
                        x_start, 
                        p_term, 
                        q_term, 
                        r_term, 
                        s_term, 
                        mask);
                    mask_pos++;
                    s_term += delta_s;
                    sem_send.wait(num_dests);
                    sem_send.set(0);
                    // Note inverted x/y order (required for readers)
                    px.write_mcast(
                        0,
                        px,
                        0,
                        C * 32,
                        x1,
                        y1,
                        x0,
                        y0,
                        num_dests);
                    sem_recv.set_mcast(
                        sem_recv, 
                        x1, 
                        y1,
                        x0,
                        y0,
                        num_dests);
                    px.push_back();
                } // s
                r_term += delta_r;
            } // r
            p_start = p_term;
            q_start = q_term;
        } // pq_start
        x_start += x_stride;
    } // n
}

void kernel_recv(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 C,
        uint32 R,
        uint32 S,
        uint32 PQ,
        uint32 Kb,
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
        uint32 x_stride,
        uint32 b_pos) {
    px.set_frame(C / 32);
    pb.set_frame(Kb / 32);
    pb.reserve_back();
    pb.read(0, gb, b_pos, Kb * 32);
    read_barrier();
    pb.push_back();
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            for (uint32 r = 0; r < R; r++) {
                for (uint32 s = 0; s < S; s++) {
                    px.reserve_back();
                    sem_recv.set(0);
                    sem_send.inc(x0, y0, 1);
                    sem_recv.wait(1);
                    px.push_back();
                } // s
            } // r
        } // pq_start
    } // n
}

void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 C,
        uint32 R,
        uint32 S,
        uint32 PQ,
        uint32 Kb,
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
        uint32 x_stride,
        uint32 b_pos) {
    if (send_mode != 0) {
        kernel_send(
            gx,
            gb,
            gzero,
            gmask,
            lzero,
            lmask,
            px,
            pb,
            sem_send,
            sem_recv,
            send_mode,
            x0,
            y0,
            x1,
            y1,
            num_dests,
            N,
            C,
            R,
            S,
            PQ,
            Kb,
            start_p,
            start_q,
            delta_p,
            delta_q,
            delta_r,
            delta_s,
            end_q,
            zero_size,
            mask_size,
            x_pos,
            x_stride,
            b_pos);
    } else {
        kernel_recv(
            gx,
            gb,
            gzero,
            gmask,
            lzero,
            lmask,
            px,
            pb,
            sem_send,
            sem_recv,
            send_mode,
            x0,
            y0,
            x1,
            y1,
            num_dests,
            N,
            C,
            R,
            S,
            PQ,
            Kb,
            start_p,
            start_q,
            delta_p,
            delta_q,
            delta_r,
            delta_s,
            end_q,
            zero_size,
            mask_size,
            x_pos,
            x_stride,
            b_pos);
    }
}

