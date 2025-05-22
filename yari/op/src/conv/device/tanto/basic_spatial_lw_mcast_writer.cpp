// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gw,
        global<T> gy,
        pipe<T> pw,
        pipe<T> py,
        semaphore sem_send,
        semaphore sem_recv,
        uint32 send_mode,
        uint32 x0,
        uint32 y0,
        uint32 x1,
        uint32 y1,
        uint32 num_dests,
        uint32 N,
        uint32 K,
        uint32 PQ,
        uint32 PQK_tail,
        uint32 RSKC,
        uint32 y_pos,
        uint32 y_stride) {
    pw.set_frame(RSKC / 1024);
    py.set_frame(K / 32);
    if (send_mode != 0) {
        sem_recv.set(1);
        pw.reserve_back();
        pw.read(0, gw, 0, RSKC);
        read_barrier();
        sem_send.wait(num_dests);
        sem_send.set(0);
        pw.write_mcast(
            0,
            pw,
            0,
            RSKC,
            x0,
            y0,
            x1,
            y1,
            num_dests);
        sem_recv.set_mcast(
            sem_recv, 
            x0, 
            y0,
            x1,
            y1,
            num_dests);
        pw.push_back();
    } else {
        pw.reserve_back();
        sem_recv.set(0);
        sem_send.inc(x0, y0, 1);
        sem_recv.wait(1);
        pw.push_back();
    }
    uint32 y_start = y_pos;
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            uint32 count = (PQ - pq_start < 32) ? PQK_tail : K * 32;
            py.wait_front();
            py.write(0, gy, y_start, count);
            write_barrier();
            py.pop_front();
            y_start += K * 32;
        } // pq_start
        y_start += y_stride;
    } // n
}

