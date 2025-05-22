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
        uint32 PQ_full,
        uint32 PQ_tail,
        uint32 RSK,
        uint32 y_pos,
        uint32 y_stride) {
    pw.set_frame(RSK / 32);
    py.set_frame(K / 32);
    if (send_mode != 0) {
        sem_recv.set(1);
        pw.reserve_back();
        pw.read(0, gw, 0, RSK * 32);
        read_barrier();
        sem_send.wait(num_dests);
        sem_send.set(0);
        pw.write_mcast(
            0,
            pw,
            0,
            RSK * 32,
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
        for (uint32 pq_full = 0; pq_full < PQ_full; pq_full++) {
            py.wait_front();
            py.write(0, gy, y_start, K * 32);
            write_barrier();
            py.pop_front();
            y_start += K * 32;
        } // pq_full
        if (PQ_tail != 0) {
            py.wait_front();
            py.write(0, gy, y_start, PQ_tail);
            write_barrier();
            py.pop_front();
        }
        y_start += y_stride;
    } // n
}

