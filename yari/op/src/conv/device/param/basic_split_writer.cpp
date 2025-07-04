// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

param<uint32> K;
param<uint32> PQ;
param<uint32> RS;
param<uint32> KC;
param<uint32> Kb;
param<uint32> KbC;
param<uint32> RSKbC;
param<uint32> y_stride;

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
        uint32 w_pos,
        uint32 y_pos) {
    pw.set_frame(RSKbC / 1024);
    py.set_frame(Kb / 32);
    if (send_mode != 0) {
        uint32 w_start = w_pos;
        uint32 b_start = 0;
        sem_recv.set(1);
        pw.reserve_back();
        for (uint32 rs = 0; rs < RS; rs++) {
            pw.read(b_start, gw, w_start, KbC);
            b_start += KbC;
            w_start += KC;
        }
        read_barrier();
        sem_send.wait(num_dests);
        sem_send.set(0);
        pw.write_mcast(
            0,
            pw,
            0,
            RSKbC,
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
            py.wait_front();
            uint32 p_start = 0;
            for (uint32 i = 0; i < 32; i++) {
                py.write(p_start, gy, y_start, Kb);
                p_start += Kb;
                y_start += K;
            }
            write_barrier();
            py.pop_front();
        } // pq_start
        y_start += y_stride;
    } // n
}

