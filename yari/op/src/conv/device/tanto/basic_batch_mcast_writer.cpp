// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel_send(
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
        uint32 C,
        uint32 K,
        uint32 PQ,
        uint32 RS,
        uint32 y_pos,
        uint32 y_stride) {
    sem_recv.set(1);
    pw.set_frame(C / 32);
    py.set_frame(K / 32);
    uint32 y_start = y_pos;
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            uint32 w_start = 0;
            for (uint32 rs = 0; rs < RS; rs++) {
                for (uint32 k_start = 0; k_start < K; k_start += 32) {
                    pw.reserve_back();
                    pw.read(0, gw, w_start, C * 32);
                    read_barrier();
                    sem_send.wait(num_dests);
                    sem_send.set(0);
                    pw.write_mcast(
                        0,
                        pw,
                        0,
                        C * 32,
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
                    w_start += C * 32;
                } // k_start
            } // rs
            py.wait_front();
            py.write(0, gy, y_start, K * 32);
            write_barrier();
            py.pop_front();
            y_start += K * 32;
        } // pq_start
        y_start += y_stride;
    } // n
}

void kernel_recv(
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
        uint32 C,
        uint32 K,
        uint32 PQ,
        uint32 RS,
        uint32 y_pos,
        uint32 y_stride) {
    pw.set_frame(C / 32);
    py.set_frame(K / 32);
    uint32 y_start = y_pos;
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            for (uint32 rs = 0; rs < RS; rs++) {
                for (uint32 k_start = 0; k_start < K; k_start += 32) {
                    pw.reserve_back();
                    sem_recv.set(0);
                    sem_send.inc(x0, y0, 1);
                    sem_recv.wait(1);
                    pw.push_back();
                } // k_start
            } // rs
            py.wait_front();
            py.write(0, gy, y_start, K * 32);
            write_barrier();
            py.pop_front();
            y_start += K * 32;
        } // pq_start
        y_start += y_stride;
    } // n
}

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
        uint32 C,
        uint32 K,
        uint32 PQ,
        uint32 RS,
        uint32 y_pos,
        uint32 y_stride) {
    if (send_mode != 0) {
        kernel_send(
            gw,
            gy,
            pw,
            py,
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
            K,
            PQ,
            RS,
            y_pos,
            y_stride);
    } else {
        kernel_recv(
            gw,
            gy,
            pw,
            py,
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
            K,
            PQ,
            RS,
            y_pos,
            y_stride);
    }
}

