// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp"

void kernel(
        pipe<T> pa,
        pipe<T> pb,
        pipe<T> pc,
        uint32 batch,
        uint32 Mt,
        uint32 Kt,
        uint32 Nt) {
    for (uint32 nb = 0; nb < batch; nb++) {
        for (uint32 mt = 0; mt < Mt; mt++) {
            for (uint32 nt = 0; nt < Nt; nt++) {
                math<T> acc;
                for (uint32 kt = 0; kt < Kt; kt++) {
                    pa.wait_front();
                    pb.wait_front();
                    acc.matmul(pa, pb, 0, 0, 0, false);
                    pb.pop_front();
                    pa.pop_front();
                }
                pc.reserve_back();
                acc.pack(0, pc);
                pc.push_back();
            }
        }
    }
}

