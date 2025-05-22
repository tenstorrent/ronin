// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

static constexpr uint32
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2;

param<uint32> op_code;

void binary_op(math<T> acc, pipe<T> pa, pipe<T> pb, uint32 index) {
    if (op_code == OP_ADD) {
        acc.add(pa, pb, index, index, index);
    } else if (op_code == OP_SUB) {
        acc.sub(pa, pb, index, index, index);
    } else if (op_code == OP_MUL) {
        acc.mul(pa, pb, index, index, index);
    }
}

void kernel(
        pipe<T> pa, 
        pipe<T> pb, 
        pipe<T> pc, 
        uint32 num_frames,
        uint32 frame_tiles) {
    pa.set_frame(frame_tiles);
    pb.set_frame(frame_tiles);
    pc.set_frame(frame_tiles);
    for (uint32 frame = 0; frame < num_frames; frame++) {
        pc.reserve_back();
        pa.wait_front();
        pb.wait_front();
        math<T> acc;
        for (uint32 i = 0; i < frame_tiles; i++) {
            binary_op(acc, pa, pb, i);
        }
        for (uint32 i = 0; i < frame_tiles; i++) {
            acc.pack(i, pc);
        }
        pa.pop_front();
        pb.pop_front();
        pc.push_back();
    }
}

