// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

static constexpr uint32
    BINARY_OP_ADD = 0,
    BINARY_OP_SUB = 1,
    BINARY_OP_MUL = 2;

static constexpr uint32
    UNARY_OP_RELU = 0,
    UNARY_OP_RELU6 = 1;

param<uint32> binary_op_code;
param<uint32> unary_op_code;

void binary_op(math<T> acc, pipe<T> pa, pipe<T> pb, uint32 index) {
    if (binary_op_code == BINARY_OP_ADD) {
        acc.add(pa, pb, index, index, index);
    } else if (binary_op_code == BINARY_OP_SUB) {
        acc.sub(pa, pb, index, index, index);
    } else if (binary_op_code == BINARY_OP_MUL) {
        acc.mul(pa, pb, index, index, index);
    }
}

void unary_op(math<T> acc, uint32 index, uint32 param0) {
    if (unary_op_code == UNARY_OP_RELU) {
        acc.relu(index);
    } else if (unary_op_code == UNARY_OP_RELU6) {
        acc.relu_max(index, 0x40c00000);
    }
}

void kernel(
        pipe<T> pa, 
        pipe<T> pb, 
        pipe<T> pc, 
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 unary_param0) {
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
            unary_op(acc, i, unary_param0);
        }
        for (uint32 i = 0; i < frame_tiles; i++) {
            acc.pack(i, pc);
        }
        pa.pop_front();
        pb.pop_front();
        pc.push_back();
    }
}

