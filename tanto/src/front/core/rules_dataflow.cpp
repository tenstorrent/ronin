// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>

#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"

#include "core/matchers.hpp"
#include "core/rules.hpp"

/*
Global buffers require these extensions to dataflow API

void noc_async_read_global_dram(
    uint32_t dst_addr,
    uint32_t src_addr,
    uint32_t src_log2_page_size,
    uint32_t src_offset,
    uint32_t len_bytes);
void noc_async_read_global_l1(
    uint32_t dst_addr,
    uint32_t src_addr,
    uint32_t src_log2_page_size,
    uint32_t src_offset,
    uint32_t len_bytes);
void noc_async_write_global_dram(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t dst_log2_page_size,
    uint32_t dst_offset,
    uint32_t len_bytes);
void noc_async_write_global_l1(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t dst_log2_page_size,
    uint32_t dst_offset,
    uint32_t len_bytes);

void noc_async_read_block_dram(
    uint32_t dst_addr,
    uint32_t src_addr,
    uint32_t src_page_size,
    uint32_t src_page_id,
    uint32_t src_offset,
    uint32_t len_bytes);
void noc_async_read_cyclic_dram(
    uint32_t dst_addr,
    uint32_t src_addr,
    uint32_t src_page_size,
    uint32_t src_page_id,
    uint32_t src_offset,
    uint32_t len_bytes);
void noc_async_write_block_dram(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t dst_page_size,
    uint32_t dst_page_id,
    uint32_t dst_offset,
    uint32_t len_bytes);
void noc_async_write_cyclic_dram(
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t dst_page_size,
    uint32_t dst_page_id,
    uint32_t dst_offset,
    uint32_t len_bytes);
*/

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace transformer;

namespace {

Stencil make_t_stencil() {
    return selectBound(
        {
            {"T_uint32", cat("uint32_t")},
            {"T_float", cat("float")},
            {"T_bfloat16", cat("bfloat16_t")}
        },
        cat("float") // cannot happen
    );
}

Stencil make_t_shift_stencil() {
    return selectBound(
        {
            {"T_uint32", cat("2")},
            {"T_float", cat("2")},
            {"T_bfloat16", cat("1")}
        },
        cat("0") // cannot happen
    );
}

Stencil make_dist_suffix_stencil() {
    return selectBound(
        {
            {"DIST_linear", cat("linear")},
            {"DIST_block", cat("block")},
            {"DIST_cyclic", cat("cyclic")}
        },
        cat("dist") // cannot happen
    );
}

Stencil make_dram_suffix_stencil() {
    return selectBound(
        {
            {"DRAM_false", cat("l1")},
            {"DRAM_true", cat("dram")}
        },
        cat("dram") // cannot happen
    );
}

const char g_cast_ptr_head[] = "reinterpret_cast<volatile tt_l1_ptr ";
const char g_cast_ptr_tail[] = " *>";
const char g_cast_ptr_op[] = "reinterpret_cast<volatile tt_l1_ptr uint32_t *>";

} // namespace

//
//    RuleFactory
//

// local

RewriteRule RuleFactory::make_local_get_rule() {
    // self.get(index)
    //     =>
    // reinterpret_cast<volatile tt_l1_ptr T *>(self.addr)[index]
    //    ACHTUNG: Support for embedded builtin calls will require normalization pass
    auto t = make_t_stencil();
    return makeRule(
        make_member_call_1_with_t_matcher("local", "get"),
        changeTo(
            node("stmt"),
            cat(
                g_cast_ptr_head, t, g_cast_ptr_tail, 
                    "(", access("self", "addr"), ")",
                    "[", expression("arg0"), "]")));
}

RewriteRule RuleFactory::make_local_set_rule() {
    // self.set(index, value);
    //    =>
    // reinterpret_cast<volatile tt_l1_ptr T *>(self.addr)[index] = value;
    auto t = make_t_stencil();
    return makeRule(
        make_member_call_2_with_t_matcher("local", "set"),
        changeTo(
            statement("stmt"),
            cat(
                g_cast_ptr_head, t, g_cast_ptr_tail, 
                    "(", access("self", "addr"), ")",
                    "[", expression("arg0"), "]",
                    " = ", expression("arg1"), ";")));
}

RewriteRule RuleFactory::make_local_read_global_rule() {
    // self.read(dst_offset, src, src_offset, count);
    //     =>
    // noc_async_read_global_[dram|l1](
    //     self.addr + (dst_offset << T_SHIFT),
    //     src.addr,
    //     src.log2_page_size,
    //     src_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dram = make_dram_suffix_stencil();
    return makeRule(
        make_member_call_4_with_t_dist_dram_matcher("local", "read"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_global_", dram, "(",
                    access("self", "addr"), 
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), " << ", t_shift, ", ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_read_global_dist_rule() {
    // self.read(dst_offset, src, src_page, src_offset, count);
    //     =>
    // noc_async_read_[linear|block|cyclic]_dram(
    //     self.addr + (dst_offset << T_SHIFT),
    //     src.addr,
    //     src.log2_page_size,
    //     src_page,
    //     src_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dist = make_dist_suffix_stencil();
    return makeRule(
        make_member_call_5_with_t_dist_dram_matcher("local", "read"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_", dist, "_dram(",
                    access("self", "addr"), 
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), ", ",
                    expression("arg3"), " << ", t_shift, ", ",
                    expression("arg4"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_read_local_rule() {
    // self.read(dst_offset, src, src_offset, count);
    //     =>
    // noc_async_read(
    //     get_noc_addr(src.addr + (src_offset << T_SHIFT)), 
    //     self.addr + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("local", "read", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(", access("arg1", "addr"), 
                        " + (", expression("arg2"), " << ", t_shift, ")), ",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_read_local_xy_rule() {
    // self.read(dst_offset, src, src_offset, count, x, y);
    //     =>
    // noc_async_read(
    //     get_noc_addr(x, y, src.addr + (src_offset << T_SHIFT)), 
    //     self.addr + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("local", "read", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_read_pipe_rule() {
    // self.read(dst_offset, src, src_offset, count);
    //     =>
    // noc_async_read(
    //     get_noc_addr(get_read_ptr(src.cb_id) + (src_offset << T_SHIFT)), 
    //     self.addr + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("local", "read", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(get_read_ptr(", access("arg1", "cb_id"), 
                        ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_read_pipe_xy_rule() {
    // self.read(dst_offset, src, src_offset, count, x, y);
    //     =>
    // noc_async_read(
    //     get_noc_addr(x, y, get_read_ptr(src.cb_id) + (src_offset << T_SHIFT)), 
    //     self.addr + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("local", "read", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        "get_read_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_write_global_rule() {
    // self.write(src_offset, dst, dst_offset, count);
    //     =>
    // noc_async_write_global_[dram|l1](
    //     self.addr + (src_offset << T_SHIFT),
    //     dst.addr,
    //     dst.log2_page_size,
    //     dst_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dram = make_dram_suffix_stencil();
    return makeRule(
        make_member_call_4_with_t_dist_dram_matcher("local", "write"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_global_", dram, "(",
                    access("self", "addr"), 
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), " << ", t_shift, ", ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_write_global_dist_rule() {
    // self.write(src_offset, dst, dst_page, dst_offset, count);
    //     =>
    // noc_async_write_[linear|block|cyclic]_dram(
    //     self.addr + (src_offset << T_SHIFT),
    //     dst.addr,
    //     dst.log2_page_size,
    //     dst_page,
    //     dst_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dist = make_dist_suffix_stencil();
    return makeRule(
        make_member_call_5_with_t_dist_dram_matcher("local", "write"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_", dist, "_dram(",
                    access("self", "addr"), 
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), ", ",
                    expression("arg3"), " << ", t_shift, ", ",
                    expression("arg4"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_write_local_rule() {
    // self.write(src_offset, dst, dst_offset, count);
    //     =>
    // noc_async_write(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_addr(dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("local", "write", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(", access("arg1", "addr"), 
                        " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_write_local_xy_rule() {
    // self.write(src_offset, dst, dst_offset, count, x, y);
    //     =>
    // noc_async_write(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_addr(x, y, dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("local", "write", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_write_pipe_rule() {
    // self.write(src_offset, dst, dst_offset, count);
    //     =>
    // noc_async_write(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_addr(get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("local", "write", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(get_write_ptr(", access("arg1", "cb_id"), 
                        ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_write_pipe_xy_rule() {
    // self.write(src_offset, dst, dst_offset, count, x, y);
    // noc_async_write(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_addr(x, y, get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("local", "write", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        "get_write_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_write_mcast_local_rule() {
    // self.write_mcast(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_async_write_multicast(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("local", "write_mcast", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_local_write_mcast_with_self_local_rule() {
    // self.write_mcast_with_self(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_async_write_multicast_loopback_src(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("local", "write_mcast_with_self", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast_loopback_src(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_local_write_mcast_pipe_rule() {
    // self.write_mcast(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_async_write_multicast(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("local", "write_mcast", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        "get_write_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_local_write_mcast_with_self_pipe_rule() {
    // self.write_mcast_with_self(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_async_write_multicast_loopback_src(
    //     self.addr + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("local", "write_mcast_with_self", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast_loopback_src(",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        "get_write_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_local_move_init_rule() {
    // self.move_init(count);
    //     =>
    // noc_async_read_one_packet_set_state(get_noc_addr(0), count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_1_with_t_matcher("pipe", "move_init"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_one_packet_set_state(get_noc_addr(0), ",
                    expression("arg0"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_local_move_local_rule() {
    // self.move(dst_offset, src, src_offset);
    //     =>
    // noc_async_read_one_packet_with_state(
    //     src.addr + (src_offset << T_SHIFT), 
    //     self.addr + (dst_offset << T_SHIFT));
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_3_with_t_arg1_matcher("local", "move", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_one_packet_with_state(",
                    access("arg1", "addr"), 
                        " + (", expression("arg2"), " << ", t_shift, "), ",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "));")));
}

RewriteRule RuleFactory::make_local_move_pipe_rule() {
    // self.move(dst_offset, src, src_offset);
    //     =>
    // noc_async_read_one_packet_with_state(
    //     get_read_ptr(src.cb_id) + (src_offset << T_SHIFT), 
    //     self.addr + (dst_offset << T_SHIFT));
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_3_with_t_arg1_matcher("local", "move", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_one_packet_with_state(",
                    "get_read_ptr(", access("arg1", "cb_id"), 
                        ") + (", expression("arg2"), " << ", t_shift, "), ",
                    access("self", "addr"),
                        " + (", expression("arg0"), " << ", t_shift, "));")));
}

// pipe

RewriteRule RuleFactory::make_pipe_read_global_rule() {
    // self.read(dst_offset, src, src_offset, count);
    //     =>
    // noc_async_read_global_[dram|l1](
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT),
    //     src.addr,
    //     src.log2_page_size,
    //     src_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dram = make_dram_suffix_stencil();
    return makeRule(
        make_member_call_4_with_t_dist_dram_matcher("pipe", "read"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_global_", dram, "(",
                    "get_write_ptr(", access("self", "cb_id"), 
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), " << ", t_shift, ", ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_read_global_dist_rule() {
    // self.read(dst_offset, src, src_page, src_offset, count);
    //     =>
    // noc_async_read_[linear|block|cyclic]_dram(
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT),
    //     src.addr,
    //     src.log2_page_size,
    //     src_page,
    //     src_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dist = make_dist_suffix_stencil();
    return makeRule(
        make_member_call_5_with_t_dist_dram_matcher("pipe", "read"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_", dist, "_dram(",
                    "get_write_ptr(", access("self", "cb_id"), 
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), ", ",
                    expression("arg3"), " << ", t_shift, ", ",
                    expression("arg4"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_read_local_rule() {
    // self.read(dst_offset, src, src_offset, count);
    //     =>
    // noc_async_read(
    //     get_noc_addr(src.addr + (src_offset << T_SHIFT)), 
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("pipe", "read", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(", access("arg1", "addr"), 
                        " + (", expression("arg2"), " << ", t_shift, ")), ",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_read_local_xy_rule() {
    // self.read(dst_offset, src, src_offset, count, x, y);
    //     =>
    // noc_async_read(
    //     get_noc_addr(x, y, src.addr + (src_offset << T_SHIFT)), 
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("pipe", "read", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_read_pipe_rule() {
    // self.read(dst_offset, src, src_offset, count);
    //     =>
    // noc_async_read(
    //     get_noc_addr(get_read_ptr(src.cb_id) + (src_offset << T_SHIFT)), 
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("pipe", "read", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(get_read_ptr(", access("arg1", "cb_id"), 
                        ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_read_pipe_xy_rule() {
    // self.read(dst_offset, src, src_offset, count, x, y);
    //     =>
    // noc_async_read(
    //     get_noc_addr(x, y, get_read_ptr(src.cb_id) + (src_offset << T_SHIFT)), 
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("pipe", "read", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read(",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        "get_read_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_write_global_rule() {
    // self.write(src_offset, dst, dst_offset, count);
    //     =>
    // noc_async_write_global_[dram|l1](
    //     get_read_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     dst.addr,
    //     dst.log2_page_size,
    //     dst_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dram = make_dram_suffix_stencil();
    return makeRule(
        make_member_call_4_with_t_dist_dram_matcher("pipe", "write"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_global_", dram, "(",
                    "get_read_ptr(", access("self", "cb_id"), 
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), " << ", t_shift, ", ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_write_global_dist_rule() {
    // self.write(src_offset, dst, dst_page, dst_offset, count);
    //     =>
    // noc_async_write_[linear|block|cyclic]_dram(
    //     get_read_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     dst.addr,
    //     dst.log2_page_size,
    //     dst_page,
    //     dst_offset << T_SHIFT,
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    auto dist = make_dist_suffix_stencil();
    return makeRule(
        make_member_call_5_with_t_dist_dram_matcher("pipe", "write"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_", dist, "_dram(",
                    "get_read_ptr(", access("self", "cb_id"), 
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    access("arg1", "addr"), ", ",
                    access("arg1", "log2_page_size"), ", ",
                    expression("arg2"), ", ",
                    expression("arg3"), " << ", t_shift, ", ",
                    expression("arg4"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_write_local_rule() {
    // self.write(src_offset, dst, dst_offset, count);
    //     =>
    // noc_async_write(
    //     get_read_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_addr(dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("pipe", "write", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    "get_read_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(", access("arg1", "addr"), 
                        " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_write_local_xy_rule() {
    // self.write(src_offset, dst, dst_offset, count, x, y);
    //     =>
    // noc_async_write(
    //     get_read_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_addr(x, y, dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("pipe", "write", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    "get_read_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_write_pipe_rule() {
    // self.write(src_offset, dst, dst_offset, count);
    //     =>
    // noc_async_write(
    //     get_read_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_addr(get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_4_with_t_arg1_matcher("pipe", "write", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    "get_read_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(get_write_ptr(", access("arg1", "cb_id"), 
                        ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_write_pipe_xy_rule() {
    // self.write(src_offset, dst, dst_offset, count, x, y);
    //     =>
    // noc_async_write(
    //     get_read_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_addr(x, y, get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_6_with_t_arg1_matcher("pipe", "write", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write(",
                    "get_read_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        "get_write_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_write_mcast_local_rule() {
    // self.write_mcast(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_async_write_multicast(
    //     get_write_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("pipe", "write_mcast", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast(",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_pipe_write_mcast_with_self_local_rule() {
    // self.write_mcast_with_self(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_async_write_multicast_loopback_src(
    //     get_write_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, dst.addr + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("pipe", "write_mcast_with_self", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast_loopback_src(",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        access("arg1", "addr"), 
                            " + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_pipe_write_mcast_pipe_rule() {
    // self.write_mcast(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_async_write_multicast(
    //     get_write_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("pipe", "write_mcast", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast(",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        "get_write_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_pipe_write_mcast_with_self_pipe_rule() {
    // self.write_mcast_with_self(
    //     src_offset, dst, dst_offset, count, x_start, y_start, x_end, y_end);
    //     =>
    // noc_async_write_multicast_loopback_src(
    //     get_write_ptr(self.cb_id) + (src_offset << T_SHIFT),
    //     get_noc_multicast_addr(
    //         x_start, y_start, x_end, y_end, get_write_ptr(dst.cb_id) + (dst_offset << T_SHIFT)),
    //     count << T_SHIFT,
    //     num_dests);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_9_with_t_arg1_matcher("pipe", "write_mcast_with_self", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_write_multicast_loopback_src(",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "), ",
                    "get_noc_multicast_addr(", 
                        node("arg4"), ", ", 
                        node("arg5"), ", ", 
                        node("arg6"), ", ", 
                        node("arg7"), ", ", 
                        "get_write_ptr(", access("arg1", "cb_id"), 
                            ") + (", expression("arg2"), " << ", t_shift, ")), ",
                    expression("arg3"), " << ", t_shift, ", ",
                    node("arg8"), ");")));
}

RewriteRule RuleFactory::make_pipe_move_init_rule() {
    // self.move_init(count);
    //     =>
    // noc_async_read_one_packet_set_state(get_noc_addr(0), count << T_SHIFT);
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_1_with_t_matcher("pipe", "move_init"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_one_packet_set_state(get_noc_addr(0), ",
                    expression("arg0"), " << ", t_shift, ");")));
}

RewriteRule RuleFactory::make_pipe_move_local_rule() {
    // self.move(dst_offset, src, src_offset);
    //     =>
    // noc_async_read_one_packet_with_state(
    //     src.addr + (src_offset << T_SHIFT), 
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT));
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_3_with_t_arg1_matcher("pipe", "move", "local"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_one_packet_with_state(",
                    access("arg1", "addr"), 
                        " + (", expression("arg2"), " << ", t_shift, "), ",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "));")));
}

RewriteRule RuleFactory::make_pipe_move_pipe_rule() {
    // self.move(dst_offset, src, src_offset);
    //     =>
    // noc_async_read_one_packet_with_state(
    //     get_read_ptr(src.cb_id) + (src_offset << T_SHIFT), 
    //     get_write_ptr(self.cb_id) + (dst_offset << T_SHIFT));
    auto t_shift = make_t_shift_stencil();
    return makeRule(
        make_member_call_3_with_t_arg1_matcher("pipe", "move", "pipe"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_async_read_one_packet_with_state(",
                    "get_read_ptr(", access("arg1", "cb_id"), 
                        ") + (", expression("arg2"), " << ", t_shift, "), ",
                    "get_write_ptr(", access("self", "cb_id"),
                        ") + (", expression("arg0"), " << ", t_shift, "));")));
}

// semaphore

RewriteRule RuleFactory::make_semaphore_set_rule() {
    // self.set(value);
    //     =>
    // noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t *>(self.addr), value);
    return makeRule(
        make_member_call_1_matcher("semaphore", "set"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_semaphore_set(", 
                    g_cast_ptr_op, "(", access("self", "addr"), "), ", 
                    node("arg0"), ");")));
}

RewriteRule RuleFactory::make_semaphore_set_remote_rule() {
    // self.set_remote(src, x, y);
    //     =>
    // noc_semaphore_set_remote(
    //     src.addr,
    //     get_noc_addr(x, y, self.addr));
    return makeRule(
        make_member_call_3_matcher("semaphore", "set_remote"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_semaphore_set_remote(", 
                    access("arg0", "addr"), ", ",
                    "get_noc_addr(",
                        node("arg1"), ", ",
                        node("arg2"), ", ",
                        access("self", "addr"), "));")));
}

RewriteRule RuleFactory::make_semaphore_set_mcast_rule() {
    // self.set_mcast(src, x_start, y_start, x_end, y_end, num_dests);
    //     =>
    // noc_semaphore_set_multicast(
    //     src.addr,
    //     get_noc_multicast_addr(x_start, y_start, x_end, y_end, self.addr),
    //     num_dests);
    return makeRule(
        make_member_call_6_matcher("semaphore", "set_mcast"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_semaphore_set_multicast(",
                    access("arg0", "addr"), ", ",
                    "get_noc_multicast_addr(",
                        node("arg1"), ", ",
                        node("arg2"), ", ",
                        node("arg3"), ", ",
                        node("arg4"), ", ",
                        access("self", "addr"), "),",
                    node("arg5"), ");")));
}

RewriteRule RuleFactory::make_semaphore_inc_rule() {
    // self.inc(x, y, value);
    //     =>
    // noc_semaphore_inc(get_noc_addr(x, y, self.addr), value);
    return makeRule(
        make_member_call_3_matcher("semaphore", "inc"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_semaphore_inc(",
                    "get_noc_addr(",
                        node("arg0"), ",",
                        node("arg1"), ",",
                        access("self", "addr"), "), ", 
                    node("arg2"), ");")));
}

RewriteRule RuleFactory::make_semaphore_wait_rule() {
    // self.wait(value);
    //     =>
    // noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t *>(self.addr), value);
    return makeRule(
        make_member_call_1_matcher("semaphore", "wait"),
        changeTo(
            statement("stmt"),
            cat(
                "noc_semaphore_wait(", 
                    g_cast_ptr_op, "(", access("self", "addr"), "), ", 
                    node("arg0"), ");")));
}

// global functions

RewriteRule RuleFactory::make_func_read_barrier_rule() {
    // read_barrier();
    //     =>
    // noc_async_read_barrier();
    return makeRule(
        make_func_call_0_matcher("read_barrier"),
        changeTo(
            statement("stmt"),
            cat("noc_async_read_barrier();")));
}

RewriteRule RuleFactory::make_func_write_barrier_rule() {
    // write_barrier();
    //     =>
    // noc_async_write_barrier();
    return makeRule(
        make_func_call_0_matcher("write_barrier"),
        changeTo(
            statement("stmt"),
            cat("noc_async_write_barrier();")));
}

} // namespace front
} // namespace tanto
} // namespace ronin

