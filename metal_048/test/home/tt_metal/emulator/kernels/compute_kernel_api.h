#pragma once

#include <cstdint>

#include "kernel_structs.h"

#define API extern "C"

namespace ckernel {

API void rsqrt_tile_init(bool fast_and_approx);
API void rsqrt_tile(uint32_t idst, bool fast_and_approx);
API void sigmoid_tile_init();
API void sigmoid_tile(uint32_t idst);
API void log_tile_init();
API void log_tile(uint32_t idst);
API void log_with_base_tile_init();
API void log_with_base_tile(uint32_t idst, uint32_t base_scale);
API void tanh_tile_init();
API void signbit_tile_init();
API void signbit_tile(uint32_t idst);
API void tanh_tile(uint32_t idst);
API void abs_tile(uint32_t idst);
API void abs_tile_init();
API void sign_tile(uint32_t idst);
API void sign_tile_init();
API void square_tile(uint32_t idst);
API void square_tile_init();
API void ltz_tile(uint32_t idst);
API void ltz_tile_init();
API void eqz_tile(uint32_t idst);
API void eqz_tile_init();
API void lez_tile(uint32_t idst);
API void lez_tile_init();
// API void tiled_prod_tile(uint32_dst); // TODO
// API void tiled_prod_tile_init();      // TODO
API void gtz_tile(uint32_t idst);
API void gtz_tile_init();
API void nez_tile(uint32_t idst);
API void nez_tile_init();
API void gez_tile(uint32_t idst);
API void gez_tile_init();
API void power_tile(uint32_t idst, uint32_t param0);
API void power_tile_init();
// API void max_tile(uint32_t dst0, uint32_t dst1); // TODO
// API void max_tile_init();                        // TODO

#if 0 // TODO: Remove this
void get_next_op_info(tt::op_info_t &op_info) {
    // SKIPPED
}

void graph_interpreter_init() {
    // SKIPPED
}
#endif

API void exp2_tile(uint32_t idst);
API void exp2_tile_init();
API void heaviside_tile(uint32_t idst, uint32_t param0);
API void heaviside_tile_init();
// API void unary_ne_tile(uint32_t dst, uint32_t param0); // TODO
// API void unary_ne_tile_init();                         // TODO
API void expm1_tile(uint32_t idst);
API void expm1_tile_init();
API void asin_tile(uint32_t idst);
API void asin_tile_init();
API void atan_tile(uint32_t idst);
API void atan_tile_init();
API void acos_tile(uint32_t idst);
API void acos_tile_init();
API void silu_tile(uint32_t idst);
API void silu_tile_init();

/* TODO
API void topk_local_sort(
    uint32_t idst, 
    int idir, 
    int i_end_phase, 
    int i_start_phase = 0, 
    int i_end_step = 0, 
    int i_start_step = 0);
API void topk_merge(uint32_t idst, int m_iter, int k);
API void topk_rebuild(
    uint32_t idst, 
    bool idir, 
    int m_iter, 
    int k, 
    int logk, 
    int skip_second);
API void topk_tile_init();
API void dbg_halt();
API void dbg_unhalt();
API void dbg_read_dest_acc_row(int row_addr, uint32_t *rd_data);
API void unary_gt_tile(uint32_t idst, uint32_t param0);
API void unary_gt_tile_init();
API void unary_lt_tile(uint32_t idst, uint32_t param0);
API void unary_lt_tile_init();
*/

template<bool fast_and_approx = true>
void rsqrt_tile_init() {
    rsqrt_tile_init(fast_and_approx);
}

template<bool fast_and_approx = true>
API void rsqrt_tile(uint32_t idst) {
    rsqrt_tile(idst, fast_and_approx);
}

} // namespace ckernel

