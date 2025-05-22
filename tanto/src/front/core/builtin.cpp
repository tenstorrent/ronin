// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/builtin.hpp"

namespace ronin {
namespace tanto {
namespace front {

namespace {

const char *g_builtin_header = R"cpp(

//
//    Scalar types
//

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

struct bfloat16 {
    uint32 value;
};

//
//    Parameters
//

template<typename T, T VALUE = T(1)>
struct param {
    static constexpr T value = VALUE;
    constexpr operator T() {
        return value;
    }
};

//
//    Global buffer distributions
//

constexpr uint32 
    linear = 0,
    block = 1,
    cyclic = 2;

//
//    Forward declarations
//

template<typename T, uint32 DIST = linear, bool DRAM = true> class global;
template<typename T> class local;
template<typename T> class pipe;
class semaphore;
template<typename T> class math;

//
//    Global buffer
//

template<typename T, uint32 DIST, bool DRAM>
class global { };

//
//    Local buffer
//

template<typename T>
class local {
public:
    T get(uint32 index);
    void set(uint32 index, T value);
    template<bool DRAM>
    void read(
        uint32 dst_offset, 
        global<T, linear, DRAM> src, 
        uint32 src_offset, 
        uint32 count);
    template<uint32 DIST>
    void read(
        uint32 dst_offset, 
        global<T, DIST, true> src, 
        uint32 src_page,
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    template<bool DRAM>
    void write(
        uint32 src_offset, 
        global<T, linear, DRAM> dst, 
        uint32 dst_offset, 
        uint32 count);
    template<uint32 DIST>
    void write(
        uint32 src_offset, 
        global<T, DIST, true> dst, 
        uint32 dst_page,
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write_mcast(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void move_init(uint32 count);
    void move(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset);
    void move(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset);
};

//
//    Pipe
//

template<typename T>
class pipe {
public:
    void set_frame(uint32 tiles);
    void reserve_back();
    void push_back();
    void wait_front();
    void pop_front();
    template<bool DRAM>
    void read(
        uint32 dst_offset, 
        global<T, linear, DRAM> src, 
        uint32 src_offset, 
        uint32 count);
    template<uint32 DIST>
    void read(
        uint32 dst_offset, 
        global<T, DIST, true> src, 
        uint32 src_page,
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    template<bool DRAM>
    void write(
        uint32 src_offset, 
        global<T, linear, DRAM> dst, 
        uint32 dst_offset, 
        uint32 count);
    template<uint32 DIST>
    void write(
        uint32 src_offset, 
        global<T, DIST, true> dst, 
        uint32 dst_page,
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write_mcast(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void move_init(uint32 count);
    void move(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset);
    void move(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset);
};

//
//    Semaphore
//

class semaphore {
public:
    void set(uint32 value);
    void set_remote(
        semaphore src, 
        uint32 x, 
        uint32 y);
    void set_mcast(
        semaphore src,
        uint32 x_start,
        uint32 y_start,
        uint32 x_end,
        uint32 y_end,
        uint32 num_dests);
    void inc(uint32 x, uint32 y, uint32 value);
    void wait(uint32 value);
};

//
//    Math
//

template<typename T>
class math {
public:
    // pack
    template<typename U> 
        void pack(uint32 isrc, pipe<U> dst);
    template<typename U> 
        void pack_row(uint32 isrc, pipe<U> dst);
    template<typename U> 
        void pack_col(uint32 isrc, pipe<U> dst);
    template<typename U> 
        void pack_scalar(uint32 isrc, pipe<U> dst);
    void pack_relu_config(uint32 mode, uint32 threshold);
    // copy
    template<typename U>
        void copy(pipe<U> src, uint32 isrc, uint32 idst);
    // eltwise binary
    template<typename U, typename V>
    void add(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    // bcast
    template<typename U, typename V>
    void add_bcast_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub_bcast_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul_bcast_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void add_bcast_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub_bcast_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul_bcast_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void add_bcast_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub_bcast_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul_bcast_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    // matmul
    template<typename U, typename V>
    void matmul(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst,
        bool transpose);
    // reduce
    template<typename U, typename V>
    void reduce_max_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_max_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_max_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_sum_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_sum_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_sum_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    // transpose
    template<typename U>
        void transpose(pipe<U> src, uint32 isrc, uint32 idst);
    // eltwise unary
    void abs(uint32 idst);
    void acos(uint32 idst);
    void add_scalar(uint32 dst, uint32 param);
    void asin(uint32 idst);
    void atan(uint32 idst);
    void cos(uint32 idst);
    void div_scalar(uint32 idst, uint32 param);
    void elu(uint32 idst, uint32 param);
    void eqz(uint32 idst);
    void erf(uint32 idst);
    void erfc(uint32 idst);
    void erfinv(uint32 idst);
    void exp(uint32 idst);
    void exp2(uint32 idst);
    void expm1(uint32 idst);
    void gelu(uint32 idst);
    void gez(uint32 idst);
    void gtz(uint32 idst);
    void heaviside(uint32 idst, uint32 param);
    void i0(uint32 idst);
    void isfinite(uint32 idst);
    void isinf(uint32 idst);
    void isnan(uint32 idst);
    void isneginf(uint32 idst);
    void isposinf(uint32 idst);
    void leaky_relu(uint32 idst, uint32 param);
    void lez(uint32 idst);
    void log(uint32 idst);
    void log_with_base(uint32 idst, uint32 param);
    void logical_not(uint32 idst);
    void ltz(uint32 idst);
    void max(uint32 idst);
    void mul_scalar(uint32 idst, uint32 param);
    void nez(uint32 idst);
    void power(uint32 idst, uint32 param);
    void recip(uint32 idst);
    void relu(uint32 idst);
    void relu_max(uint32 idst, uint32 param);
    void relu_min(uint32 idst, uint32 param);
    void rsqrt(uint32 idst);
    void rsub_scalar(uint32 idst, uint32 param);
    void sigmoid(uint32 idst);
    void sign(uint32 idst);
    void signbit(uint32 idst);
    void sin(uint32 idst);
    void sqrt(uint32 idst);
    void square(uint32 idst);
    void sub_scalar(uint32 dst, uint32 param);
    void tan(uint32 idst);
    void tanh(uint32 idst);
};

//
//    Global dataflow functions
//

void read_barrier();
void write_barrier();

//
//    Global compute functions
//

template<typename U, typename V>
    void tilize_block(pipe<U> src, uint32 block, pipe<V> dst);
template<typename U, typename V>
    void untilize_block(pipe<U> src, uint32 block, pipe<V> dst);

)cpp";

} // namespace

const char *get_builtin_header() {
    return g_builtin_header;
}

} // namespace front
} // namespace tanto
} // namespace ronin

