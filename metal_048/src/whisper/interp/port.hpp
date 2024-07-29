#pragma once

#ifdef _MSC_VER

#include <cstdint>
#include <cstdlib>
#include <bitset>
#include <algorithm>

#include <io.h>
#include <time.h>

//
//    Temporary portability definitions
//

#define __x86_64__ 1

#define __MINGW64__

#define __attribute(x)
#define __attribute__(x)

inline uint32_t __builtin_bswap32(uint32_t x) {
    return _byteswap_ulong(x);
}

inline uint64_t __builtin_bswap64(uint64_t x) {
    return _byteswap_uint64(x);
}

inline uint32_t __builtin_popcount(uint32_t x) {
    return uint32_t(std::bitset<32>(x).count());
}

inline uint32_t __builtin_popcountl(uint64_t x) {
    return uint32_t(std::bitset<64>(x).count());
}

inline uint32_t __builtin_clz(uint32_t x) {
    unsigned long y = 0;
    if (_BitScanReverse(&y, (unsigned long)x)) {
        return 32 - uint32_t(y);
    } else {
        return 32;
    }
}

inline uint32_t __builtin_clzl(uint64_t x) {
    unsigned long y = 0;
    if (_BitScanReverse64(&y, x)) {
        return 64 - uint32_t(y);
    } else {
        return 64;
    }
}

inline uint32_t __builtin_ctz(uint32_t x) {
    unsigned long y = 0;
    if (_BitScanForward(&y, (unsigned long)x)) {
        return uint32_t(y);
    } else {
        return 32;
    }
}

inline uint32_t __builtin_ctzl(uint64_t x) {
    unsigned long y = 0;
    if (_BitScanForward64(&y, x)) {
        return uint32_t(y);
    } else {
        return 64;
    }
}

typedef int ssize_t;

#else

#include <cstring> // missing in vector.cpp

#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>

// define anyway to disable most of syscalls
#define __MINGW64__

#define __cpp_lib_filesystem

#endif // _MSC_VER

