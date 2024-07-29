#pragma once

#include <string>
#include <unordered_map>
#include <utility>

//
//    Generic list of stdlib builtins
//

#define STDLIB_BUILTINS \
    DECL_BUILTIN(memset, 3) \

//
//    Stdlib builtin enumeration
//

#define DECL_BUILTIN(name, count) name,

enum class StdlibBuiltinId {
    START = 2048,
STDLIB_BUILTINS
};

#undef DECL_BUILTIN

// public functions

std::unordered_map<StdlibBuiltinId, std::pair<std::string, int>> &get_stdlib_builtin_map();

