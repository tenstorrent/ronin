
#include <string>
#include <unordered_map>
#include <utility>

#include "riscv/builtin_stdlib.hpp"

namespace {

#define DECL_BUILTIN(name, count) \
    {StdlibBuiltinId::name, {#name, count}},

std::unordered_map<StdlibBuiltinId, std::pair<std::string, int>> stdlib_builtin_map = {
STDLIB_BUILTINS
};

#undef DECL_BUILTIN

} // namespace

std::unordered_map<StdlibBuiltinId, std::pair<std::string, int>> &get_stdlib_builtin_map() {
    return stdlib_builtin_map;
}

