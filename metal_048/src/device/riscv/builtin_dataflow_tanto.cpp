
#include <string>
#include <unordered_map>

#include "riscv/builtin_dataflow_tanto.hpp"

namespace {

#define DECL_BUILTIN(name, count, result) \
    {DataflowTantoBuiltinId::name, {#name, count, result}},

std::unordered_map<DataflowTantoBuiltinId, DataflowTantoBuiltinEntry> 
    dataflow_tanto_builtin_map = {
DATAFLOW_TANTO_BUILTINS
};

#undef DECL_BUILTIN

} // namespace

std::unordered_map<DataflowTantoBuiltinId, DataflowTantoBuiltinEntry> &
        get_dataflow_tanto_builtin_map() {
    return dataflow_tanto_builtin_map;
}

