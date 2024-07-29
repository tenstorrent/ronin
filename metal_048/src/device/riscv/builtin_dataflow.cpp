
#include <string>
#include <unordered_map>

#include "riscv/builtin_dataflow.hpp"

namespace {

#define DECL_BUILTIN(name, count, result) \
    {DataflowBuiltinId::name, {#name, count, result}},

std::unordered_map<DataflowBuiltinId, DataflowBuiltinEntry> dataflow_builtin_map = {
DATAFLOW_BUILTINS
};

#undef DECL_BUILTIN

} // namespace

std::unordered_map<DataflowBuiltinId, DataflowBuiltinEntry> &get_dataflow_builtin_map() {
    return dataflow_builtin_map;
}

