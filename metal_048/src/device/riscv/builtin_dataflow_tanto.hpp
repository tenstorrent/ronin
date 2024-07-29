#pragma once


#include <string>
#include <unordered_map>

//
//    Generic list of dataflow builtins
//

#define DATAFLOW_TANTO_BUILTINS \
    DECL_BUILTIN(noc_async_read_global_dram, 5, 0) \
    DECL_BUILTIN(noc_async_read_global_l1, 5, 0) \
    DECL_BUILTIN(noc_async_write_global_dram, 5, 0) \
    DECL_BUILTIN(noc_async_write_global_l1, 5, 0)

//
//    Dataflow builtin enumeration
//

#define DECL_BUILTIN(name, count, result) name,

enum class DataflowTantoBuiltinId {
    START = 4096,
DATAFLOW_TANTO_BUILTINS
};

#undef DECL_BUILTIN

// public functions

struct DataflowTantoBuiltinEntry {
    std::string name;
    int count;
    int result;
};

std::unordered_map<DataflowTantoBuiltinId, DataflowTantoBuiltinEntry> &
    get_dataflow_tanto_builtin_map();

