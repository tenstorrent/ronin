
# TT-Metal emulator

## Introduction

The TT-Metal emulator is a software framework that emulates key functional aspects
of the TT-Metal programming model on conventional CPUs.

The main purpose of this project is providing a convenient toolchain for debugging and 
testing of TT-Metal applications on the conventional CPUs, 
without using the real Tenstorrent AI hardware.

The project aims at modeling the inherent hardware parallelism, 
so that a model of each hardware component runs in its own parallel light-weight thread. 
Modeling of the entire chip may require running many hundreds of parallel threads. 
This aspect makes using the standard OS threads impractical. 
Instead, the implementation uses a light-weight model of parallelism (asymmetric coroutines).

## Features

The TT-Emulator implements these principal features:

- Support of original TT-Metal APIs
- Strict separation of host and device models
- Dynamically compiled kernels
- Concurrency model based on asymmetric coroutines
- Concurrency units: Tensix RISC-V cores (up to 360)
    - BRISC
    - NCRISC
    - TRISC (one unit represents all three cores) 
- Granularity: monolith emulation of primitives:
    - Compute: low level kernels (LLK)
    - Dataflow: circular buffers, NoC, etc.

The current implementation corresponds to the original TT-Metal release 0.48.

The Grayskull and Wormhole architectures are supported.

## Prerequisites

The TT-Metal emulator requires LLVM 16. Make sure that the CLANG C++ compiler `clang++`
is installed and directly callable from the command line (this might require updating 
the `PATH` variable):

```
clang++ --version
```

## Code structure

The TT-Metal emulator code is contained in the subdirectory `src`.
It is structured into these modules:

```
device                   device side functionality
    api                  top level device side API
    arch                 architecture descriptors
    core                 core functionality
    dispatch             dispatch functionality
    ref                  reference implementation
    riscv                interfaces specific to RISC-V
    schedule             scheduler managing concurrency and synchronization
    vendor               third party software

tt_metal                 TT-Metal host side functionality and host API
    emulator             host side functionality specific for emulator
    ...                  directories inherited from original TT-Metal framework

whisper                  functionality related to RISC-V ISA simulator
    interp               RISC-V ISA simulator
    linker               lightweight linker for kernels
    riscv                top level API 
```

The emulator is implemented completely in C++17 programming language.
This source code has no external dependencies besides the standard C++ libraries.

NOTE: The libraries in the `tt_metal` section must be recompiled for each supported
architecture. The target architecture is specified by the macro `HW_ARCH_CONFIG` defined
in the `tt_metal/emulator/hw/config/hw_arch.h` header file. 

## Building from source

The shell scripts for building the emulator libraries and examples are contained
in the subdirectory `prj`. They will place the built libraries and example applications
into subdirectories `lib` and `bin` respectively.

To build all libraries and applications, set `prj` as your current directory and
run this command:

```
./build_all.sh
```

## Running examples

To run programming examples, set `test` as your current directory.
Then set these environment variables:

```
export ARCH_NAME=grayskull
export TT_METAL_HOME=./home
```

Make sure that `clang++` is directly callable from the command line as described above
in section "Prerequisites".

Run individual examples located in `bin/examples` directory, for instance:

```
../bin/examples/eltwise_binary
```

