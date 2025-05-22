
# Reference algorithms

This subdirectory contains implementation of reference algorithms for the Ronin framework.

Reference algorithms are grouped into categories.
Currently, only one category "basic" is available.
Further algorithm categories may be added in the future.

## 1. General information

Information in this section applies to all algorithm categories.

## Prerequisites

The prerequisites of Tanto core SDK must be fulfilled.
The Tanto SDK must be installed.
The respective environment variables must be set.
Detailed description of these steps can be found in the section `tanto` of this repository.

## Deploying device kernels

Before running the test applications, the device kernels must
be deployed to the dedicated subdirectory of the TT-Metalium home directory.
To perform deployment, make `algo/prj` your current directory and run this script:

```
./deploy_metal.sh
```

## 2. Basic algorithms

The basic algorithms demonstrate usage of individual Ronin device primitives and 
can be used as both unit tests and basic programming examples. They are roughly
equivalent to the respective TT-Metalium programming examples.

## Code structure

The code is contained in the subdirectory `src/basic`.
It is structured into these modules:

```
basic
    device          device code (kernels)
        metal       automatically generated TT-Metalium kernels
        tanto       original Tanto kernels
    host            host code
        tanto       algorithm library implemented using Tanto host API
    test            test applications
        tanto       tests for Tanto host API
        util        test utility library
```

## Building from source

The shell scripts for building the libraries and examples are contained
in the subdirectory `prj/basic`. They will place the built libraries and example applications
into subdirectories `lib/basic` and `bin/basic` respectively.

To build all libraries and applications, set `prj/basic` as your current directory and
run this script:

```
./build_all.sh
```

## Running test application

Make `algo` your current directory.
To run the test application, use this command:

```
./bin/basic/test_tanto <op>
```

where `<op>` is one of:

```
eltwise_binary
eltwise_sfpu
bcast
matmul_single
matmul_multi
reduce
transpose_wh
unpack_tilize
unpack_untilize
```

