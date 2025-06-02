
# Tanto core SDK

This section contains components of the Tanto core SDK.

Tanto is a general-purpose high-level programming model for Tenstorrent AI processors. 
It is based on a conventional heterogeneous programming paradigm that views a computing system 
consisting of a host computer and one or more compute devices. 
Tanto applications consist of a host program and multiple device programs. 
The host program manages and coordinates execution of the device programs. 
It also creates and manages memory objects on computing devices. The device programs perform computations.

The Tanto core SDK contains the following principal components:

- Tanto kernel compiler frontend
- Tanto host API library
- Tanto device runtime extensions

Tanto compiler frontend translates Tanto device kernels to their TT-Metalium equivalents.

Tanto host API library implements programming interface for Tanto host programs
built on top of the TT-Metalium host API.

Tanto device runtime extensions augment the compute and dataflow device APIs of
TT-Metalium.

The specifications for Tanto programming interfaces are available here:

- [Tanto device programming interface specification](/tanto/doc/spec/tanto_device_api.md)
- [Tanto host programming interface specification](/tanto/doc/spec/tanto_host_api.md)


## 1. Tanto kernel compiler frontend

This section contains the source code of Tanto kernel compiler frontend.
This frontend translates Tanto device kernels to their TT-Metalium equivalents.

### Prerequisites

Building and running Tanto frontend requires Clang and LLVM version 16.0.x.

Installation of Clang / LLVM packages for Ubuntu is described at [apt.llvm.org](https://apt.llvm.org/).
The recommended method is described in the "Automatic installation script" section.
Tanto compiler frontend requires the "dev" Clang / LLVM packages, therefore the option
"install all apt.llvm.org packages at once" shall be chosen.

In this document it is assumed that the LLVM 16 installation directory is
`/usr/lib/llvm-16`. For a different location, the example
shell commands and build scripts must be updated accordingly.

### Code structure

The Tanto frontend code is contained in the subdirectory `src/front`.

### Building from source

The shell script for building Tanto frontend is located in the subdirectory `prj/front`.
It will place the built frontend executable into the subdirectory `bin/front`.

To build the frontend, set `prj/front` as your current directory and run these commands:

```
./build_core.sh
./build_cmd.sh
```

### Running Tanto frontend

To run Tanto frontend, use this command format:

```
tanto --mode=<mode> [-D<name>[=<value> ...] [-P<param>=<value> ...] <input>
```

where

* option `--mode` specifies compilation mode denoting the type of compiler Tanto kernel,
supported values for `<mode>` are `read`, `write`, and `compute`;
* option `-D` defines macro with name `<name>` and value `<value>`;
* option `-P` specifies compile time parameter with number `<param>`
and value `<value>`;
* argument `<input>` specifies the path to the file containing
the source code of a Tanto kernel to be compiled.

All compile time parameter declarations in the input Tanto kernel (specified as variables
of the built-in type `param`) receive numbers according to order of
their occurrence in the kernel code (the first parameter receives number 0).
For each compile time parameter, the respective value must be specified on
the command line. The value syntax must conform to the declared parameter type.

The frontend translates the input Tanto kernel into its TT-Metalium equivalent
and writes the result kernel code to the standard output.


### Running examples

To run examples, set `test/front` as your current directory.
To compile a set of Tanto kernels for various basic algorithms,
run this script:

```
./front_basic.sh
```

The frontend will read Tanto kernels in the `basic/tanto` subdirectory
and write compiled results to the `basic/metal` directory.


## 2. Tanto host API library

Tanto host API library implements programming interface for Tanto host programs
built on top of the TT-Metalium host API.

### Prerequisites

Tanto core SDK requires:

- Host machine running Ubuntu 22.04;
- Tenstorrent Wormhole device;
- TT-Metalium Release 0.58.

TT-Metalium must be downloaded and built from the 
[official repository](https://github.com/tenstorrent/tt-metal/tree/v0.58.1).

The following environment variables must be set for using Tanto and TT-Metalium:

```
export TT_METAL_HOME=<TT-Metalium home directory>
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH
export ARCH_NAME=wormhole_b0
```

where `<TT-Metalium home directory>` stands for the installation directory of TT-Metalium.

For example, if TT-Metalium was cloned into `/home/user/metal/r058` and built from source
as described in the TT-Metalium documentation, the `TT_METAL_HOME` must be set as:

```
export TT_METAL_HOME=/home/user/metal/r058/tt-metal
```

Building Tanto applications requires Clang 17. It is assumed that Clang C++ compiler is located at:

```
/usr/lib/llvm-17/bin/clang++
```

Installation of Clang / LLVM packages for Ubuntu is described at [apt.llvm.org](https://apt.llvm.org/).
We recommend using the method described in the "Automatic installation script" section.


### Building Tanto host API library

Tanto host API is a layer on top of Metal host API. 
The source code and build scripts are located in `tanto/src/host` and `tanto/prj/host` respectively.

To build the library, make `tanto/prj/host` your current directory and run this build script:

```
cd tanto/prj/host
./build_core.sh
```

The library file will be placed in `tanto/lib/host`.


## 3. Tanto device runtime extensions

Tanto device runtime extensions augment the compute and dataflow device APIs of
TT-Metalium.

Tanto device runtime extensions are located in `tanto/src/device/metal`. 
They must be copied to the dedicated location in the TT-Metalium home directory.
To perform this copy, make `tanto/prj/device` your current directory and run
the script `deploy_metal.sh`:

```
cd tanto/prj/device
./deploy_metal.sh
```

