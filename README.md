
# Project Ronin


## Objectives

Project Ronin aims at development of a technology stack including high-level general purpose programming model 
as well as respective software tools and libraries for Tenstorrent AI processors. 
The target audience includes software engineers implementing numeric applications
for Tenstorrent hardware.

The project envisages a stepwise development of a series of programming models of increased complexity. 
The first model in this series (code name Tanto) is currently available.  


## Tanto programming model

Tanto is a general-purpose high-level programming model for Tenstorrent AI processors. 
It is based on a conventional heterogeneous programming paradigm that views a computing system 
consisting of a host computer and one or more compute devices. 
Tanto applications consist of a host program and multiple device programs. 
The host program manages and coordinates execution of the device programs. 
It also creates and manages memory objects on computing devices. 
The device programs perform computations.

Tanto is designed on top of [TT-Metalium](https://github.com/tenstorrent/tt-metal) 
and introduces various high-level features that include:

- high-level abstractions for principal programming model objects 
  (including global and local buffers, pipes, semaphores, and math units);
- new compute and dataflow APIs based on these abstractions;
- explicit specification for kernel interfaces (runtime parameters);
- explicit specification for compile time parameters;
- automated insertion of code implementing low-level features;
  (including initialization of math primitives and register management functions).

The detailed specification of Tanto programming model can be found in these documents:

- [Tanto device programming interface specification](/tanto/doc/spec/tanto_device_api.md)
- [Tanto host programming interface specification](/tanto/doc/spec/tanto_host_api.md)


## Tanto core SDK

The Tanto core SDK implements Tanto programming model and
contains the following principal components:

- Tanto kernel compiler frontend;
- Tanto host API library;
- Tanto device runtime extensions.

Tanto compiler frontend translates Tanto device kernels to their TT-Metalium equivalents.

Tanto host API library implements programming interface for Tanto host programs
built on top of the TT-Metalium host API.

Tanto device runtime extensions augment the compute and dataflow device APIs of
TT-Metalium.


## Tanto kernel programming language

Tanto kernel programming language is designed for programming compute and dataflow kernels 
running on Tenstorrent AI devices. It is based on a subset of C++ with extensions implementing 
Tanto high-level abstractions.

Tanto kernel compiler frontend implements source-to-source translation of Tanto kernels into 
C++ TT-Metalium kernels. It maps Tanto abstractions to their lower-level equivalents and 
implements the respective optimisations. 
Tanto kernel language compiler is based on the Clang / LLVM compiler framework.


## Tanto host API

Tanto host API includes a collection of functions running on the host side and controlling 
all aspects of running Tanto kernels of a device. It is implemented as a C++ library mapping 
Tanto host functions to the respective lower-level TT-Metalium host API components. 
Tanto host API expresses its functionality in terms of higher-level abstractions.


## Yari reference framework

Yari is a reference software framework for deployment of neural network models.
It is designed for assessment and demonstration of Ronin capability for lean and flexible 
implementation of CNN models on TT AI processors.

Yari framework contains two parts:

- reference library of commonly used neural network operations;
- reference library of neural network models.


## Code structure

The project components are located in these subdirectories:

- `tanto`: Tanto core SDK;
- `algo`: simple reference algorithms;
- `yari`: Yari framework.


## Supported hardware

Currently, the single-chip Wormhole architecture is supported.
Support for further architectures is planned for the future releases.


