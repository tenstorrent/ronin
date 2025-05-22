
## Project Ronin

## Tanto device programming interface specification

*Version 1.1*

*Edition 07.05.2025*

<br>

## 1 Introduction

Tanto is a general-purpose high-level programming model for Tenstorrent AI processors. 
It is based on a conventional heterogeneous programming paradigm that views a computing system 
consisting of a host computer and one or more compute devices. 
Tanto applications consist of a host program and multiple device programs. 
The host program manages and coordinates execution of the device programs. 
It also creates and manages memory objects on computing devices. The device programs perform computations.

This document specifies a programming interface for implementation of Tanto device programs.


## 2 Architecture

This chapter specifies the architecture of Tanto device programs.

### 2.1 Abstract hardware model

This section describes an abstract high-level view of Tanto hardware used as the base 
for specification of Tanto programming interfaces. 
Tanto device represents a single Tenstorrent AI processor chip (like Wormhole or Blackhole).

NOTE: At present, Tanto does not support clusters of multiple processor chips. 
This support may be added in the future versions.

The Tanto device contains multiple processing cores arranged as a rectangular grid. 
The number of processing cores and grid dimensions are architecture dependent. 
(For example, on Wormhole the grid of processing cores has dimensions of 8 x 8.) 

NOTE: Tanto processing cores correspond to Tensix worker cores.

Every processing core is addressed by a pair of coordinates. 
There are two coordinate spaces: logical and physical. 
Logical coordinates define a position of the processing core in the Tanto device grid. 
Physical coordinates define a position of the corresponding hardware element (Tensix core) 
in a physical device grid. Logical and physical coordinates of a Tanto processing core may differ. 
The programming model provides functionality for mapping logical to physical coordinates. 
Rectangular regions in the logical coordinate space are mapped to rectangular regions 
in the physical coordinate space.

Each processing core consists of a set of functional units implementing various types of data processing. 
Each processing core contains a block of fast (on-chip) memory referred to as L1. 
The size of L1 memory is architecture dependent (for example, the size of L1 memory on each Tensix core
is approximately 1.5 MB for Wormhole).

Tanto device contains global memory referred to as DRAM and arranged in several banks. 
The global memory size and number of banks is architecture dependent 
(for example, the size of is 12 GB for Wormhole; a single bank holds 1 GB).

Processing cores are interconnected via the Network on Chip (NoC) that can transfer data across L1 memory units. 
NoC can be also used to transfer data between L1 memory of processing cores and DRAM.

### 2.2 Execution model

A top execution unit in a Tanto programming model is represented by a device program. 
A program runs on a Tanto device and specifies computations collectively performed on a subset 
of its processing cores. 
Typically, an application specifies multiple programs that are sequentially launched on the device. 
Device programs are created and launched by the host program.

The device program specifies a collection of lower-level execution units referred to as kernels. 
Each kernel specifies a serial code running on one processing core. 

Kernels are divided into two categories: compute and dataflow. 
Compute kernels perform computations and have access to the respective functional units of processing cores. 
Dataflow kernels perform data transfers and have access to NoC.

Each processing core can concurrently run up to three kernels. 
These include two dataflow kernels and one compute kernel. 
The two dataflow kernels are referred to as read and write kernels. 
The compute kernel is referred to as the math kernel. 
The compute kernel is optional and may be omitted in some cases. 
The program does not need to utilize all processing cores; some cores may execute no kernels at all.

Conventionally, the kernels running on one processing core form a pipeline. 
The read kernel collects input data and transmits it to the math core. 
The math kernel performs computations and transfers results to the write kernel. 
The write kernel is responsible for storing the results.

NOTE: These conventional roles can be modified in more complex scenarios. 
Capabilities of both dataflow kernels are fully symmetric, therefore, for example, 
the write kernel can be also used for collecting some of input data.

Each kernel in the program is defined on a grid of processing cores. 
The grid represents a union of distinct rectangular regions in the logical coordinate space. 
Such regions are referred to as core ranges. 
When the program is executed, every processing core in the grid runs an instance of the kernel.

Device programs use certain memory resources that can be global or local. 
Global resources are defined for the entire device and persist across program invocations. 
Local resources are attached to individual processing cores and their lifetimes are limited 
by a single program invocation. 
Each local resource is defined on a grid of processing cores so that each core owns 
a resource instance (typically occupying a region in its L1 memory).

Global resources include global buffers. Local resources include local buffers, pipes, and semaphores.

All resources are created and managed by the host program. 
The host program passes the resource descriptors to kernels as arguments of kernel main functions.


## 3 Programming interface

This chapter specifies the programming interface for Tanto device kernels.

### 3.1 Kernel structure

Tanto uses a C++-like programming language for implementing kernels. 
This language is based on a representative subset of C++ with some extensions. 
The extensions include programming constructs for build-in scalar types, compile time parameters, 
and built-in object types.

Each kernel is coded in a separate source file. 
The kernel code contains the kernel main function and, optionally, further global declarations 
of functions, constants, and data types.

Tanto framework includes a specialized kernel compiler that translates kernel source files 
into the binary executable format.

The host program supplies a reference to the kernel source file at creation of the host kernel object. 
Tanto runtime invokes the kernel compiler to process the kernel source code and 
transfers the compiled executable to the compute device before the first execution of the kernel.

### 3.1.1 Built-in scalar types

Tanto introduces several built-in scalar types.

The following types are the aliases to the standard C++ types defined in the `cstdint` header file:

- `int8`: same as `int8_t`
- `int16`: same as `int16_t`
- `int32`: same as `int32_t`
- `int64`: same as `int64_t`
- `uint8`: same as `uint8_t`
- `uint16`: same as `uint16_t`
- `uint32`: same as `uint32_t`
- `uint64`: same as `uint64_t`

The built-in type `float16` denotes a 16-bit floating point value represented in 
the IEEE half-precision floating point format (1-bit sign, 5-bit exponent, and 10-bit fraction). 

The built-in type `bfloat16` denotes a 16-bit floating point value represented in 
the bfloat16 floating point format (1-bit sign, 8-bit exponent, and 7-bit fraction).

### 3.1.2 Main function

Each kernel has a main function representing a kernel entry point. 
When a device program is launched, all its kernel instances are launched concurrently. 
Execution of each kernel is started by execution of its main function and 
is finished upon return from the main function.

The kernel main function must have a predefined name `kernel`. 
The main function does not return any value (that is, its return type must be `void`).

Parameters of the main function correspond to the arguments passed by the host program 
to the kernel at launch time. 
These arguments may include the built-in objects describing the global buffers, local buffers, pipes, 
and semaphores used by the kernel as well as numeric kernel configuration data. 
The host program attaches the lists of arguments to all kernels in the device program before launching.

### 3.1.3 Compile time parameters

Compile time parameters specify the scalar values that are known at kernel compile time. 
Compile type parameters are denoted using a built-in type represented by this pseudo-class declaration:

```
template<typename T> class param;
```

where the template parameter has the following meaning:

`T       ` parameter scalar type

Compile time parameters are declared at the global scope of kernels as variables of built-in type `param`, 
for example:

```
param<uint32> block_size;
```

The actual values of compile time parameters are passed to the Tanto kernel compiler. 
The compiler replaces parameter declarations with corresponding constant declarations and 
performs certain optimizations of generated code.

### 3.1.4 Built-in object types

Tanto extends the host C++ language with several built-in types that represent various resources and 
functional units available for device programs. 
Syntax and semantics of the built-in types are like those of the regular C++ classes; 
however, their implementation may differ. 
These types will be therefore specified as "pseudo-classes" in this document. 
Like the regular C++ classes, objects of built-in types can be constructed and destroyed. 
The functionality of these objects is exposed via method sets. 
Most built-in types are parameterized like the regular C++ template classes.

The internal structure of built-in objects is not exposed to the programmer. 
It is assumed that each built-in object holds a reference to an instance of the respective program resource. 
When the object is copied via an assignment statement or copy constructor, this reference is copied; 
therefore, copying built-in objects never causes creation of a new resource instance.

There are the following built-in types:

- `global`: global buffers
- `local`: local buffers
- `pipe`: pipes
- `semaphore`: semaphores
- `math`: math object

Global buffers represent allocations in DRAM that are persistent over device program invocations.
Global buffers are split into pages that are allocated on all available DRAM banks 
in the round-robin fashion.

Local buffers represent contiguous blocks of data allocated in L1 memory of each core of a specified grid.

Pipes represent contiguous blocks of data allocated in L1 memory of each core of a specified grid. 
Pipes are organized as FIFO data structures and are used for synchronized sharing data between kernels running on the same core. 

Semaphores represent 4-byte data blocks of data allocated in L1 memory of each core of a specified grid. 
Data in these blocks are interpreted as 32-bit unsigned integers that are used for 
synchronization between kernels running on different processing cores.

Math built-in type is an abstraction representing the hardware computational units of processing cores. 
To perform most of the computations, a kernel must instantiate a math object. 
Respective computation primitives are implemented as methods of the built-in math type.
As an exception, a few computation primitives do not require an instance of a math object and are
implemented as global functions. These primitives implement tilize and untilize operations.

The following sections specify interfaces for all built-in object types. 
This specification uses the following common conventions.

The term "this core" refers to a processing core running a kernel that invoked a given method.

The term "this object" (where "object" can be a global buffer, a local buffer, a pipe, or a semaphore) 
refers to an object on which a given method is invoked.

### 3.2 Global buffers

Global buffers represent allocations in DRAM that are persistent over device program invocations. 
Global buffers are split into pages that are allocated on all available DRAM banks 
in the round-robin fashion. 
For each global buffer a scalar type of its elements must be specified. 
The global buffer page size is set at creation of the global buffer. 
It represents the number of elements in the page that must be the power of two. 
The total number of elements in a global buffer is also specified at buffer creation.

Kernels cannot access individual elements of global buffers. 
Instead, data chunks shall be copied between global buffers and built-in local L1 objects 
(that is, local buffers or pipes) via NoC using methods of respective built-in types.

The built-in type denoting global buffers is represented by this pseudo-class declaration:

```
template<typename T> class global;
```

where the template parameter has the following meaning:

`T       ` element scalar type

Built-in `global` type has no methods; however, objects of this type can be passed as arguments 
to methods of some other built-in types.

Global buffers can be referenced by dataflow kernels only.

Global buffers are always created at the host side; therefore, they must be passed as arguments 
to the main functions of kernels that need to access them.

### 3.3 Local buffers

### 3.3.1 Overview

Local buffers represent contiguous blocks of data allocated in L1 memory of each core of a specified grid. 
Each core in the grid owns an instance of the local buffer. 
All instances of a local buffer have the same size and local address in L1. 
For each local buffer a scalar type of its elements must be specified. 
Local buffers are not persistent over device program invocations.

Data chunks can be copied between local buffers and other built-in memory objects 
(that is, global buffers, local buffers, or pipes) via NoC using methods of respective built-in types.

The built-in type denoting local buffers is represented by this pseudo-class declaration:

```
template<typename T> class local;
```

where the template parameter has the following meaning:

`T       ` element scalar type

Local buffers can be referenced by dataflow kernels only.
Local buffers are always created at the host side; 
therefore, they must be passed as arguments to the main functions of kernels that need to access them.

### 3.3.2 Interface

This section specifies a programming interface of the built-in type `local` represented as a set of methods. 
Identifier `T` represents the scalar element type of a local buffer object.

All read and write operations related to local buffers are asynchronous, that is, 
the respective methods only initiate these operations and do not block until their completion. 
Barrier synchronization functions should be used to block kernel execution until completion 
of previously initiated read and write operations.

Parameters that denote indices and offsets in access, read, and write methods are expressed 
in element units (rather than bytes).

```
T local<T>::get(uint32 index);
```

Returns value of an element of this buffer at a given index.

`index   ` element index

```
void local<T>::set(uint32 index, T value);
```

Sets an element of this buffer at a given index to a given value.

`index   ` element index<br>
`value   ` element value

```
void local<T>::read(
    uint32 dst_offset, 
    global<T> src, 
    uint32 src_offset, 
    uint32 count);
```

Reads a contiguous data chunk from a global buffer to this local buffer.

`dst_offset      ` chunk offset in the destination buffer<br>
`src source      ` buffer<br>
`src_offset      ` chunk offset in the source buffer<br>
`count           ` number of elements in the chunk

```
void local<T>::read(
    uint32 dst_offset, 
    local<T> src, 
    uint32 src_offset, 
    uint32 count);
```

Reads a contiguous data chunk from a local buffer on this core to this local buffer.

`dst_offset      ` chunk offset in the destination buffer<br>
`src             ` source buffer<br>
`src_offset      ` chunk offset in the source buffer<br>
`count           ` number of elements in the chunk

```
void local<T>::read(
    uint32 dst_offset, 
    local<T> src, 
    uint32 src_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Reads a contiguous data chunk from a local buffer on a remote core to this local buffer. 

`dst_offset      ` chunk offset in the destination buffer<br>
`src             ` source buffer<br>
`src_offset      ` chunk offset in the source buffer<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void local<T>::read(
    uint32 dst_offset, 
    pipe<T> src, 
    uint32 src_offset, 
    uint32 count);
```

Reads a contiguous data chunk from the read frame of a pipe on this core to this local buffer.

`dst_offset      ` chunk offset in the destination buffer<br>
`src             ` source pipe<br>
`src_offset      ` chunk offset in the read frame of the source pipe<br>
`count           ` number of elements in the chunk

```
void local<T>::read(
    uint32 dst_offset, 
    pipe<T> src, 
    uint32 src_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Reads a contiguous data chunk from the read frame of a pipe on a remote core to this local buffer. 
A grid on which the source pipe is defined must include this core.

`dst_offset      ` chunk offset in the destination buffer<br>
`src             ` source pipe<br>
`src_offset      ` chunk offset in the read frame of the source pipe<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void local<T>::write(
    uint32 src_offset, 
    global<T> dst, 
    uint32 dst_offset, 
    uint32 count);
```

Writes a contiguous data chunk from this local buffer to a global buffer.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk

```
void local<T>::write(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count);
```

Writes a contiguous data chunk from this local buffer to a local buffer on this core.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk

```
void local<T>::write(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Writes a contiguous data chunk from this local buffer to a local buffer on a remote core.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void local<T>::write(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count);
```

Writes a contiguous data chunk from this local buffer to the write frame of a pipe on this core.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk

```
void local<T>::write(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Writes a contiguous data chunk from this local buffer to the write frame of a pipe on a remote core. 
A grid on which the destination pipe is defined must include this core.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void local<T>::write_mcast(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from this local buffer to all instances of a local buffer 
located within a specified core range. 
If this core belongs to the specified range, writing is not applied to the corresponding instance 
of the destination buffer.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination buffer instances

```
void local<T>::write_mcast_with_self(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from this local buffer to all instances of a local buffer located 
within a specified core range. 
If this core belongs to the specified range, writing is applied to the corresponding instance 
of the destination buffer.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination buffer instances

```
void local<T>::write_mcast(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from this local buffer to write frames of all instances 
of a pipe located within a specified core range. 
If this core belongs to the specified range, writing is not applied to the corresponding instance 
of the destination pipe. 
A grid on which the destination pipe is defined must include this core.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination pipe instances

```
void local<T>::write_mcast_with_self(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from this local buffer to write frames of all instances 
of a pipe located within a specified core range. 
If this core belongs to the specified range, writing is applied to the corresponding instance 
of the destination pipe. 
A grid on which the destination pipe is defined must include this core.

`src_offset      ` chunk offset in the source buffer<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination pipe instances

```
void local<T>::move_init(uint32 count);
```

Initializes a context for moving data between L1 memory regions on this core
and associates the context with this local buffer and the specified number of elements.
A call to this method is supposed to be followed by multiple calls to the method `move`
for this local buffer. The move context remains valid until a call to any
function involving data transfer across NoC other than `move` for this local buffer.

The `move` method is semantically similar to `read` but is significantly more
efficient when a series of L1 data transfers of the same chunk size needs 
to be performed.

`count   ` number of elements to move

```
void local<T>::move(
    uint32 dst_offset, 
    local<T> src, 
    uint32 src_offset);
```

Reads a contiguous data chunk from a local buffer on this core to this local buffer.
Requires a valid move context associated with this local buffer initialized by 
a prior call to `move_init`. The number of elements in the chunk is defined by this context.

`dst_offset      ` chunk offset in the write frame of the destination buffer<br>
`src             ` source buffer<br>
`src_offset      ` chunk offset in the source buffer

```
void local<T>::move(
    uint32 dst_offset, 
    pipe<T> src, 
    uint32 src_offset);
```

Reads a contiguous data chunk from the read frame of a pipe on this core to this local buffer.
Requires a valid move context associated with this local buffer initialized by 
a prior call to `move_init`. The number of elements in the chunk is defined by this context.

`dst_offset      ` chunk offset in the write frame of the destination buffer<br>
`src             ` source pipe<br>
`src_offset      ` chunk offset in the read frame of the source pipe

### 3.4 Pipes

### 3.4.1 Overview

Pipes represent contiguous blocks of data allocated in L1 memory of each core of a specified grid. 
Pipes are organized as FIFO data structures and are used for synchronized sharing data 
between kernels running on the same core. 
Each core in the grid owns an instance of the pipe. 
All instances of a pipe have the same size and local address in L1. 
For each pipe a scalar type of its elements must be specified. 
Pipes are not persistent over device program invocations.

A pipe holds a sequence of data blocks of 1024 elements each. 
These blocks may contain genuine 2D tiles of 32 x 32 size but can also hold data chunks with any other semantics. 
For simplicity, these blocks will be referred to as tiles in this section.

Each pipe contains two distinct regions of the same size referred to as read frame and write frame. 
Kernels use these frames for reading and writing data respectively. 
The number of tiles in each frame is referred to as frame size. 
The initial frame size is specified during the pipe creation. 
If necessary, the frame size can be changed by the kernel using the dedicated pipe method. 
The programming model maintains the read and write pointers that correspond to start addresses 
of the respective frame memory regions. 
The dedicated pipe methods are provided for synchronized advancement of these pointers.

Each pipe belongs to one of three categories: input, output, or intermediate. 
Input pipes provide write access to dataflow kernels and read access to compute and dataflow kernels. 
Output pipes provide write access to compute and dataflow kernels and read access to dataflow kernels. 
Intermediate pipes provide read and write access to compute kernels only. 
The pipe category is determined based on its usage in the program and is not explicitly reflected 
in the pipe declaration in the kernel code.

Data chunks can be copied between pipes and other built-in memory objects 
(that is, global buffers, local buffers, or pipes) via NoC using methods of respective built-in types.

The built-in type denoting pipes is represented by this pseudo-class declaration:

```
template<typename T> class pipe;
```

where the template parameter has the following meaning:

`T       ` element scalar type

Pipes can be referenced by compute and dataflow kernels.

Pipes are always created at the host side; 
therefore, they must be passed as arguments to the main functions of kernels that need to access them.

### 3.4.2 Interface

This section specifies a programming interface of the built-in type `pipe` represented as a set of methods. 
Identifier T represents the scalar element type of a pipe object.

All read and write operations related to pipes are asynchronous, 
that is, the respective methods only initiate these operations and do not block until their completion. 
Barrier synchronization functions should be used to block kernel execution until completion 
of previously initiated read and write operations.

Parameters that denote indices and offsets in read and write methods are expressed in element units 
(rather than bytes).

```
void pipe<T>::set_frame(uint32 tiles);
```

Sets size of read and write frames of this pipe to a given number of tiles.

`tiles   ` frame size in tiles

```
void pipe<T>::reserve_back();
```

Blocks until free space becomes available in this pipe for a new write frame. 
Sets the write frame pointer to the beginning of this space.

```
void pipe<T>::push_back();
```

Marks data in the current write frame as available for reading. Invalidates the write frame pointer.

```
void pipe<T>::wait_front();
```

Blocks until a new frame filled with data becomes available in this pipe. 
Sets the read frame pointer to the beginning of this space.

```
void pipe<T>::pop_front();
```

Marks the current read frame as available for writing. Invalidates the read frame pointer.

```
void pipe<T>::read(
    uint32 dst_offset, 
    global<T> src, 
    uint32 src_offset, 
    uint32 count);
```

Reads a contiguous data chunk from a global buffer to the write frame of this pipe.

`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`src             ` source buffer<br>
`src_offset      ` chunk offset in the source buffer<br>
`count           ` number of elements in the chunk

```
void pipe<T>::read(
    uint32 dst_offset, 
    local<T> src, 
    uint32 src_offset, 
    uint32 count);
```

Reads a contiguous data chunk from a local buffer on this core to the write frame of this pipe.

`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`src             ` source buffer<br>
`src_offset      ` chunk offset in the source buffer<br>
`count           ` number of elements in the chunk

```
void pipe<T>::read(
    uint32 dst_offset, 
    local<T> src, 
    uint32 src_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Reads a contiguous data chunk from a local buffer on a remote core to the write frame of this pipe. 
A grid on which the source buffer is defined must include this core.

`dst_offset      ` chunk offset in the write frame the destination pipe<br>
`src             ` source buffer<br>
`src_offset      ` chunk offset in the source buffer<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void pipe<T>::read(
    uint32 dst_offset, 
    pipe<T> src, 
    uint32 src_offset, 
    uint32 count);
```

Reads a contiguous data chunk from the read frame of a pipe on this core to the write frame of this pipe.

`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`src             ` source pipe<br>
`src_offset      ` chunk offset in the read frame of the source pipe<br>
`count           ` number of elements in the chunk

```
void pipe<T>::read(
    uint32 dst_offset, 
    pipe<T> src, 
    uint32 src_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Reads a contiguous data chunk from the read frame of a pipe on a remote core to the write frame of this pipe.

`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`src             ` source pipe<br>
`src_offset      ` chunk offset in the read frame of the source pipe<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void pipe<T>::write(
    uint32 src_offset, 
    global<T> dst, 
    uint32 dst_offset, 
    uint32 count);
```

Writes a contiguous data chunk from the read frame of this local pipe to a global buffer.

`src_offset      ` chunk offset on the write frame of the source pipe<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk

```
void pipe<T>::write(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count);
```

Writes a contiguous data chunk from the read frame of this pipe to a local buffer on this core.

`src_offset      ` chunk offset in the read frame of the source pipe<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk

```
void pipe<T>::write(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Writes a contiguous data chunk from the read frame of this pipe to a local buffer on a remote core.

`src_offset      ` chunk offset in read frame of the source pipe<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void pipe<T>::write(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count);
```

Writes a contiguous data chunk from the read frame of this pipe to the write frame of a pipe on this core.

`src_offset      ` chunk offset in the read frame of the source pipe<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk

```
void pipe<T>::write(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count, 
    uint32 x, 
    uint32 y);
```

Writes a contiguous data chunk from the read frame of this pipe to the write frame of a pipe on a remote core. 
A grid on which the destination pipe is defined must include this core.

`src_offset      ` chunk offset in the read frame of the source pipe<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk<br>
`x               ` physical x coordinate of the remote core<br>
`y               ` physical y coordinate of the remote core

```
void pipe<T>::write_mcast(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from the write frame of this pipe to all instances of a local buffer 
located within a specified core range. 
If this core belongs to the specified range, writing is not applied to the corresponding instance 
of the destination buffer.

`src_offset      ` chunk offset in the read frame of the source pipe<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination buffer instances

```
void pipe<T>::write_mcast_with_self(
    uint32 src_offset, 
    local<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from the write frame of this pipe to all instances of a local buffer 
located within a specified core range. 
If this core belongs to the specified range, writing is applied to the corresponding instance 
of the destination buffer.

`src_offset      ` chunk offset in the read frame of the source pipe<br>
`dst             ` destination buffer<br>
`dst_offset      ` chunk offset in the destination buffer<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination buffer instances

```
void pipe<T>::write_mcast(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from the write frame of this pipe to write frames of all instances of a pipe 
located within a specified core range. 
If this core belongs to the specified range, writing is not applied to the corresponding instance 
of the destination pipe. A grid on which the destination pipe is defined must include this core.

`src_offset      ` chunk offset in read frame of the source pipe<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination pipe instances

```
void pipe<T>::write_mcast_with_self(
    uint32 src_offset, 
    pipe<T> dst, 
    uint32 dst_offset, 
    uint32 count,
    uint32 x_start, 
    uint32 y_start, 
    uint32 x_end, 
    uint32 y_end, 
    uint32 num_dests);
```

Writes a contiguous data chunk from the write frame of this pipe to write frames of all instances of a pipe 
located within a specified core range. 
If this core belongs to the specified range, writing is applied to the corresponding instance 
of the destination pipe. A grid on which the destination pipe is defined must include this core.

`src_offset      ` chunk offset in the read frame of the source pipe<br>
`dst             ` destination pipe<br>
`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`count           ` number of elements in the chunk<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical start x coordinate of the destination range<br>
`y_end           ` physical start y coordinate of the destination range<br>
`num_dests       ` number of destination pipe instances

```
void pipe<T>::move_init(uint32 count);
```

Initializes a context for moving data between L1 memory regions on this core and
associates the context with this pipe and the specified number of elements.
A call to this method is supposed to be followed by multiple calls to the method `move`
for this pipe. The move context remains valid until a call to any
function involving data transfer across NoC other than `move` for this pipe.

The `move` method is semantically similar to `read` but is significantly more
efficient when a series of L1 data transfers of the same chunk size needs 
to be performed.

`count   ` number of elements to move

```
void pipe<T>::move(
    uint32 dst_offset, 
    local<T> src, 
    uint32 src_offset);
```

Reads a contiguous data chunk from a local buffer on this core to the write frame of this pipe.
Requires a valid move context associated with this pipe initialized by 
a prior call to `move_init`. The number of elements in the chunk is defined by this context.

`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`src             ` source buffer<br>
`src_offset      ` chunk offset in the source buffer

```
void pipe<T>::move(
    uint32 dst_offset, 
    pipe<T> src, 
    uint32 src_offset);
```

Reads a contiguous data chunk from the read frame of a pipe on this core to the write frame of this pipe.
Requires a valid move context associated with this pipe initialized by 
a prior call to `move_init`. The number of elements in the chunk is defined by this context.

`dst_offset      ` chunk offset in the write frame of the destination pipe<br>
`src             ` source pipe<br>
`src_offset      ` chunk offset in the read frame of the source pipe

### 3.5 Semaphores

### 3.5.1 Overview

Semaphores represent 4-byte data blocks of data allocated in L1 memory of each core of a specified grid. 
Data in these blocks are interpreted as 32-bit unsigned integers that are used for synchronization 
between kernels running on different processing cores. 
Each core in the grid owns an instance of the semaphore. 
All instances of a semaphore have the same local address in L1. 
Semaphores are not persistent over device program invocations.

The built-in type denoting semaphores is represented by this pseudo-class declaration:

```
class semaphore;
```

Semaphores can be referenced by dataflow kernels only.

Semaphores are always created at the host side; 
therefore, they must be passed as arguments to the main functions of kernels that need to access them.

### 3.5.2 Interface

This section specifies a programming interface of built-in type `semaphore` represented as a set of methods.

All remote semaphore setting and increment operations are asynchronous, 
that is, the respective methods only initiate these operations and do not block until their completion. 
Barrier synchronization functions should be used to block kernel execution until completion 
of previously initiated read and write operations.

```
void semaphore::set(uint32 value);
```

Sets a given value to this semaphore instance located on this core. This is a synchronous operation.

`value   ` new semaphore value

```
void semaphore::set_remote(
    semaphore src, 
    uint32 x, 
    uint32 y);
```

Sets a value of a semaphore instance located on this core to an instance of this semaphore located on a remote core.

`src     ` source semaphore instance<br>
`x       ` physical x coordinate of the remote core<br>
`y       ` physical y coordinate of the remote core

```
void semaphore::set_mcast(
    semaphore src,
    uint32 x_start,
    uint32 y_start,
    uint32 x_end,
    uint32 y_end,
    uint32 num_dests);
```

Sets a value of a semaphore instance located on this core to all instances of this semaphore located 
within a specified core range. 
If this core belongs to the specified range, writing is not applied to the corresponding instance 
of the destination semaphore.

`src             ` source semaphore instance<br>
`x_start         ` physical start x coordinate of the destination range<br>
`y_start         ` physical start y coordinate of the destination range<br>
`x_end           ` physical end x coordinate of the destination range<br>
`y_end           ` physical end y coordinate of the destination range<br>
`num_dests       ` number of destination semaphore instances

```
void semaphore::inc(uint32 x, uint32 y, uint32 value);
```

Atomically increments a value of an instance of this semaphore located on a remote core by a given value.

`x       ` physical x coordinate of the remote core<br>
`y       ` physical y coordinate of the remote core<br>
`value   ` increment value

```
void semaphore::wait(uint32 value);
```

Blocks until a value of an instance of this semaphore located on this core becomes equal to a given requested value.

`value   ` requested semaphore value


### 3.6 Math object

### 3.6.1 Overview

Math built-in type is an abstraction representing the hardware computational units of processing cores. 
To perform any computations, a kernel must instantiate a math object. 
Most computation primitives are implemented as methods of the built-in math type. 
Each math object is parameterized with a compute scalar type that specifies the type 
of operands and results used by the computational units. 

At most one instance of a math object may exist at any moment during the kernel execution.

The math object exposes a linear array of destination slots. 
Each destination slot contains 1024 elements of the specified compute scalar type. 
The number of available destination slots depends on the processor architecture and 
the size of the compute scalar type. (For example, on the Wormhole architecture, 
the destination array contains 8 slots for 16-bit types and 4 slots for 32-bit types.)

Destination slots are used as a fast memory for storing computation results. 
(They are implemented as regions of register files of the hardware computational units.)

Data in the destination slot array are initialized to zero upon instantiation of the math object. 
Data in the destination slot array are invalidated upon destruction of the math object 
(that is, once the kernel execution leaves the static scope in which the math object was instantiated).

The built-in type denoting math objects is represented by this pseudo-class declaration:

```
template<typename T> class math;
```

where the template parameter has the following meaning:

`T       ` compute scalar type

Math objects can be instantiated and referenced by compute kernels only.

### 3.6.2 Interface

This section specifies a programming interface of built-in type `math` represented as a set of methods.

### 3.6.2.1 Pack

```
template<typename U> 
void math<T>::pack(uint32 isrc, pipe<U> dst);
```

Packs a tile located in a specified slot of the destination array and writes it 
into the next free slot of the write frame of a given pipe. 
The next free slot pointer is set to the first tile of the pipe write frame when a new write frame is reserved. 
This pointer is automatically moved by one tile upon each call or pipe operation.

`isrc    ` index of a destination slot containing the source tile<br>
`dst     ` pipe for the result

```
template<typename U> 
void math<T>::pack_row(uint32 isrc, pipe<U> dst);
```

Packs the first row of a tile located in a specified slot of the destination array and writes it 
into the next free slot of the write frame of a given pipe. 
The next free slot pointer is set to the first tile of the pipe write frame when a new write frame is reserved. 
This pointer is automatically moved by one tile upon each call or pipe operation.

Elements in the destination outside the first tile row are not affected.
This method is typically used to pack results of row reduction.

`isrc    ` index of a destination slot containing the source tile<br>
`dst     ` pipe for the result

```
template<typename U> 
void math<T>::pack_col(uint32 isrc, pipe<U> dst);
```

Packs the first column of a tile located in a specified slot of the destination array and writes it 
into the next free slot of the write frame of a given pipe. 
The next free slot pointer is set to the first tile of the pipe write frame when a new write frame is reserved. 
This pointer is automatically moved by one tile upon each call or pipe operation.

Elements in the destination outside the first tile column are not affected.
This method is typically used to pack results of column reduction.

`isrc    ` index of a destination slot containing the source tile<br>
`dst     ` pipe for the result

```
template<typename U> 
void math<T>::pack_scalar(uint32 isrc, pipe<U> dst);
```

Packs the first element of a tile located in a specified slot of the destination array and writes it 
into the next free slot of the write frame of a given pipe. 
The next free slot pointer is set to the first tile of the pipe write frame when a new write frame is reserved. 
This pointer is automatically moved by one tile upon each call or pipe operation.

Elements in the destination other than the first tile element are not affected.
This method is typically used to pack results of scalar reduction.

`isrc    ` index of a destination slot containing the source tile<br>
`dst     ` pipe for the result

### 3.6.2.2 Copy

```
template<typename U>
void math<T>::copy(pipe<U> src, uint32 isrc, uint32 idst);
```

Copies a tile located at a given index of the read frame of a given pipe 
to a specified slot of the destination array.

`src     ` pipe containing the source tile<br>
`isrc    ` index of the source tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

### 3.6.2.3 Elementwise binary

Elementwise binary operations perform elementwise addition, subtraction, or multiplication of two tiles. 
The operand tiles are located at specified indices in the read frames of two pipes. 
A tile containing the operation result is stored in a specified slot of the destination array.

Elementwise binary operations are described using this pseudocode:

```
dst[h, w] = src0[h, w] OP src1[h, w]
```

where 

`OP` is one of `+`, `-`, `*`<br>
`h` in `[0 .. 31]`, `w` in `[0 .. 31]`

```
template<typename U, typename V>
void math<T>::add(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs elementwise addition of two tiles.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::sub(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs elementwise subtraction of two tiles.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::mul(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs elementwise multiplication of two tiles.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

### 3.6.2.4 Broadcasted binary

Elementwise binary operations perform addition, subtraction, or multiplication of two tiles. 
The first row, the first column, or the first element of the second tile is used as the second operand. 
Its content is appropriately broadcasted to match a shape of the first operand. 
The operand tiles are located at specified indices in the read frames of two pipes. A tile containing the operation result is stored in the specified slot of the destination array.

Broadcasted binary operations are described using this pseudocode:

Row broadcast:

```
dst[h, w] = src0[h, w] OP src1[0, w]
```

Column broadcast:

```
dst[h, w] = src0[h, w] OP src1[h, 0]
```

Scalar broadcast:

```
dst[h, w] = src0[h, w] OP src1[0, 0]
```

where

`src0` and `src1` are the operand tiles<br>
`dst` is the result tile<br>
`OP` is one of `+`, `-`, `*`<br> 
`h` in `[0 .. 31]`, `w` in `[0 .. 31]`

```
template<typename U, typename V>
void math<T>::add_bcast_rows(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs addition of two tiles with row broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::sub_bcast_rows(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs subtraction of two tiles with row broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::mul_bcast_rows(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs multiplication of two tiles with row broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::add_bcast_cols(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs addition of two tiles with column broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::sub_bcast_cols(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs subtraction of two tiles with column broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::mul_bcast_cols(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs multiplication of two tiles with column broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::add_bcast_scalar(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs addition of two tiles with scalar broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::sub_bcast_scalar(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs subtraction of two tiles with scalar broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::mul_bcast_scalar(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);
```

Performs multiplication of two tiles with scalar broadcast.

`src0    ` pipe containing the first tile<br>
`src1    ` pipe containing the second tile<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

### 3.6.2.5 Matrix multiply

Matrix multiply operation performs multiplication of two tiles as 32 x 32 matrices. 
The operand tiles are located at specified indices in the read frames of two pipes. 
A tile containing the operation result is accumulated in the specified slot of the destination array.

Elementwise binary operations are described using this pseudocode:

```
dst[h, w] += SUM[i](src0[h, i] * src1[i, w])

```

where 

`SUM[i](...)` is sum operator over index `i`<br>
`h` in `[0 .. 31]`, `w` in `[0 .. 31]`, `i` in `[0 .. 31]` 

```
template<typename U, typename V>
void math<T>::matmul(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst,
    bool transpose);
```

Performs matrix multiplication of two tiles.

`src0            ` pipe containing the first tile<br>
`src1            ` pipe containing the second tile<br>
`isrc0           ` index of the first tile in the pipe read frame<br>
`isrc1           ` index of the second tile in the pipe read frame<br>
`idst            ` index of a destination slot for the result<br>
`transpose       ` if set to true, transpose the second tile

### 3.6.2.6 Reduce

Reduce operations reduce contents of the first tile operand in horizontal direction (row reduce), 
vertical direction (column reduce), or both directions (scalar reduce). 
Reduction computes either sum or maximum of reduced values. 
Result of reduction is scaled by the value provided by the first element of the second tile operand and 
accumulated in the specified slot of the destination array. 
Result of reduction forms the first column (row reduce), the first row (column reduce) or 
the first element (scalar reduce) in the result tile. 
The values of other elements in the result tile are undefined.

Reduce operations are described using this pseudocode:

Row sum reduce:

```
dst[h, 0] += SUM[w](src0[h, w]) * src1[0, 0]
```

Column sum reduce:

```
dst[0, w] += SUM[h](src0[h, w]) * src1[0, 0]
```

Scalar sum reduce:

```
dst[0, 0] += SUM[h, w](src0[h, w]) * src1[0, 0]
```

Row maximum reduce:

```
dst[h, 0] = max(dst[h, 0], MAX[w](src0[h, w]) * src1[0, 0])
```

Column maximum reduce:

```
dst[0, w] = max(dst[0, w], MAX[h](src0[h, w]) * src1[0, 0])
```

Scalar maximum reduce:

```
dst[0, 0] = max(dst[0, 0], MAX[h, w](src0[h, w]) * src1[0, 0])
```

where

`src0` and `src1` are the operand tiles<br>
`dst` is the result tile<br>
`SUM[i](...)` is sum operator over index `i`<br>
`MAX[i](...)` is maximum operator over index `i`<br>
`max(a, b)` computes maximum of its scalar operands<br>
`h` in `[0 .. 31]`, `w` in `[0 .. 31]`

```
template<typename U, typename V>
void math<T>::reduce_max_rows(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);    
```

Performs row maximum reduce.

`src0    ` pipe containing the first operand<br>
`src1    ` pipe containing the second operand<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::reduce_max_cols(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);    
```

Performs column maximum reduce.

`src0    ` pipe containing the first operand<br>
`src1    ` pipe containing the second operand<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::reduce_max_scalar(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);    
```

Performs scalar maximum reduce.

`src0    ` pipe containing the first operand<br>
`src1    ` pipe containing the second operand<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::reduce_sum_rows(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);    
```

Performs row sum reduce.

`src0    ` pipe containing the first operand<br>
`src1    ` pipe containing the second operand<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::reduce_sum_cols(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);    
```

Performs column sum reduce.

`src0    ` pipe containing the first operand<br>
`src1    ` pipe containing the second operand<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

```
template<typename U, typename V>
void math<T>::reduce_sum_scalar(
    pipe<U> src0, 
    pipe<V> src1, 
    uint32 isrc0, 
    uint32 isrc1, 
    uint32 idst);    
```

Performs scalar sum reduce.

`src0    ` pipe containing the first operand<br>
`src1    ` pipe containing the second operand<br>
`isrc0   ` index of the first tile in the pipe read frame<br>
`isrc1   ` index of the second tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

### 3.6.2.7 Transpose

Transpose operation transposes content of a tile located at a specified index in the read frame 
of a specified pipe, places result in a specified slot of the destination array.

Transpose operation is described using this pseudocode:

```
dst[h, w] = src[w, h]
```

where 

`h` in `[0 .. 31]`, `w` in `[0 .. 31]`

```
template<typename U>
void math<T>::transpose(pipe<U> src, uint32 isrc, uint32 idst);
```

Transposes a tile.

`src     ` pipe containing the source tile<br>
`isrc    ` index of the source tile in the pipe read frame<br>
`idst    ` index of a destination slot for the result

### 3.6.2.8 Elementwise unary

Elementwise unary operations apply transformations to all elements of a tile operand. 
Both the operand and result are stored in the same specified slot of the destination array.

Elementwise unary operations are described using this pseudocode:

```
dst[h, w] = OP(dst[h, w])
```

where 

`OP` is an unary scalar function<br>
`dst` is the location of both source and result tiles<br>
`h` in `[0 .. 31]`, `w` in `[0 .. 31]`

Methods for all elementwise unary operations require an index of a slot in the destination array 
as their first parameter:

`idst    ` destination slot index for the source and result tiles

```
void math<T>::abs(uint32 idst);
```

Computes the absolute value for each element of a tile operand.

```
void math<T>::acos(uint32 idst);
```

Computes the arccosine function for each element of a tile operand.

```
void math<T>::add_scalar(uint32 dst, uint32 param);
```

Adds a scalar value to each element of a tile operand.

```
add_scalar(x) = x + scalar
```

The scalar value is passed as the second method parameter:

`param   ` scalar value

```
void math<T>::asin(uint32 idst);
```

Computes the arcsine function for each element of a tile operand.

```
void math<T>::atan(uint32 idst);
```

Computes the arctangent function for each element of a tile operand.

```
void math<T>::cos(uint32 idst);
```

Computes the cosine function for each element of a tile operand.

```
void math<T>::div_scalar(uint32 idst, uint32 param);
```

Divides each element of a tile operand by a scalar value.

```
div_scalar(x) = x / scalar
```

The scalar value is passed as the second method parameter:

`param   ` scalar value

```
void math<T>::elu(uint32 idst, uint32 param);
```

Computes the Exponential Linear Unit (ELU) function for each element of a tile operand.

The ELU function is defined as:

```
elu(x) = (x <= 0) ? slope * (exp(x) - 1) : x
```

The slope is passed as the second method parameter:

`param   ` ELU slope

```
void math<T>::eqz(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element is equal to zero and 0 otherwise.

```
eqz(x) = (x == 0) ? 1 : 0
```

```
void math<T>::erf(uint32 idst);
```

Computes the error function for each element of a tile operand.

```
void math<T>::erfc(uint32 idst);
```

Computes the complementary error function for each element of a tile operand.

```
void math<T>::erfinv(uint32 idst);
```

Computes the inverse error function for each element of a tile operand.

```
void math<T>::exp(uint32 idst);
```

Computes the base-e exponential for each element of a tile operand (where e is Euler’s number).

```
void math<T>::exp2(uint32 idst);
```

Computes the base-2 exponential for each element of a tile operand.

```
void math<T>::expm1(uint32 idst);
```

Computes the base-`e` exponential minus 1 for each element of a tile operand (where e is Euler’s number).

```
expm1(x) = exp(x) - 1
```

```
void math<T>::gelu(uint32 idst);
```

Computes the Gaussian Error Linear Unit (GELU) function for each element of a tile operand.

The GELU function is defined as:

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x ^ 3)))
```

```
void math<T>::gez(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element is greater than or equal to zero and 0 otherwise.

```
gez(x) = (x >= 0) ? 1 : 0
```

```
void math<T>::gtz(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element is greater than zero and 0 otherwise.

```
gtz(x) = (x > 0) ? 1 : 0
```

```
void math<T>::heaviside(uint32 idst, uint32 param);
```

Computes the Heaviside function for each element of a tile operand.

The Heaviside function is defined as:

```
heaviside(x) = (x < 0) ? 0 : (x > 0) ? 1 : param
```

The value returned for zero argument is passed as the second method parameter

`param   ` value returned for zero argument

```
void math<T>::i0(uint32 idst);
```

Computes the modified Bessel function of the first kind, order 0 for each element of a tile operand.

```
void math<T>::isfinite(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element has a finite value and 0 otherwise.

```
void math<T>::isinf(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element has an infinite value and 0 otherwise.

```
void math<T>::isnan(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element has a not-a-number (NaN) value and 0 otherwise.

```
void math<T>::isneginf(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element has a negative infinite value and 0 otherwise.

```
void math<T>::isposinf(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element has a positive infinite value and 0 otherwise.

```
void math<T>::leaky_relu(uint32 idst, uint32 param);
```

Computes the Leaky Rectified Linear Unit (LReLU) function for each element of a tile operand.

The LReLU function is defined as:

```
leaky_relu(x) = (x <= 0) ? slope * x : x
```

The slope is passed as the second method parameter:

`param   ` LReLU slope

```
void math<T>::lez(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element is less than or equal to zero and 0 otherwise.

```
lez(x) = (x <= 0) ? 1 : 0
```

```
void math<T>::log(uint32 idst);
```

Computes the natural logarithm for each element of a tile operand.

```
void math<T>::log_with_base(uint32 idst, uint32 param);
```

Computes the logarithm with a given base for each element of a tile operand.

The logarithm base is passed as the second method parameter:

`param   ` logarithm base

```
void math<T>::logical_not(uint32 idst);
```

Computes the logical NOT for each element of a tile operand.

```
void math<T>::ltz(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element is less than zero and 0 otherwise.

```
ltz(x) = (x < 0) ? 1 : 0
```

```
void math<T>::max(uint32 idst);
```

Computes elementwise maximum between two tile operands located in consecutive destination slots
with indices `idst` and `idst + 1`. Assigns the result to the first slot.

```
void math<T>::mul_scalar(uint32 idst, uint32 param);
```

Multiplies each element of a tile operand by a scalar value.

```
mul_scalar(x) = x * scalar
```

The scalar value is passed as the second method parameter:

`param   ` scalar value

```
void math<T>::nez(uint32 idst);
```

For each element of a tile operand, computes a function that returns 1 
if the element is not equal to zero and 0 otherwise.

```
nez(x) = (x != 0) ? 1 : 0
```

```
void math<T>::power(uint32 idst, uint32 param);
```

Computes the power operation for each element of a tile operand.

```
power(x) = x ^ param
```

The natural power exponent is passed as the second method parameter:

`param   ` power exponent

```
void math<T>::recip(uint32 idst);
```

Computes the reciprocal for each element of a tile operand.

```
recip(x) = 1 / x
```

```
void math<T>::relu(uint32 idst);
```

Computes the Rectified Linear Unit (ReLU) function for each element of a tile operand.

The ReLU function is defined as:

```
relu(x) = (x < 0) ? 0 : x
```

```
void math<T>::relu_max(uint32 idst, uint32 param);
```

Computes the Rectified Linear Unit (ReLU) Max function for each element of a tile operand.

The ReLU Max function is defined as:

```
relu_max(x) = (x > threshold) ? threshold : (x < 0) ? 0 : x
```

where `threshold > 0`.

The threshold is passed as the second method parameter:

`param   ` threshold

```
void math<T>::relu_min(uint32 idst, uint32 param);
```

Computes the Rectified Linear Unit (ReLU) Min function for each element of a tile operand.

The ReLU Min function is defined as:

```
relu_min(x) = (x < threshold) ? 0 : x
```

where `threshold > 0`.

The threshold is passed as the second method parameter:

`param   ` threshold

```
void math<T>::rsqrt(uint32 idst);
```

Computes the reciprocal square root for each element of a tile operand.

```
rsqrt(x) = 1 / sqrt(x)
```

```
void math<T>::rsub_scalar(uint32 idst, uint32 param);
```

Subtracts each element of a tile operand from a scalar value.

```
rsub_scalar(x) = scalar - x
```

The scalar value is passed as the second method parameter:

`param   ` scalar value

```
void math<T>::sigmoid(uint32 idst);
```

Computes the sigmoid function for each element of a tile operand.

The sigmoid function is defined as:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

```
void math<T>::sign(uint32 idst);
```

Computes the sign for each element of a tile operand.

```
sign(x) = (x < 0) ? -1 : (x > 0) ? 1 : 0
```

```
void math<T>::signbit(uint32 idst);
```

Computes the sign bit for each element of a tile operand.

```
void math<T>::sin(uint32 idst);
```

Computes the sine function for each element of a tile operand.

```
void math<T>::sqrt(uint32 idst);
```

Computes the square root for each element of a tile operand.

```
void math<T>::square(uint32 idst);
```

Computes the square for each element of a tile operand.

```
void math<T>::sub_scalar(uint32 dst, uint32 param);
```

Subtracts a scalar value from each element of a tile operand.

```
sub_scalar(x) = x - scalar
```

```
void math<T>::tan(uint32 idst);
```

Computes the tangent function for each element of a tile operand.

```
void math<T>::tanh(uint32 idst);
```

Computes the hyperbolic function for each element of a tile operand.


### 3.7 Global functions

Global built-in functions implement functionality not related to any built-in object.

### 3.7.1 Barrier synchronization

Functions in this group are used to block the kernel execution until the completion 
of previously initiated asynchronous read and write operations.
They can be used in read and write kernels only.

```
void read_barrier();
```

Blocks until completion of all read operations initiated by the calling kernel.

```
void write_barrier();
```

Blocks until completion of all write operations initiated by the calling kernel.

### 3.7.2 Math primitives

Functions in this group implement math primitives that do not require an instance of `math` object.
They can be used in math kernels only. No instance of `math` object must exist at the time
of calling these functions.

### 3.7.2.1 Tilize

Block tilize operation inputs a block of data from the read frame of a source pipe, 
transforms it into a sequence of tiles, and places result in the write frame of a destination pipe.

The input block represents a matrix with row-major layout, 32 rows and 32 x `block` columns. 
The result contains `block` tiles of 32 rows and 32 columns. Parameter `block` is referred to as block size.

```
template<typename U, typename V>
void tilize_block(pipe<U> src, uint32 block, pipe<V> dst);
```

Performs the block tilize operation.

`src     ` source pipe<br>
`block   ` block size<br>
`dst     ` destination pipe

### 3.7.2.2 Untilize

Block untilize operation inputs a sequence of tiles from the read frame of the source pipe, 
transforms it into a block of data, and places result in the write frame of the destination pipe.

The input contains `block` tiles of 32 rows and 32 columns. 
The result block represents a matrix with row-major layout, 32 rows and 32 x `block` columns. 
Parameter `block` is referred to as block size.

This is the inverse of the block tilize operation.

```
template<typename U, typename V>
void untilize_block(pipe<U> src, uint32 block, pipe<V> dst);
```

Performs the block untilize operation.

`src     ` source pipe<br>
`block   ` block size<br>
`dst     ` destination pipe


## Appendix A: Kernel programming example

This example includes three kernels (reader, math, and writer) collectively implementing 
a device part of a program that performs elementwise operation on two input global buffers 
and writes result to an output global buffer.

This Appendix is not normative.

### A.1 Reader kernel

```
void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 N,
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 start,
        uint32 stride) {
    pa.set_frame(frame_tiles);
    pb.set_frame(frame_tiles);
    uint32 frame_items = frame_tiles * 1024;
    for (uint32 n = 0; n < N; n++) {
        uint32 pos = start;
        for (uint32 i = 0; i < num_frames; i++) {
            pa.reserve_back();
            pb.reserve_back();
            pa.read(0, ga, pos, frame_items);
            pb.read(0, gb, pos, frame_items);
            read_barrier();
            pa.push_back();
            pb.push_back();
            pos += frame_items;
        }
        start += stride;
    }
}
```

### A.2 Math kernel

```
static constexpr uint32
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2;

param<uint32> op_code;

void binary_op(math<T> acc, pipe<T> pa, pipe<T> pb, uint32 index) {
    if (op_code == OP_ADD) {
        acc.add(pa, pb, index, index, index);
    } else if (op_code == OP_SUB) {
        acc.sub(pa, pb, index, index, index);
    } else if (op_code == OP_MUL) {
        acc.mul(pa, pb, index, index, index);
    }
}

void kernel(
        pipe<T> pa, 
        pipe<T> pb, 
        pipe<T> pc, 
        uint32 num_frames,
        uint32 frame_tiles) {
    pa.set_frame(frame_tiles);
    pb.set_frame(frame_tiles);
    pc.set_frame(frame_tiles);
    for (uint32 frame = 0; frame < num_frames; frame++) {
        pc.reserve_back();
        pa.wait_front();
        pb.wait_front();
        math<T> acc;
        for (uint32 i = 0; i < frame_tiles; i++) {
            binary_op(acc, pa, pb, i);
        }
        for (uint32 i = 0; i < frame_tiles; i++) {
            acc.pack(i, pc);
        }
        pa.pop_front();
        pb.pop_front();
        pc.push_back();
    }
}
```

### A.3 Writer kernel

```
void kernel(
        global<T> gc,
        pipe<T> pc,
        uint32 N,
        uint32 num_frames,
        uint32 frame_tiles,
        uint32 start,
        uint32 stride) {
    pc.set_frame(frame_tiles);
    uint32 frame_items = frame_tiles * 1024;
    for (uint32 n = 0; n < N; n++) {
        uint32 pos = start;
        for (uint32 i = 0; i < num_frames; i++) {
            pc.wait_front();
            pc.write(0, gc, pos, frame_items);
            write_barrier();
            pc.pop_front();
            pos += frame_items;
        }
        start += stride;
    }
}
```

## Appendix B: Kernel programming interface summary

The following is Tanto kernel programming interface summary expressed using C++ notation. 
Note that C++ classes serve for the descriptive purposes only and are not necessarily used 
for the actual implementation.

This Appendix is not normative.

```
//
//    Global buffer
//

template<typename T>
class global { };

//
//    Local buffer
//

template<typename T>
class local {
public:
    T get(uint32 index);
    void set(uint32 index, T value);
    void read(
        uint32 dst_offset, 
        global<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write(
        uint32 src_offset, 
        global<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write_mcast(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void move_init(uint32 count);
    void move(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset);
    void move(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset);
};

//
//    Pipe
//

template<typename T>
class pipe {
public:
    void set_frame(uint32 tiles);
    void reserve_back();
    void push_back();
    void wait_front();
    void pop_front();
    void read(
        uint32 dst_offset, 
        global<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count);
    void read(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write(
        uint32 src_offset, 
        global<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count);
    void write(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count, 
        uint32 x, 
        uint32 y);
    void write_mcast(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        local<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void write_mcast_with_self(
        uint32 src_offset, 
        pipe<T> dst, 
        uint32 dst_offset, 
        uint32 count,
        uint32 x_start, 
        uint32 y_start, 
        uint32 x_end, 
        uint32 y_end, 
        uint32 num_dests);
    void move_init(uint32 count);
    void move(
        uint32 dst_offset, 
        local<T> src, 
        uint32 src_offset);
    void move(
        uint32 dst_offset, 
        pipe<T> src, 
        uint32 src_offset);
};

//
//    Semaphore
//

class semaphore {
public:
    void set(uint32 value);
    void set_remote(
        semaphore src, 
        uint32 x, 
        uint32 y);
    void set_mcast(
        semaphore src,
        uint32 x_start,
        uint32 y_start,
        uint32 x_end,
        uint32 y_end,
        uint32 num_dests);
    void inc(uint32 x, uint32 y, uint32 value);
    void wait(uint32 value);
};

//
//    Math
//

template<typename T>
class math {
public:
    // pack
    template<typename U> 
        void pack(uint32 isrc, pipe<U> dst);
    template<typename U> 
        void pack_row(uint32 isrc, pipe<U> dst);
    template<typename U> 
        void pack_col(uint32 isrc, pipe<U> dst);
    template<typename U> 
        void pack_scalar(uint32 isrc, pipe<U> dst);
    // copy
    template<typename U>
        void copy(pipe<U> src, uint32 isrc, uint32 idst);
    // eltwise binary
    template<typename U, typename V>
    void add(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    // bcast
    template<typename U, typename V>
    void add_bcast_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub_bcast_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul_bcast_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void add_bcast_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub_bcast_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul_bcast_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void add_bcast_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void sub_bcast_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    template<typename U, typename V>
    void mul_bcast_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);
    // matmul
    template<typename U, typename V>
    void matmul(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst,
        bool transpose);
    // reduce
    template<typename U, typename V>
    void reduce_max_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_max_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_max_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_sum_rows(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_sum_cols(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    template<typename U, typename V>
    void reduce_sum_scalar(
        pipe<U> src0, 
        pipe<V> src1, 
        uint32 isrc0, 
        uint32 isrc1, 
        uint32 idst);    
    // transpose
    template<typename U>
        void transpose(pipe<U> src, uint32 isrc, uint32 idst);
    // eltwise unary
    void abs(uint32 idst);
    void acos(uint32 idst);
    void add_scalar(uint32 dst, uint32 param);
    void asin(uint32 idst);
    void atan(uint32 idst);
    void cos(uint32 idst);
    void div_scalar(uint32 idst, uint32 param);
    void elu(uint32 idst, uint32 param);
    void eqz(uint32 idst);
    void erf(uint32 idst);
    void erfc(uint32 idst);
    void erfinv(uint32 idst);
    void exp(uint32 idst);
    void exp2(uint32 idst);
    void expm1(uint32 idst);
    void gelu(uint32 idst);
    void gez(uint32 idst);
    void gtz(uint32 idst);
    void heaviside(uint32 idst, uint32 param);
    void i0(uint32 idst);
    void isfinite(uint32 idst);
    void isinf(uint32 idst);
    void isnan(uint32 idst);
    void isneginf(uint32 idst);
    void isposinf(uint32 idst);
    void leaky_relu(uint32 idst, uint32 param);
    void lez(uint32 idst);
    void log(uint32 idst);
    void log_with_base(uint32 idst, uint32 param);
    void logical_not(uint32 idst);
    void ltz(uint32 idst);
    void max(uint32 idst);
    void mul_scalar(uint32 idst, uint32 param);
    void nez(uint32 idst);
    void power(uint32 idst, uint32 param);
    void recip(uint32 idst);
    void relu(uint32 idst);
    void relu_max(uint32 idst, uint32 param);
    void relu_min(uint32 idst, uint32 param);
    void rsqrt(uint32 idst);
    void rsub_scalar(uint32 idst, uint32 param);
    void sigmoid(uint32 idst);
    void sign(uint32 idst);
    void signbit(uint32 idst);
    void sin(uint32 idst);
    void sqrt(uint32 idst);
    void square(uint32 idst);
    void sub_scalar(uint32 dst, uint32 param);
    void tan(uint32 idst);
    void tanh(uint32 idst);
};

//
//    Global dataflow functions
//

void read_barrier();
void write_barrier();

//
//    Global compute functions
//

template<typename U, typename V>
    void tilize_block(pipe<U> src, uint32 block, pipe<V> dst);
template<typename U, typename V>
    void untilize_block(pipe<U> src, uint32 block, pipe<V> dst);

```


