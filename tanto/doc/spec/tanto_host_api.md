
## Project Ronin

## Tanto host programming interface specification

*Version 1.0*

*Edition 12.05.2025*

<br>

## 1 Introduction

Tanto is a general-purpose high-level programming model for Tenstorrent AI processors. 
It is based on a conventional heterogeneous programming paradigm that views a computing system 
consisting of a host computer and one or more compute devices. 
Tanto applications consist of a host program and multiple device programs. 
The host program manages and coordinates execution of the device programs. 
It also creates and manages memory objects on computing devices. The device programs perform computations.

This document specifies a programming interface for implementation of Tanto host programs.


## 2 Architecture

This chapter specifies the architecture of Tanto programs.


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


## 3 Programming interface overview

Tanto host programming interface is constituted by host interface classes
representing the principal host program functional elements:

- `Platform`
- `Device`
- `Program`
- `Grid`
- `Global`
- `Local`
- `Pipe`
- `Semaphore`
- `Kernel`
- `Queue`


### 3.1 Host interface classes

The `Platform` class represents a Tenstorrent accelerator platform accessible from
a host computer. The platform is constituted by one or more Tenstorrent accelerator
devices and the low-level systems software which implements interaction between 
the host and the devices. A platform designated as "default" must be always available.

The `Device` class represents an accelerator device. 
Each device is associated with a certain platform.
Multiple devices can be associated with one platform.
Each device has an integer ID unique within the associated platform.

The `Program` class represents a device program. 
Each program is associated with a certain device.
Multiple programs can be associated with one device.

The `Grid` class represents a set of processing cores.
A grid is constituted as a union of disjoint core ranges. 
A core range represents a rectangular subset of the device grid of processing cores.
Each grid is associated with a certain program.
Multiple grids can be associated with one program.
Grids associated with one program must represent disjoint sets of processing cores.

The `Global` class represents a global buffer. 
Each global buffer is associated with a certain device.
Multiple global buffers can be associated with one device.

The `Local` class represents a local buffer.
Each local buffer is associated with a certain program.
Multiple local buffers can be associated with one program.

The `Pipe` class represents a pipe.
Each pipe is associated with a certain program.
Multiple pipes can be associated with one program.

The `Semaphore` class represents a semaphore.
Each semaphore is associated with a certain program.
Multiple semaphores can be associated with one program.

The `Kernel` class represents a kernel.
Each kernel is associated with a certain program.
Up to three kernels (read, write, and math) can be associated with one program.

The `Queue` class represents a device queue.
Queues implement interaction between the host program and devices.
The host program submits to device queues requests for
reading and writing data from and to global buffers as well
as launching device programs. These requests are enqueued
and executed asynchronously in order of their arrival to the queue.
Each queue is associated with a certain device.
Multiple queues can be associated with one device.
The number of available queues is defined by the device hardware capabilities.
Each queue has an integer ID unique within the associated device.
A queue with ID 0 must be always available.


### 3.2 Common reference semantics

The principal elements of host program functionality are implemented by 
the opaque objects which are not directly visible to programmers. 
These implementation objects are represented by the objects of host interface classes.
Multiple host interface objects can represent the same implementation object.

Conceptually, host interface objects hold shared ownership of implementation objects.
A host interface object may also own no implementation object.
An implementation object is destroyed once it is not owned by any host interface
object and is not associated with any other implementation object. 

For brevity, implementation objects will be referred to as "implementations" in this document.

NOTE: Semantics of host interface objects is similar to that of `std::shared_ptr` smart pointers.


## 4 Common types

The `DataFormat` enumeration class represents a scalar data type on a Tanto device.

```
enum class DataFormat {
    UINT8,
    UINT16,
    UINT32,
    FLOAT32,
    BFLOAT16
};
```


## 5 Platform class

The `Platform` class represents a Tenstorrent accelerator platform accessible from
a host computer. The platform is constituted by one or more Tenstorrent accelerator
devices and the low-level systems software which implements interaction between 
the host and the devices. A platform designated as "default" must be always available.

```
class Platform {
public:
    Platform();
    Platform(const Platform &other);
    Platform(Platform &&other) noexcept;
    ~Platform();
public:
    Platform &operator=(const Platform &other);
    Platform &operator=(Platform &&other) noexcept;
    static Platform get_default();
};

```


### 5.1 Constructors

```
Platform();
```

Constructs a `Platform` object which owns no implementation.

```
Platform(const Platform &other);
```

Constructs a `Platform` object which shares ownership of the implementation owned by 
another `Platform` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Platform(Platform &&other) noexcept;
```

Move-constructs a `Platform` object from another `Platform` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from


### 5.2 Member functions

```
Platform &operator=(const Platform &other);
```

Shares ownership of the implementation owned by another `Platform` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Platform &operator=(Platform &&other) noexcept;
```

Move-assigns a `Platform` object from another `Platform` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
static Platform get_default();
```

Returns an object which owns the default platform implementation.


## 6 Device class

The `Device` class represents an accelerator device. 
Each device is associated with a certain platform.
Multiple devices can be associated with one platform.
Each device has an integer ID unique within the associated platform.

```
class Device {
public:
    Device();
    Device(const Device &other);
    Device(Device &&other) noexcept;
    explicit Device(const Platform &platform, uint32_t id);
    ~Device();
public:
    Device &operator=(const Device &other);
    Device &operator=(Device &&other) noexcept;
    bool is_null() const;
    Platform platform() const;
    uint32_t id() const;
    void dram_grid_size(uint32_t &x, uint32_t &y) const;
    void worker_grid_size(uint32_t &x, uint32_t &y) const;
    void worker_core_from_logical_core(
        uint32_t logical_x,
        uint32_t logical_y,
        uint32_t &worker_x,
        uint32_t &worker_y) const;
    void close();
};
```


### 6.1 Constructors

```
Device();
```

Constructs a `Device` object which owns no implementation.

```
Device(const Device &other);
```

Constructs a `Device` object which shares ownership of the implementation owned by 
another `Device` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Device(Device &&other) noexcept;
```

Move-constructs a `Device` object from another `Device` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Device(const Platform &platform, uint32_t id);
```

Constructs a `Device` object for the specified platform and device ID.

`platform        ` platform<br>
`id              ` device ID


### 6.2 Member functions

```
Device &operator=(const Device &other);
```

Shares ownership of the implementation owned by another `Device` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Device &operator=(Device &&other) noexcept;
```

Move-assigns a `Device` object from another `Device` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Platform platform() const;
```

Returns the platform associated with this device.

```
uint32_t id() const;
```

Returns the device ID.

```
void dram_grid_size(uint32_t &x, uint32_t &y) const;
```

Obtains the grid size of DRAM cores.

`x       ` on return: grid width<br>
`y       ` on return: grid height

```
void worker_grid_size(uint32_t &x, uint32_t &y) const;
```

Obtains the grid size of worker cores.

`x       ` on return: grid width<br>
`y       ` on return: grid height

```
void worker_core_from_logical_core(
    uint32_t logical_x,
    uint32_t logical_y,
    uint32_t &worker_x,
    uint32_t &worker_y) const;
```

Translates logical coordinates of a worker core to physical coordinates.

`logical_x       ` logical x coordinate<br>
`logical_y       ` logical y coordinate<br>
`worker_x        ` on return: physical x coordinate<br>
`worker_y        ` on return: physical y coordinate

```
void close();
```

Closes the device. 
Operations with objects associated with closed devices are not allowed.


## 7 Program class

The `Program` class represents a device program. 
Each program is associated with a certain device.
Multiple programs can be associated with one device.

```
class Program {
public:
    Program();
    Program(const Program &other);
    Program(Program &&other) noexcept;
    explicit Program(const Device &device);
    ~Program();
public:
    Program &operator=(const Program &other);
    Program &operator=(Program &&other) noexcept;
    bool is_null() const;
    Device device() const;
};
```


### 7.1 Constructors

```
Program();
```

Constructs a `Program` object which owns no implementation.

```
Program(const Program &other);
```

Constructs a `Program` object which shares ownership of the implementation owned by 
another `Program` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Program(Program &&other) noexcept;
```

Move-constructs a `Program` object from another `Program` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Program(const Device &device);
```

Constructs a `Program` object which owns a new program implementation
associated with the specified device.

`device          ` device


### 7.2 Member functions

```
Program &operator=(const Program &other);
```

Shares ownership of the implementation owned by another `Program` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Program &operator=(Program &&other) noexcept;
```

Move-assigns a `Program` object from another `Program` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Device device() const;
```

Returns the device associated with this program.


## 8 Grid class

The `Grid` class represents a set of processing cores.
A grid is constituted as a union of disjoint core ranges. 
A core range represents a rectangular subset of the device grid of processing cores.
Core ranges constituting the grid are numbered by consecutive 0-based integer indices. 
Each grid is associated with a certain program.
Multiple grids can be associated with one program.
Grids associated with one program must represent disjoint sets of processing cores.

Core ranges are represented by the `Range` structure.

```
struct Range {
    uint32_t x_start;
    uint32_t y_start;
    uint32_t x_end;
    uint32_t y_end;
};
```

Members of the `Range` structure specify start and end logical coordinates of 
the respective rectangle in the device grid of processing cores.

`x_start         ` start x coordinate<br>
`y_start         ` start y coordinate<br>
`x_end           ` end x coordinate<br>
`y_end           ` end y coordinate

The range contains all cores with coordinates `(x, y)` such that
`x_start <= x <= x_end` and `y_start <= y <= y_end`.

```
class Grid {
public:
    Grid();
    Grid(const Grid &other);
    Grid(Grid &&other) noexcept;
    explicit Grid(
        const Program &program, 
        uint32_t x, 
        uint32_t y);
    explicit Grid(
        const Program &program,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end, 
        uint32_t y_end);
    explicit Grid(
        const Program &program,
        const std::vector<Range> &ranges);
    ~Grid();
public:
    Grid &operator=(const Grid &other);
    Grid &operator=(Grid &&other) noexcept;
    bool is_null() const;
    Program program() const;
    int range_count() const;
    Range range_at(int index) const;
};
```


### 8.1 Constructors

```
Grid();
```

Constructs a `Grid` object which owns no implementation.

```
Grid(const Grid &other);
```

Constructs a `Grid` object which shares ownership of the implementation owned by 
another `Grid` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Grid(Grid &&other) noexcept;
```

Move-constructs a `Grid` object from another `Grid` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Grid(
    const Program &program, 
    uint32_t x, 
    uint32_t y);
```

Constructs a `Grid` object which owns a new grid implementation
associated with the specified program. The new grid contains a single processing core with
the specified logical coordinates.

`program         ` program<br>
`x               ` x coordinate of the processing core<br>
`y               ` y coordinate of the processing core

```
explicit Grid(
    const Program &program,
    uint32_t x_start,
    uint32_t y_start,
    uint32_t x_end, 
    uint32_t y_end);
```

Constructs a `Grid` object which owns a new grid implementation
associated with the specified program. The new grid contains a single core range 
with the specified start and end logical coordinates.

`program         ` program<br>
`x_start         ` start x coordinate<br>
`y_start         ` start y coordinate<br>
`x_end           ` end x coordinate<br>
`y_end           ` end y coordinate

The range contains all processing cores with coordinates `(x, y)` such that
`x_start <= x <= x_end` and `y_start <= y <= y_end`.

```
explicit Grid(
    const Program &program,
    const std::vector<Range> &ranges);
```

Constructs a `Grid` object which owns a new grid implementation
associated with the specified program. The new grid contains the specified 
disjoint core ranges.

`program         ` program<br>
`ranges          ` vector of core ranges


### 8.2 Member functions

```
Grid &operator=(const Grid &other);
```

Shares ownership of the implementation owned by another `Grid` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Grid &operator=(Grid &&other) noexcept;
```

Move-assigns a `Device` object from another `Device` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Program program() const;
```

Returns the program object associated with this grid.

```
int range_count() const;
```

Returns the number of disjoint core ranges used for construction of this grid.

```
Range range_at(int index) const;
```

Returns a core range with the specified index.

`index   ` core range index


## 9 Global class

Global buffers represent allocations in DRAM that are persistent over device program invocations. 
Global buffers are split into pages that are allocated on all available DRAM banks 
in the round-robin fashion. 
The page size is expressed in elements and must be a power of 2.
The upper limit of the page size is implementation-specific.

The `Global` class represents a global buffer. 
Each global buffer is associated with a certain device.
Multiple global buffers can be associated with one device.

```
class Global {
public:
    Global();
    Global(const Global &other);
    Global(Global &&other) noexcept;
    explicit Global(
        const Device &device,
        DataFormat data_format,
        uint32_t size,
        uint32_t log2_page_size);
    ~Global();
public:
    Global &operator=(const Global &other);
    Global &operator=(Global &&other) noexcept;
    bool is_null() const;
    Device device() const;
    DataFormat data_format() const;
    uint32_t size() const;
    uint32_t page_size() const;
    uint32_t log2_page_size() const;
    uint32_t bytes() const;
    uint32_t page_bytes() const;
};
```

A global buffer has the following attributes:

- *data format* of elements
- *size* specifying number of elements in the buffer
- *logarithmic page size* specifying 2-based logarithm of the buffer page size

The global buffer size does not need to be a multiple of its page size.


### 9.1 Constructors

```
Global();
```

Constructs a `Global` object which owns no implementation.

```
Global(const Global &other);
```

Constructs a `Global` object which shares ownership of the implementation owned by 
another `Global` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Global(Global &&other) noexcept;
```

Move-constructs a `Global` object from another `Global` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Global(
    const Device &device,
    DataFormat data_format,
    uint32_t size,
    uint32_t log2_page_size);
```

Constructs a `Global` object which owns a new global buffer implementation associated
with the specified device. The new global buffer has the specified data format,
size, and logarithmic page size.

`device          ` device<br>
`data_format     ` data format of elements<br>
`size            ` number of elements<br>
`log2_page_size  ` logarithmic page size


### 9.2 Member functions

```
Global &operator=(const Global &other);
```

Shares ownership of the implementation owned by another `Global` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Global &operator=(Global &&other) noexcept;
```

Move-assigns a `Global` object from another `Global` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Device device() const;
```

Returns the device object associated with this global buffer.

```
DataFormat data_format() const;
```

Returns the data format of elements.

```
uint32_t size() const;
```

Returns the total number of elements.

```
uint32_t page_size() const;
```

Returns the number of elements in one page.

```
uint32_t log2_page_size() const;
```

Returns the logarithmic page size.

```
uint32_t bytes() const;
```

Returns the total number of bytes allocated for this global buffer.
This number is a multiple of the number of bytes in one page.

```
uint32_t page_bytes() const;
```

Returns the number of bytes in one bytes.


## 10 Local class

Local buffers represent contiguous blocks of data allocated 
in L1 memory of each core of a specified grid. 
Each core in the grid owns an instance of the local buffer. 
All instances of a local buffer have the same size and local address in L1. 
For each local buffer a scalar type of its elements must be specified. 
Local buffers are not persistent over device program invocations.

The `Local` class represents a local buffer.
Each local buffer is associated with a certain program.
Multiple local buffers can be associated with one program.

```
class Local {
public:
    Local();
    Local(const Local &other);
    Local(Local &&other) noexcept;
    explicit Local(
        const Program &program,
        const Grid &grid,
        DataFormat data_format,
        uint32_t size);
    ~Local();
public:
    Local &operator=(const Local &other);
    Local &operator=(Local &&other) noexcept;
    bool is_null() const;
    Device device() const;
    Program program() const;
    Grid grid() const;
    DataFormat data_format() const;
    uint32_t size() const;
    LocalScope scope() const;
};
```

A local buffer has the following attributes:

- *data format* of elements
- *size* specifying number of elements in the buffer


### 10.1 Constructors

```
Local();
```

Constructs a `Local` object which owns no implementation.

```
Local(const Local &other);
```

Constructs a `Local` object which shares ownership of the implementation owned by 
another `Local` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Local(Local &&other) noexcept;
```

Move-constructs a `Local` object from another `Local` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Local(
    const Program &program,
    const Grid &grid,
    DataFormat data_format,
    uint32_t size);
```

`program         ` program<br>
`grid            ` grid<br>
`data_format     ` data format of elements<br>
`size            ` number of elements

Constructs the `Local` object which owns a new local buffer implementation associated
with the specified program and grid. The new local buffer has the specified data format and size.


### 10.2 Member functions

```
Local &operator=(const Local &other);
```

Shares ownership of the implementation owned by another `Local` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Local &operator=(Local &&other) noexcept;
```

Move-assigns a `Local` object from another `Local` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Device device() const;
```

Returns the device object associated with this local buffer.

```
Program program() const;
```

Returns the program object associated with this local buffer.

```
Grid grid() const;
```

Returns the grid associated with this local buffer.

```
DataFormat data_format() const;
```

Returns the data format of elements.

```
uint32_t size() const;
```

Returns the total number of elements.


## 11 Pipe class

Pipes represent contiguous blocks of data allocated in L1 memory of each core of a specified grid. 
Pipes are organized as FIFO data structures and are used for synchronized sharing data 
between kernels running on the same core. 
Each core in the grid owns an instance of the pipe. 
All instances of a pipe have the same size and local address in L1. 
For each pipe a scalar type of its elements must be specified. 
Pipes are not persistent over device program invocations.

A pipe holds a sequence of data blocks of 1024 elements each. 
These blocks may contain genuine 2D tiles of 32 x 32 size 
but can also hold data chunks with any other semantics. 

Each pipe contains two distinct regions of the same size referred to as read frame and write frame. 
Kernels use these frames for reading and writing data respectively. 
The number of tiles in each frame is referred to as frame size. 
The initial frame size is specified during the pipe creation. 
If necessary, the frame size can be changed by the kernel using the dedicated pipe method. 

Each pipe belongs to one of three kinds: input, output, or intermediate. 
Input pipes provide write access to dataflow kernels and read access to compute and dataflow kernels. 
Output pipes provide write access to compute and dataflow kernels and read access to dataflow kernels. 
Intermediate pipes provide read and write access to compute kernels only. 

The `PipeKind` enumeration class represents the pipe kind values.

```
enum class PipeKind {
    INPUT,
    OUTPUT,
    INTERMED
};
```

A pipe can have an attached local buffer. In this case, the memory allocated for
the local buffer is used to hold the pipe contents.

The `Pipe` class represents a pipe.
Each pipe is associated with a certain program.
Multiple pipes can be associated with one program.

```
class Pipe {
public:
    Pipe();
    Pipe(const Pipe &other);
    Pipe(Pipe &&other) noexcept;
    explicit Pipe(
        const Program &program,
        const Grid &grid,
        PipeKind kind,
        DataFormat data_format,
        uint32_t size,
        uint32_t frame_size);
    ~Pipe();
public:
    Pipe &operator=(const Pipe &other);
    Pipe &operator=(Pipe &&other) noexcept;
    bool is_null() const;
    Program program() const;
    Grid grid() const;
    PipeKind kind() const;
    DataFormat data_format() const;
    uint32_t size() const;
    uint32_t frame_size() const;
    void set_local(const Local &local) const;
};
```

A pipe has the following attributes:

- *kind* or the pipe (input, output, or intermediate)
- *data format* of elements
- *size* specifying total number of blocks allocated for the pipe
- *frame size* specifying number of blocks in one frame


### 11.1 Constructors

```
Pipe();
```

Constructs a `Pipe` object which owns no implementation.

```
Pipe(const Pipe &other);
```

Constructs a `Pipe` object which shares ownership of the implementation owned by 
another `Pipe` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Pipe(Pipe &&other) noexcept;
```

Move-constructs a `Pipe` object from another `Pipe` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Pipe(
    const Program &program,
    const Grid &grid,
    PipeKind kind,
    DataFormat data_format,
    uint32_t size,
    uint32_t frame_size);
```

Constructs a `Pipe` object which owns a new pipe implementation associated
with the specified program and grid. The new pipe has the specified kind,
data format, size, and frame size.

`program         ` program<br>
`grid            ` grid<br>
`kind            ` kind<br>
`data_format     ` data format<br>
`size            ` size<br>
`frame_size      ` frame size


### 11.2 Member functions

```
Pipe &operator=(const Pipe &other);
```

Shares ownership of the implementation owned by another `Pipe` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Pipe &operator=(Pipe &&other) noexcept;
```

Move-assigns a `Pipe` object from another `Pipe` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Program program() const;
```

Returns the program associated with this pipe.

```
Grid grid() const;
```

Returns the grid associated with this pipe.

```
PipeKind kind() const;
```

Returns the pipe kind.

```
DataFormat data_format() const;
```

Returns the data format of elements.

```
uint32_t size() const;
```

Returns the total number of blocks allocated.

```
uint32_t frame_size() const;
```

Returns the number of blocks in one frame.

```
void set_local(const Local &local) const;
```

Attaches the specified local buffer to this pipe.


## 12 Semaphore class

Semaphores represent 4-byte data blocks of data allocated in L1 memory of each core of a specified grid. 
Data in these blocks are interpreted as 32-bit unsigned integers that are used for synchronization 
between kernels running on different processing cores. 
Each core in the grid owns an instance of the semaphore. 
All instances of a semaphore have the same local address in L1. 
Semaphores are not persistent over device program invocations.

The `Semaphore` class represents a semaphore.
Each semaphore is associated with a certain program.
Multiple semaphores can be associated with one program.

```
class Semaphore {
public:
    Semaphore();
    Semaphore(const Semaphore &other);
    Semaphore(Semaphore &&other) noexcept;
    explicit Semaphore(
        const Program &program,
        const Grid &grid, 
        uint32_t init_value);
    ~Semaphore();
public:
    Semaphore &operator=(const Semaphore &other);
    Semaphore &operator=(Semaphore &&other) noexcept;
    bool is_null() const;
    Program program() const;
    Grid grid() const;
    uint32_t init_value() const;
};
```

A semaphore has the attribute *initial value* specifying the initial value
assigned to the semaphore when the associated program is launched.


### 12.1 Constructors

```
Semaphore();
```

Constructs a `Semaphore` object which owns no implementation.

```
Semaphore(const Semaphore &other);
```

Constructs a `Semaphore` object which shares ownership of the implementation owned by 
another `Semaphore` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Semaphore(Semaphore &&other) noexcept;
```

Move-constructs a `Semaphore` object from another `Semaphore` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Semaphore(
    const Program &program,
    const Grid &grid, 
    uint32_t init_value);
```

Constructs a `Semaphore` object which owns a new semaphore implementation associated
with the specified program and grid. The new semaphore has the specified initial value

`program         ` program<br>
`grid            ` grid<br>
`init_value      ` initial value


### 12.2 Member functions

```
Semaphore &operator=(const Semaphore &other);
```

Shares ownership of the implementation owned by another `Semaphore` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Semaphore &operator=(Semaphore &&other) noexcept;
```

Move-assigns a `Semaphore` object from another `Semaphore` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Program program() const;
```

Returns the program associated with this semaphore.

```
Grid grid() const;
```

Returns the grid associated with this semaphore.

```
uint32_t init_value() const;
```

Returns the initial value.


## 13 Kernel class

The device program specifies a collection of lower-level execution units referred to as kernels. 
Each kernel specifies a serial code running on one processing core. 
Each kernel in the program is defined on a grid of processing cores. 

Each processing core can concurrently run up to three kernels referred to as reader,
writer, and math kernel. The reader and writer kernels can use NoC to transfer data
between L1 memory unit of its processing core and any L1 or DRAM memory unit of the device.
The math kernel can use the math units of its processing core to perform computations.

The `KernelKind` enumeration class represents the kernel kind values.

```
enum class KernelKind {
    READER,
    WRITER,
    MATH
};
```

The source code that constitutes the kernel has one of two formats: *Metal* or *Tanto*.
The Tanto format corresponds to original Tanto kernels confogming to the Tanto
device programming interface specification. 
The Metal format corresponds to TT-Metalium kernels obtained by the offline compilation
of the original Tanto kernels using the Tanto frontend compiler.

NOTE: Current implementation of the host API supports only the Metal kernel format.
This means that the original Tanto kernels must be transfomred offline into their
TT-Metalium equivalents using the Tanto frontend compiler.
Future implementations will support the Tanto format by performing 
this transformation online during the kernel construction.

The `KernelFormat` enumeration type represents the kernel format values.

```
enum class KernelFormat {
    METAL,
    TANTO
};
```

Kernels accept run time *arguments*. The arguments are specified in Tanto kernel
code as parameters of the `kernel` function. Each argument may have on of the following
types:

- `uint32`: 32-bit unsigned integer
- `Global`: global buffer
- `Local`: local buffer
- `Pipe`: pipe
- `Semaphore`: semaphore

The `KernelArg` type represents the argument values.

```
using KernelArg = std::variant<uint32_t, Global, Local, Pipe, Semaphore>;
```

The full array of kernel arguments if specified as a C++ vector of `KernelArg` values.
The ordering and types of the array values must conform to the `kernel` function
declaration in the Tanto kernel code.

Provided that the kernel is defined on a grid of processing cores, kernel instances
on different cores may have different arrays of arguments. Therefore, a distinct
array of arguments can be specified for a single core or a core range within
the processing core grid of the kernel. Nevertheless, the array of arguments
must be specified for each processing core in the kernel core grid.

The `Kernel` class represents a kernel.
Each kernel is associated with a certain program.
Up to three kernels (read, write, and math) can be associated with one program.

```
class Kernel {
public:
    Kernel();
    Kernel(const Kernel &other);
    Kernel(Kernel &&other) noexcept;
    explicit Kernel(
        const Program &program,
        const Grid &grid,
        KernelKind kind,
        KernelFormat format,
        const std::string &path,
        const std::vector<uint32_t> &compile_args,
        const std::map<std::string, std::string> &defines);
    ~Kernel();
public:
    Kernel &operator=(const Kernel &other);
    Kernel &operator=(Kernel &&other) noexcept;
    bool is_null() const;
    Program program() const;
    Grid grid() const;
    KernelKind kind() const;
    KernelFormat format() const;
    std::string path() const;
    const std::vector<uint32_t> &compile_args() const;
    const std::map<std::string, std::string> &defines() const;
    void set_args(
        uint32_t x,
        uint32_t y,
        const std::vector<KernelArg> &args) const;
    void set_args(
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end,
        const std::vector<KernelArg> &args) const;
    void set_args(const Grid &grid, const std::vector<KernelArg> &args) const;
};
```

A kernel has the following attributes:

- *kind* of the kernel (reader, writer, or math)
- *format* of the kernel code (Metal or Tanto)
- *path* to the kernel source code
- *compile time arguments*
- *compile time definitions*

The source code path specifies the location of the kernel source code in Metal or Tanto
format (depending on the kernel format attribute). It has the implementation-specific
interpretation.

NOTE: Currently, the kernel path is interpreted relative to the TT-Metal home location
designated via the `TT_METAL_HOME` environment variable.

The platform performs "just in time" compilation of the kernel source code
into a binary format that can be transmitted to the device. 
Compilation details are implementation-specific.

The compile time arguments are represented as an array of 32-bit unsigned integer values
which are mapped to the `param` declarations in the Tanto kernel code, in order
of their appearance. These values are passed to the Tanto compiler frontend.
The compile time arguments are applicable for the Tanto kernel format only.

The compile time definitions specify a set of name / value parts for macro definitions
passed to the Tanto compiler frontend in case of the Tanto kernel format or directly to
the TT-Metal device compiler in case of the Metal format.


### 13.1 Constructors

```
Kernel();
```

Constructs a `Kernel` object which owns no implementation.

```
Kernel(const Kernel &other);
```

Constructs a `Kernel` object which shares ownership of the implementation owned by 
another `Kernel` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Kernel(Kernel &&other) noexcept;
```

Move-constructs a `Kernel` object from another `Kernel` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Kernel(
    const Program &program,
    const Grid &grid,
    KernelKind kind,
    KernelFormat format,
    const std::string &path,
    const std::vector<uint32_t> &compile_args,
    const std::map<std::string, std::string> &defines);
```

Constructs a `Kernel` object which owns a new kernel implementation associated
with the specified program and grid. The new kernel has the specified kind,
format, code path, compile time arguments, and compile time definitions.

`program         ` program<br>
`grid            ` grid<br>
`kind            ` kind<br>
`format          ` format<br>
`path            ` code path<br>
`compile_args    ` compile time arguments<br>
`defines         ` compile time definitions


### 13.2 Member functions

```
Kernel &operator=(const Kernel &other);
```

Shares ownership of the implementation owned by another `Kernel` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Kernel &operator=(Kernel &&other) noexcept;
```

Move-assigns a `Kernel` object from another `Kernel` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Program program() const;
```

Returns the program associated with this kenel.

```
Grid grid() const;
```

Returns the grid associated with this kernel.

```
KernelKind kind() const;
```

Returns the kernel kind.

```
KernelFormat format() const;
```

Returns the kernel format.

```
std::string path() const;
```

Returns the code path.

```
const std::vector<uint32_t> &compile_args() const;
```

Returns the compile time arguments.

```
const std::map<std::string, std::string> &defines() const;
```

Returns the compile time definitions

```
void set_args(
    uint32_t x,
    uint32_t y,
    const std::vector<KernelArg> &args) const;
```

Sets the specified array of arguments for a processing core with the specified logical coordinates.

`x       ` x coordinate of the processing core<br>
`y       ` y coordinate of the processing core<br>
`args    ` array of arguments

```
void set_args(
    uint32_t x_start,
    uint32_t y_start,
    uint32_t x_end,
    uint32_t y_end,
    const std::vector<KernelArg> &args) const;
```

Sets the specified array of arguments for a core range with the specified start and end logical coordinates.

`x_start         ` start x coordinate<br>
`y_start         ` start y coordinate<br>
`x_end           ` end x coordinate<br>
`y_end           ` end y coordinate<br>
`args            ` array of arguments

```
void set_args(const Grid &grid, const std::vector<KernelArg> &args) const;
```

Sets the specified array of arguments for a a specified grid of processing cores.

`grid    ` grid<br>
`args    ` array of arguments

NOTE: Currently, it is required that all `Grid` objects must represent disjoint processing
core grids. To fullfill this requirement, the `grid` argument must represent
the entire grid associated with this kernel. This restriction may be relaxed in
the future versions of this specification/


## 14 Queue class

The `Queue` class represents a device queue.
Queues implement interaction between the host program and devices.
The host program submits to device queues requests for
reading and writing data from and to global buffers as well
as launching device programs. These requests are enqueued
and executed asynchronously in order of their arrival to the queue.
Each queue is associated with a certain device.
Multiple queues can be associated with one device.
The number of available queues is defined by the device hardware capabilities.
Each queue has an integer ID unique within the associated device.
A queue with ID 0 must be always available.

```
class Queue {
public:
    Queue();
    Queue(const Queue &other);
    Queue(Queue &&other) noexcept;
    explicit Queue(const Device &device, uint32_t id);
    ~Queue();
public:
    Queue &operator=(const Queue &other);
    Queue &operator=(Queue &&other) noexcept;
    bool is_null() const;
    Device device() const;
    uint32_t id() const;
    void enqueue_read(
        const Global &global, 
        void *dst,
        bool blocking) const;
    void enqueue_write(
        const Global &global, 
        const void *src,
        bool blocking) const;
    void enqueue_program(const Program &program, bool blocking) const;
    void finish() const;
};
```

The host program can use the queue objects to submit (enqueue) requests the following operations:

- read a device global buffer to a host array
- write a host array to a device global buffer
- execute a device program

Generally, these operations are performed asynchronously with the host program.
The enqueue request can be *blocking*, in this case the enqueueing function will not
return and further execution of the host program will be blocked until
completion of the requested operation.


# 14.1 Constructors

```
Queue();
```

Constructs a `Queue` object which owns no implementation.

```
Queue(const Queue &other);
```

Constructs a `Queue` object which shares ownership of the implementation owned by 
another `Queue` object. If the other object owns no implementation, 
this object owns no implementation too.

`other   ` another object to share the implementation ownership

```
Queue(Queue &&other) noexcept;
```

Move-constructs a `Queue` object from another `Queue` object. 
After the construction, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 

`other   ` another object to acquire the implementation ownership from

```
explicit Queue(const Device &device, uint32_t id);
```

Constructs a `Queue` object for the specified device and queue ID.

`device          ` device<br>
`id              ` queue ID


# 14.2 Member functions

```
Queue &operator=(const Queue &other);
```

Shares ownership of the implementation owned by another `Queue` object.
Returns a reference to this object.

`other   ` another object to share the implementation ownership

```
Queue &operator=(Queue &&other) noexcept;
```

Move-assigns a `Queue` object from another `Queue` object. 
After the assignment, this object contains a copy of the previous state of the other object 
and the other object owns no implementation. 
Returns a reference to this object.

`other   ` another object to acquire the implementation ownership from

```
bool is_null() const;
```

Returns `true` if this object owns no implementation, `false` otherwise.

```
Device device() const;
```

Returns the device associated with this queue.

```
uint32_t id() const;
```

Returns the queue ID.

```
void enqueue_read(
    const Global &global, 
    void *dst,
    bool blocking) const;
```

Enqueues reading of a device global buffer to a host array.

`global          ` global buffer<br>
`dst             ` host array pointer<br>
`blocking        ` if `true`, blocks host program until reading completion

```
void enqueue_write(
    const Global &global, 
    const void *src,
    bool blocking) const;
```

Enqueues writing of a host array to a device global buffer.

`global          ` global buffer<br>
`src             ` host array pointer<br>
`blocking        ` if `true`, blocks the host program until writing completion

```
void enqueue_program(const Program &program, bool blocking) const;
```

Enqueues execution of a device program.

`program         ` program<br>
`blocking        ` if `true`, blocks the host program until device program completion

```
void finish() const;
```

Blocks the host program execution until completion of all operations
previously enqueued on this queue.


