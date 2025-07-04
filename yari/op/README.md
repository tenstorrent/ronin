
# Reference operation library

This section contains a reference library of commonly used neural network
operations implemented using the Tanto programming model.

The source code is located in `yari/op/src` and includes the following sections:

```
binary        elementwise binary operations
common        common basic functionality
conv          convolution
deform_conv   deformable convolution
fc            fully connected layer (matmul + bias)
group_conv    group convolution (including depthwise)
interp        interpolation
move          data move between DRAM and L1 (experimental)
pool          pooling
reduce        reduction
```

The common directory layout for all operations is:

```
device       Device code (kernels)
    metal    TT-Metalium kernels (auto generated)
    tanto    Tanto kernels
host         Host code
    ref      Reference implementation (CPU)
    tanto    Tanto implementation
test         Test programs
    tanto    Test driver
```

TT-Metalium kernels are automatically generated (precompiled) from respective Tanto kernels. 
In some cases one Tanto kernel can produce multiple TT-Metalium kernels for different parameter settings 
(for example, different binary or unary op codes).
The kernel compilation scripts for all operation categories are located in the respective 
`device` subdirectories and have the name `front.sh`.

Host side implementations of operations are packaged as C++ classes with the uniform interface.
Member functions of these classes are responsible for initialisation and running of operations. 
Also, there are functions that implement required preprocessing of input and postprocessing 
of output on the host side.


## Prerequisites

The prerequisites of Tanto core SDK must be fulfilled.
The Tanto SDK must be installed.
The respective environment variables must be set.
Detailed description of these steps can be found in the section `tanto` of this repository.


## Building operation library

The source code and build scripts for the Yari operation library are located in 
`yari/op/src` and `yari/op/prj` respectively.

To build the library and the unit tests, make  `yari/op/prj` your current directory 
and run this script:

```
cd yari/op/prj
./build_all.sh
```

The library and executable files will be placed in `yari/op/lib` and `yari/op/bin` respectively.


## Deploying device kernels

TT-Metalium requires all device code be placed in its home directory. 
This directory is referenced via `TT_METAL_HOME` environment variable.

Operation kernels are located in the `device` subdirectories of the operation source tree; 
the general directory naming convention is `yari/op/src/<optype>` 
where `<optype>` stands for the operation type (e.g., `binary`). 
Contents of all these subdirectories must be copied to the `op` subdirectory in TT-Metalium home.
To perform this copy, make `yari/op/prj` your current directory and run this script:

```
cd yari/op/prj
./deploy_metal.sh
```


## Running Yari operation unit tests

Yari operation unit tests are built in `yari/op/bin`. 
There is a separate executable file for each operation. 
To run unit tests, make `yari/op` your current directory.

The general command line syntax for unit tests is:

```
bin/<optype>/test_tanto [-r <repeat>] <op> [<N> [<B>]]
```

where

- `<optype>` is operation type (one of the listed above);
- `<op>` is implementation name (operation-specific);
- `<N>` is optional input batch size (default is 16);
- `<B>` is optional internal batch size (default is implementation-specific);
- `<repeat>` is optional number of repetitions, when set, the benchmarking will be performed.

The internal batch size is a number of elements in a batch that can be simultaneously processed 
by all Tensix cores on a chip. 
The input batch size `<N>` must be divisible by the internal batch size `<B>`.

When requested, the benchmarking repeatedly enqueues the operation device kernels `<repeat>` times 
and measures the total wall-clock time until completion divided by the number of repetitions.

Example:

```
bin/binary/test_tanto binary_batch 64
```

The list of available implementations for each operation type can be obtained by running the respective 
unit test without arguments.

NOTE: At present, unit tests set too rigid tolerance and PCC ranges, 
therefore some unit tests will report failures. These should be ignored for the time being. 
Silicon tests perform bfloat16 calculations while reference implementations are float32, 
hence comparison can show visible difference in some cases. 
Anyway, for most tests PCC should be above `0.9999` (although sometimes lower PCC values can be produced).

