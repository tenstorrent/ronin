
# Reference model library

This section contains a reference library of neural network models.

The following models are currently supported:

```
resnet18           ResNet18
resnet50_v1_7      ResNet50 v1
mobilenetv2_050    MobileNetV2 50
```

The models are derived from sources documented [here](./MODELS.md).


## Prerequisites

The prerequisites of Tanto core SDK must be fulfilled.
The Tanto SDK must be installed.
The respective environment variables must be set.
Detailed description of these steps can be found in the section `tanto` of this repository.

The reference operation library must be built and deployed as described
in the section `yari/op` of this repository.


## Building model library

The source code and build scripts for the Yari NN framework are located 
in `yari/nn/src` and `yari/nn/prj` respectively.

To build the library and the models, make  `yari/nn/prj` your current directory and run this script:

```
cd yari/nn/prj
./build_all.sh
```

The library and executable files will be placed in `yari/nn/lib` and `yari/nn/bin` respectively.

For each model, the reference (CPU, Float32) and Tanto (Tensix, BFloat16) implementations will be built, 
named `test_ref` and `test_tanto` respectively.


## Obtaining model parameters and test inputs

Model parameters and test inputs are stored in this repository via LFS
in the `yari/nn/test` subdirectory.


## Running reference models

Before running reference models, make `yari/nn/test` your current directory.

Tanto versions of reference models can be run using the command line of this format:

```
../bin/<model>/test_tanto --mode <mode> [--batch <batch>] --input husky01.dat --data <model> [--repeat <repeat>]
```

where

- `<model>` is model identifier, one of `resnet18`, `resnet50_v1_7`, and `mobilenetv2_050`;
- `<mode>` is an implementation variant; all models support `global` and 
MobileNetV2 additionally supports `global_dsc`;
- `<batch>` is optional integer batch size, usual values are 16, 32, and 64; 
MobileNetV2 additionally supports 1; default is 16;
- `<repeat>` is optional integer repeat count, when supplied, 
the model will be repeatedly run specified number of times and 
average wall clock time for one run will be displayed; usual repeat count is 100.

Implementation variant `global` assumes that intermediate tensor data for network layers 
are stored in global memory (DRAM). Variant `global_dsc` implements fusion of depthwise and 
pointwise layers of Depthwise Separable Convolutions (DSC) into one set of Tanto kernels.

The input image(s) must be supplied in preprocessed binary format (based on NNEF specification). 
For initial demonstration, the prefabricated input `husky01.dat` shall be used. 
The preprocessing utility suitable for user input images in JPEG and PNG formats
is described below. 

Example:

```
../bin/mobilenetv2_050/test_tanto --mode global --batch 1 --input husky01.dat --data mobilenetv2_050 --repeat 100
```

Reference model versions are run using the similar command (note that mode and repeat count are not used):

```
../bin/<model>/test_ref [--batch <batch>] --input husky01.dat --data <model>
```


## Preprocessing input images

The code of a preprocessing utility which converts for user input images in JPEG and PNG formats
to the [NNEF format](https://www.khronos.org/nnef) is located in 
the `yari/nn/src/vendor/arhat/nnef` subdirectory.

The preprocessing utility is implemented in Go programming language and requires Go version 1.14 or higher.
For building the utility, Go must be installed as described at the [installation page](https://go.dev/doc/install).

The build script `build_nnef.sh` is located in the `yari/nn/prj/vendor/arhat` subdirectory.
To build the preprocessing utility, make it your current directory and run this script:

```
cd yari/nn/prj/vendor/arhat
./build_nnef.sh
```

The utility executable code will be placed in `yari/nn/bin/vendor/arhat/image_to_tensor`.

To run the utility for the input image file `<input>` and produce the preprocessed
output file `<output>` use this command (assuming the current directory `yari/nn/test`):

```
../bin/vendor/arhat/image_to_tensor <input> --size 224 224 --range 0 1 --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --output <output>
```

The utility will resize the input image 224 x 224 pixels, scale the RGB channel values to the range
`[0, 1]`, and normalize the channel values with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
The resulting tensor will have the NCHW layout.

NOTE: This preprocessing method differs from that commonly used for the image recognition
CNN models and requires the input image first being scaled to minimum size of 256 x 256 while 
keeping aspect ratio and then cropped to 224 x 224. However, it is sufficient for demonstrating 
the functionality of models in this section.



