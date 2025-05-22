//
// Copyright (c) 2019-2020 FRAGATA COMPUTER SYSTEMS AG
// Copyright (c) 2017 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

//
// Based on the code of The Khronos Group Inc. NNEF Tools.
// Ported from C++ to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package core

import (
    "fmt"
    "sort"
)

//
//    Shape
//

type Shape []int

func(s Shape) String() string {
    str := "["
    for i, v := range s {
        if i != 0 {
            str += ","
        }
        str += fmt.Sprintf("%d", v)
    }
    str += "]"
    return str
}

func(s Shape) Clone() Shape {
    n := len(s)
    r := make(Shape, n)
    copy(r, s)
    return r
}

func(s Shape) Eq(other Shape) bool {
    n := len(s)
    if len(other) != n {
        return false
    }
    for i := 0; i < n; i++ {
        if s[i] != other[i] {
            return false
        }
    }
    return true
}

func MakeShape(arg Value, offset int) Shape {
    // arg: integer[]
    size := arg.Size()
    shape := make(Shape, offset+size);
    for i := 0; i < offset; i++ {
        shape[i] = 1
    }
    for i := 0; i < size; i++ {
        shape[i+offset] = arg.At(i).Integer()
    }
    return shape
}

func MakePaddingShape(arg Value, offset int) Shape {
    // arg: integer [][2]
    size := arg.Size()
    padding := make(Shape, offset+size)
    for i := 0; i < offset; i++ {
        padding[i] = 0
    }
    for i := 0; i < size; i++ {
        v := arg.At(i)
        padding[i+offset] = v.At(0).Integer() + v.At(1).Integer()
    }
    return padding
}

func(s Shape) VolumeOf() int {
    r := 1
    for _, v := range s {
        r *= v
    }
    return r
}
    
func(s Shape) VolumeOfSlice(offset int, length int) int {
    r := 1
    for _, v := range s[offset:offset+length] {
        r *= v
    }
    return r
}

func BroadcastableN(xShape Shape, yShape Shape, n int) bool {
    xSize := len(xShape)
    ySize := len(yShape)
    for i := 0; i < n; i++ {
        xi := 1
        yi := 1
        if i < xSize {
            xi = xShape[i]
        }
        if i < ySize {
            yi = yShape[i]
        }
        if xi != yi && xi != 1 {
            return false
        }
    }
    return true
}
    
func Broadcastable(xShape Shape, yShape Shape) bool {
    rank := IntMax(len(xShape), len(yShape))
    return BroadcastableN(xShape, yShape, rank)
}

func BroadcastCompatibleN(xShape Shape, yShape Shape, n int) bool {
    xSize := len(xShape)
    ySize := len(yShape)
    for i := 0; i < n; i++ {
        xi := 1
        yi := 1
        if i < xSize {
            xi = xShape[i]
        }
        if i < ySize {
            yi = yShape[i]
        }
        if xi != yi && xi != 1 && yi != 1 {
            return false
        }
    }
    return true
}

func BroadcastCompatible(xShape Shape, yShape Shape) bool {
    rank := IntMax(len(xShape), len(yShape))
    return BroadcastCompatibleN(xShape, yShape, rank)
}

func AxesCompatibleWithRank(axes Value, rank int) bool {
    size := axes.Size()
    for i := 0; i < size; i++ {
        axis := axes.At(i).Integer()
        if axis < 0 || axis >= rank {
            return false
        }
    }
    return true
}

func ContainsAxis(axes Value, axis int) bool {
    size := axes.Size()
    for i := 0; i < size; i++ {
        if axes.At(i).Integer() == axis {
            return true
        }
    }
    return false
}

func Sign(val int) int {
    switch {
    case val > 0:
        return 1
    case val < 0:
        return -1
    default:
        return 0
    }
}
    
func Downsize(input int, size int, padding int, stride int, dilation int) int {
    window := 1 + (size - 1) * dilation
    return Sign(input) * ((IntAbs(input) + padding - window) / stride + 1)
}
    
func DownsizeBasic(input int, stride int) int {
    return Sign(input) * ((IntAbs(input) + stride - 1) / stride)
}
   
func Upsize(input int, size int, padding int, stride int, dilation int) int {
    window := 1 + (size - 1) * dilation
    return Sign(input) * ((IntAbs(input) - 1) * stride + window - padding)
}
    
func UpsizeBasic(input int, stride int) int {
    return input * stride
}

func Check(condition bool, message string, args ...interface{}) {
    if !condition {
        LogicError(message, args...)
    }
}

func CheckAxisCompatibleWithRank(axis Value, rank int) {
    a := axis.Integer()
    Check(a >= 0 && a < rank,
        "axis must be in range [0, %d); found %d", rank, a)
    }

func CheckAxesCompatibleWithRank(axes Value, rank int) {
    Check(AxesCompatibleWithRank(axes, rank),
        "axes must be in range [0, %d); found %s", rank, axes.Repr())
    }

func CheckRange(name string, value Value, min int) {
    switch value.Kind() {
    case ValueKindArray, ValueKindTuple:
        size := value.Size()
        for i := 0; i < size; i++ {
            CheckRange(name, value.At(i), min)
        }
    case ValueKindInteger:
        v := value.Integer()
        Check(v >= min, "'%s' must be >= %d (found %d)", name, min, v)
    }
}

func CheckRank(name string, value Value, rank int) {
    size := value.Size()
    Check(size == rank, 
        "length of array '%s' must be %d to match rank of operation (found %d)",
            name, rank, size)
    }

func BroadcastShapeN(xShape Shape, yShape Shape, n int) Shape {
    xSize := len(xShape)
    ySize := len(yShape)
    rank := IntMax(xSize, ySize)
    zShape := make(Shape, rank)
    for i := 0; i < n; i++ {
        xi := 1
        yi := 1
        if i < xSize {
            xi = xShape[i]
        }
        if i < ySize {
            yi = yShape[i]
        }
        zShape[i] = IntMax(xi, yi)
    }
    return zShape
}
    
func BroadcastShape(xShape Shape, yShape Shape) Shape {
    rank := IntMax(len(xShape), len(yShape))
    return BroadcastShapeN(xShape, yShape, rank)
}

func NullaryShape(shape Value) Shape {
    return MakeShape(shape, 0)
}

func ConstantShape(shape Value, value Value) Shape {
    result := NullaryShape(shape)
    Check(value.Size() == result.VolumeOf() || value.Size() == 1,
        "shape volume (%d) does not match number of values (%d)", 
            result.VolumeOf(), value.Size())
    return result
}

func UnaryShape(shape Shape) Shape {
    return shape.Clone()
}

func BinaryShape(shape1 Shape, shape2 Shape) Shape {
    Check(BroadcastCompatible(shape1, shape2),
        "incompatible tensor shapes for broadcasting (%s vs %s)",
            shape1.String(), shape2.String())
    return BroadcastShape(shape1, shape2)
}
    
func AsymmetricBinaryShape(shape1 Shape, shape2 Shape) Shape {
    Check(Broadcastable(shape2, shape1),
        "cannot broadcast second argument shape (%s) to first argument shape (%s)",
            shape2.String(), shape1.String())
    return shape1.Clone()
}

func TernaryShape(shape1 Shape, shape2 Shape, shape3 Shape) Shape {
    return BinaryShape(BinaryShape(shape1, shape2), shape3)
}

func ReduceShape(input Shape, axes Value) Shape {
    CheckAxesCompatibleWithRank(axes, len(input))
    output := input.Clone()
    axesSize := axes.Size()
    for i := 0; i < axesSize; i++ {
        axis := axes.At(i).Integer()
        output[axis] = 1
    }    
    return output
}
    
func DownsampleShape(input Shape, factor Value) Shape {
    factorSize := factor.Size()
    for i := 0; i < factorSize; i++ {
        scale := factor.At(i).Integer()
        Check(input[i+2] % scale == 0, 
            "input extent (%d) must be divisible by factor (%d)", 
                input[i+2], scale)
    }
    output := input.Clone()
    for i := 0; i < factorSize; i++ {
        output[i+2] /= factor.At(i).Integer()
    }
    return output
}
    
func UpsampleShape(input Shape, factor Value) Shape {
    CheckRank("factor", factor, len(input)-2)
    output := input.Clone()
    factorSize := factor.Size()
    for i := 0; i < factorSize; i++ {
        output[i+2] *= factor.At(i).Integer()
    }
    return output
}

func DownsizeShape(
        input Shape, 
        kernel Shape,
        padding Shape,
        stride Shape,
        dilation Shape,
        offset int) Shape {
    size := len(input)
    output := make(Shape, size)
    withPadding := (len(padding) != 0)
    for i := offset; i < size; i++ {
        if withPadding {
            output[i] = Downsize(input[i], kernel[i], padding[i], stride[i], dilation[i])
        } else {
            output[i] = DownsizeBasic(input[i], stride[i])
        }
    }
    return output
}
    
func UpsizeShape(
        input Shape,
        kernel Shape,
        padding Shape,
        stride Shape,
        dilation Shape,
        offset int) Shape {
    size := len(input)
    output := make(Shape, size)
    withPadding := (len(padding) != 0)
    for i := offset; i < size; i++ {
        if withPadding {
            output[i] = Upsize(input[i], kernel[i], padding[i], stride[i], dilation[i])
        } else {
            output[i] = UpsizeBasic(input[i], stride[i])
        }
    }
    return output
}

func ConvLikeShape(
        input Shape,
        filter Shape,
        bias Shape,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        groups Value,
        outputShape Value,
        transposed bool) Shape {
    rank := len(input)
    if padding.Size() != 0 {
        CheckRank("padding", padding, rank-2)
    }
    if stride.Size() != 0 {
        CheckRank("stride", stride, rank-2)
    }
    if dilation.Size() != 0 {
        CheckRank("dilation", dilation, rank-2)
    }
    CheckRange("stride", stride, 1)
    CheckRange("dilation", dilation, 1)
    CheckRange("groups", groups, 0)
    var groupCount int
    switch {
    case groups.Integer() != 0:
        groupCount = groups.Integer()
    case transposed && !outputShape.IsNone() && outputShape.Size() != 0:
        groupCount = outputShape.At(1).Integer()
    default: 
        groupCount = input[1]
    }
    if transposed {
        Check(input[1] == filter[0], 
            "filter batch (%d) does not match input channels (%d)",
                filter[0], input[1])
    } else {
        Check(input[1] == filter[1] * groupCount, 
            "filter channels (%d) does not match input channels (%d) times groups (%d)",
                filter[1], input[1], groupCount)
    }    
    Check(filter[0] % groupCount == 0, 
        "filter batch (%d) must be divisible by groups (%d)", 
            filter[0], groupCount)
    Check(len(bias) <= 2, "bias shape must be of rank at most 2, found %d", len(bias))
    if len(bias) == 2 {
        Check(bias[0] == 1, "bias shape must be singular for the batch dimension")
    }
    if len(bias) != 0 {
        var channels int
        if transposed {
            channels = filter[1] * groupCount
        } else {
            channels = filter[0]
        }
        lastBias := bias[len(bias)-1]
        Check(lastBias == channels || lastBias == 1, 
            "bias channels (%d) does not match output channels (%d)",
                lastBias, channels)
    }
    var n int
    if stride.Size() != 0 {
        n = 2
    } else {
        n = rank
    } 
    strideShape := MakeShape(stride, n)
    if dilation.Size() != 0 {
        n = 2
    } else {
        n = rank
    }
    dilationShape := MakeShape(dilation, n)
    var paddingShape Shape
    if padding.Size() != 0 {
        paddingShape = MakePaddingShape(padding, 2)
    }   
    if !outputShape.IsNone() && outputShape.Size() != 0 {
        output := MakeShape(outputShape, 0)
        CheckRank("output_shape", outputShape, rank);
        CheckRange("output_shape", outputShape, 1)
        Check(output[0] == input[0], 
            "output batch (%d) does not match input batch (%d)", 
                output[0], input[0])
        Check(output[1] == filter[1] * groupCount, 
            "output channels (%d) does not match filter channels (%d) times groups (%d)",
                output[1], filter[1], groupCount)
        expected := DownsizeShape(output, filter, paddingShape, strideShape, dilationShape, 2)
        copy(expected, input[:2])
        Check(input.Eq(expected), 
            "expected input shape %s derived from output shape "+
                "is incompatible with actual input shape %s",
                    expected.String(), input.String())   
        return output
    }
    if transposed {
        output := UpsizeShape(input, filter, paddingShape, strideShape, dilationShape, 2)
        output[0] = input[0]
        output[1] = filter[1] * groupCount
        return output
    } else {
        output := DownsizeShape(input, filter, paddingShape, strideShape, dilationShape, 2)
        output[0] = input[0]
        output[1] = filter[0]
        return output
    }
}

func SeparableConvLikeShape(
        input Shape,
        planeFilter Shape,
        pointFilter Shape,
        bias Shape,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        groups Value,
        outputShape Value,
        transposed bool) Shape {
    for _, v := range pointFilter[2:] {
        Check(v == 1, "point filter must have singular extents in spatial dimensions")
    }
    Check(pointFilter[1] == planeFilter[0], 
        "channel dimension of point filter must equal batch dimension of plane filter")
    Check(planeFilter[1] == 1, "channel dimension of plane filter must be singular")
    filter := planeFilter.Clone()
    filter[0] = pointFilter[0]
    if transposed {
        filter[1] = pointFilter[1]
    } else {
        filter[1] = input[1]
    }
    return ConvLikeShape(
        input, filter, bias, border, padding, stride, dilation, groups, outputShape, transposed)
}
    
func ConvShape(
        input Shape,
        filter Shape,
        bias Shape,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        groups Value) Shape {
    return ConvLikeShape(
        input, filter, bias, border, padding, stride, dilation, groups, None(), false)
}

func DeconvShape(
        input Shape,
        filter Shape,
        bias Shape,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        outputShape Value,
        groups Value) Shape {
    return ConvLikeShape(
        input, filter, bias, border, padding, stride, dilation, groups, outputShape, true)
}
    
func SeparableConvShape(
        input Shape,
        planeFilter Shape,
        pointFilter Shape,
        bias Shape,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        groups Value) Shape {
    return SeparableConvLikeShape(
        input, planeFilter, pointFilter, bias, 
        border, padding, stride, dilation, groups, None(), false)
}
    
func SeparableDeconvShape(
        input Shape,
        planeFilter Shape,
        pointFilter Shape,
        bias Shape,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        outputShape Value,
        groups Value) Shape {
    return SeparableConvLikeShape(
        input, planeFilter, pointFilter, bias, 
        border, padding, stride, dilation, groups, outputShape, true)
}

func PoolLikeShape(
        input Shape,
        size Value,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        outputShape Value,
        transposed bool) Shape {
    rank := len(input)
    CheckRank("size", size, rank)
    if padding.Size() != 0 {
        CheckRank("padding", padding, rank)
    }
    if stride.Size() != 0 {
        CheckRank("stride", stride, rank)
    }
    if dilation.Size() != 0 {
        CheckRank("dilation", dilation, rank)
    }
    CheckRange("size", size, 1)
    CheckRange("stride", stride, 1)
    CheckRange("dilation", dilation, 1)
    kernelShape := MakeShape(size, 0)
    var n int
    if stride.Size() != 0 {
        n = 0
    } else {
        n = rank
    }
    strideShape := MakeShape(stride, n)
    if dilation.Size() != 0 {
        n = 0
    } else {
        n = rank
    }
    dilationShape := MakeShape(dilation, n)
    var paddingShape Shape
    if padding.Size() != 0 {
        paddingShape = MakePaddingShape(padding, 0)
    }
    if !outputShape.IsNone() && outputShape.Size() != 0 {
        output := MakeShape(outputShape, 0)
        CheckRank("output_shape", outputShape, rank)
        CheckRange("output_shape", outputShape, 1)
        expected := DownsizeShape(output, kernelShape, paddingShape, strideShape, dilationShape, 0)
        Check(input.Eq(expected), 
            "expected input shape %s derived from output shape "+
                "is incompatible with actual input shape %s",
                    expected.String(), input.String())
        return output
    }    
    if transposed {
        return UpsizeShape(input, kernelShape, paddingShape, strideShape, dilationShape, 0)
    } else {
        return DownsizeShape(input, kernelShape, paddingShape, strideShape, dilationShape, 0)
    }
}

func SampleLikeShape(
        input Shape,
        index Shape, 
        size Value,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        outputShape Value,
        transposed bool) Shape {
    Check(index.Eq(input), 
        "index shape incompatible with input shape (%s vs %s)", 
            index.String(), input.String())
    return PoolLikeShape(input, size, border, padding, stride, dilation, outputShape, transposed)
}

func PoolShape(
        input Shape,
        size Value,
        border Value,
        padding Value,
        stride Value,
        dilation Value) Shape {
    return PoolLikeShape(input, size, border, padding, stride, dilation, None(), false)
}

func UnpoolShape(
        input Shape,
        size Value,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        outputShape Value) Shape {
    return PoolLikeShape(input, size, border, padding, stride, dilation, outputShape, true)
}
    
func SampleShape(
        input Shape,
        index Shape,
        size Value,
        border Value,
        padding Value,
        stride Value,
        dilation Value) Shape {
    return SampleLikeShape(input, index, size, border, padding, stride, dilation, None(), false)
}
    
func DesampleShape(
        input Shape,
        index Shape,
        size Value,
        border Value,
        padding Value,
        stride Value,
        dilation Value,
        outputShape Value) Shape {
    return SampleLikeShape(input, index, size, border, padding, stride, dilation, outputShape, true)
}


func NormalizeShapeAxes(input Shape, axes Value) Shape {
    CheckAxesCompatibleWithRank(axes, len(input))
    return input.Clone()
}

func NormalizeShapeSize(input Shape, size Value) Shape {
    CheckRank("size", size, len(input))
    CheckRange("size", size, 1)
    return input.Clone()
}

func BatchnormShape(
        input Shape,
        mean Shape,
        variance Shape,
        offset Shape,
        scale Shape,
        epsilon Value) Shape {
    Check(Broadcastable(mean, input), 
        "cannot broadcast 'mean' shape (%s) to 'input' shape (%s)",
            mean.String(), input.String())
    Check(Broadcastable(variance, input), 
        "cannot broadcast 'variance' shape (%s) to 'input' shape (%s)",
            variance.String(), input.String())
    Check(Broadcastable(offset, input), 
        "cannot broadcast 'offset' shape (%s) to 'input' shape (%s)",
            offset.String(), input.String())
    Check(Broadcastable(scale, input), 
        "cannot broadcast 'scale' shape (%s) to 'input' shape (%s)",
            scale.String(), input.String())
    return input.Clone()
}

func RoiShape(input Shape, rois Shape, index Shape, size Value) Shape {
    CheckRank("output_size", size, len(input)-2)
    CheckRange("output_size", size, 1)
    Check(len(rois) == 2, "'rois' must be a rank-2 tensor")
    Check(len(index) == 1, "'batch_index' must be a rank-1 tensor")
    Check(rois[1] == 4, "rois must be of extent 4 along dimension 1 (found %d)", rois[1])
    Check(index[0] == rois[0], 
        "'batch_index' must be of same length as dimension 0 of rois; found (%d vs %d)", 
            index[0], rois[0])
    output := make(Shape, len(input))
    output[0] = rois[0]
    output[1] = input[1]
    for i := 0; i < size.Size(); i++ {
        output[i+2] = size.At(i).Integer()
    }
    return output
}

func RoiShapeResample(input Shape, rois Shape, index Shape, size Value, rate Value) Shape {
    CheckRank("sampling_rate", rate, len(input)-2)
    CheckRange("sampling_rate", rate, 1)     
    return RoiShape(input, rois, index, size)
}

func ReshapeShape(input Shape, shape Value, axisStart Value, axisCount Value) Shape {
    CheckAxisCompatibleWithRank(axisStart, len(input)+1)
    CheckRange("axis_count", axisStart, -1)       
    offset := axisStart.Integer()
    length := axisCount.Integer()
    if length == -1 {
        length = len(input) - offset
    }   
    Check(offset + length <= len(input), 
        "'axis_start' + 'axis_count' must be in range [0, %d], found %d",
            len(input), offset+length)
    size := shape.Size()
    output := make(Shape, len(input)+size-length)
    copy(output, input[:offset])
    autoAxis := -1
    for i := 0; i < size; i++ {
        s := shape.At(i).Integer()
        switch s {
        case 0:
            s = input[offset+i]
        case -1:
            Check(autoAxis < 0, "shape may only contain at most one -1 value")
            s = 1
            autoAxis = offset + i
        }
        output[offset+i] = s
    }
    copy(output[offset+size:], input[offset+length:])
    inputVolume := input.VolumeOfSlice(offset, length)
    outputVolume := output.VolumeOfSlice(offset, size)        
    if autoAxis >= 0 {
        // ACHTUNG: Apparent bug in original code:
        //     "outputVolume, inputVolume" instead of "output, input"
        Check(inputVolume % outputVolume == 0, 
            "automatic output shape (%s) incompatible with input shape (%s)", 
                output.String(), input.String())
        output[autoAxis] = inputVolume / outputVolume
    } else {
        Check(inputVolume == outputVolume, 
            "input volume (%d) does not equal output volume (%d)", 
                inputVolume, outputVolume)
    }    
    return output
}

func TransposeShape(input Shape, axes Value) Shape {
    size := axes.Size()
    perm := make([]int, size)
    for i := 0; i < size; i++ {
        perm[i] = axes.At(i).Integer()
    }
    sort.Ints(perm)
    for i := 0; i < size; i++ {
        Check(perm[i] == i, 
            "'axes' array must contain a permutation of dimensions from 0 to %d-1", size)
    }    
    output := input.Clone()
    for i := 0; i < size; i++ {
        j := axes.At(i).Integer()
        output[i] = input[j]
    }
    return output
}

func SplitShape(value Shape, axis Value, ratios Value) []Shape {
    CheckAxisCompatibleWithRank(axis, len(value))
    CheckRange("ratios", ratios, 1)
    idx := axis.Integer()
    total := 0
    size := ratios.Size()
    for i := 0; i < size; i++ {
        total += ratios.At(i).Integer()
    }
    Check(value[idx] % total == 0, 
        "sum of split ratios (%d) does not divide whole extent (%d)", 
            total, value[idx])
    unit := value[idx] / total
    values := make([]Shape, size)
    for i := 0; i < size; i++ {
        item := value.Clone()
        item[idx] = unit * ratios.At(i).Integer()
        values[i] = item
    }
    return values
}

func ConcatShape(valuesShape []Shape, axis Value) Shape {
    Check(len(valuesShape) != 0, "input array must be non-empty")
    outputShape := valuesShape[0].Clone()
    size := len(outputShape)
    CheckAxisCompatibleWithRank(axis, size)
    idx := axis.Integer()
    compatibleShape := true
    for _, partShape := range valuesShape[1:] {
        if len(partShape) != size {
            compatibleShape = false
            break
        }    
        for i := 0; i < size; i++ {
            if i == idx {
                outputShape[i] += partShape[i]
            } else {
                compatibleShape = compatibleShape && (outputShape[i] == partShape[i])
            }
        }
    }    
    Check(compatibleShape, "incompatible tensor shapes in input array")
    return outputShape
}

func SliceShape(input Shape, axes Value, begin Value, end Value) Shape {
    Check(begin.Size() == axes.Size() && end.Size() == axes.Size(), 
        "'axes', 'begin' and 'end' arrays must have the same length")
    CheckAxesCompatibleWithRank(axes, len(input))
    output := input.Clone()
    for i := 0; i < axes.Size(); i++ {
        axis := axes.At(i).Integer()
        extent := input[axis]
        first := begin.At(i).Integer()
        if first < 0 {
            first += extent
        }
        last := end.At(i).Integer()
        if last <= 0 {
            last += extent
        }
        Check(last > first, 
            "slice range (%d, %d) is empty for axis %d", 
                first, last, axis)
        Check(first >= 0 && last <= extent, 
            "slice range (%d, %d) is out of tensor shape for axis %d", 
                first, last, axis)
        output[axis] = last - first;
    }
    return output
}

func StackShape(inputs []Shape, axis Value) Shape {
    input := inputs[0]
    compatibleShapes := true
    for _, shape := range inputs[1:] {
        if !shape.Eq(input) {
            compatibleShapes = false
            break
        }
    }
    Check(compatibleShapes, "incompatible tensor shapes in input array")
    size := len(input) + 1
    output := make(Shape, size)        
    CheckAxisCompatibleWithRank(axis, size)
    idx := axis.Integer()
    for i := 0; i < idx; i++ {
        output[i] = input[i]
    }
    output[idx] = len(inputs)
    for i := idx + 1; i < size; i++ {
        output[i] = input[i-1]
    }
    return output
}

func UnstackShape(input Shape, axis Value) []Shape {
    size := len(input)
    CheckAxisCompatibleWithRank(axis, size)        
    idx := axis.Integer()
    output := make(Shape, size-1)
    for i := 0; i < idx; i++ {
        output[i] = input[i]
    }
    for i := idx; i < size - 1; i++ {
        output[i] = input[i+1]
    }
    n := input[idx]
    result := make([]Shape, n)
    for i := 0; i < n; i++ {
        if i == 0 {
            result[i] = output
        } else {
            result[i] = output.Clone()
        }
    }
    return result
}

func SqueezeShape(input Shape, axes Value) Shape {
    inputSize := len(input)
    axesSize := axes.Size()
    CheckAxesCompatibleWithRank(axes, inputSize)
    for i := 0; i < axesSize; i++ {
        axis := axes.At(i).Integer()
        Check(input[axis] == 1, 
            "squeezed dimension is not singleton (has extent %d)", input[axis])
    }    
    output := make(Shape, inputSize-axesSize)
    k := 0
    for i := 0; i < inputSize; i++ {
        if !ContainsAxis(axes, i) {
            output[k] = input[i]
            k++
        }
    }
    return output
}

func UnsqueezeShape(input Shape, axes Value) Shape {
    size := len(input) + axes.Size()
    CheckAxesCompatibleWithRank(axes, size)
    output := make(Shape, size)
    k := 0        
    for i := 0; i < size; i++ {
        if ContainsAxis(axes, i) {
            output[i] = 1
        } else {
            output[i] = input[k]
            k++
        }
    }
    return output
}

func TileShape(input Shape, repeats Value) Shape {
    size := len(input)
    CheckRank("repeats", repeats, size)
    CheckRange("repeats", repeats, 1);
    output := make(Shape, size)
    for i := 0; i < size; i++ {
        output[i] = input[i] * repeats.At(i).Integer()
    }
    return output
}
    
func PadShape(input Shape, padding Value) Shape {
    size := len(input)
    CheckRank("padding", padding, size)
    output := make(Shape, size)
    for i := 0; i < size; i++ {
        v := padding.At(i)
        output[i] = v.At(0).Integer() + input[i] + v.At(1).Integer()
    }
    return output
}

func MatmulShape(a Shape, b Shape, trA Value, trB Value) Shape {
    aSize := len(a)
    bSize := len(b)
    Check(aSize == bSize, "rank mismatch for A and B (%d vs %d)", aSize, bSize)
    rank := aSize
    Check(rank >= 2, "rank of A and B must be at least 2, found %d", rank)
    batchDims := rank - 2
    Check(BroadcastCompatibleN(a, b, batchDims),
        "incompatible tensor shapes for broadcasting first %d dimensions (%s vs %s)",
            batchDims, a.String(), b.String())
    i0 := batchDims
    i1 := batchDims + 1
    var m, n, kA, kB int
    if trA.Logical() {
        m = a[i1]
        kA = a[i0]
    } else {
        m = a[i0]
        kA = a[i1]
    }
    if trB.Logical() {
        n = b[i0]
        kB = b[i1]
    } else {
        n = b[i1]
        kB = b[i0]
    }
    Check(kA == kB, "inner dimensions must agree (%d vs %d)", kA, kB)
    c := BroadcastShapeN(a, b, batchDims)
    c[i0] = m
    c[i1] = n
    return c
}

func LinearShape(input Shape, filter Shape, bias Shape) Shape {
    Check(len(input) == 2, "input shape must be of rank 2 (found %d)", len(input))
    Check(len(filter) == 2, "filter shape must be of rank 2 (found %d)", len(filter))
    Check(input[1] == filter[1], "inner dimensions must agree (%d vs %d)", input[1], filter[1])
    if len(bias) != 0 {
        Check(len(bias) == 2, "bias shape must be of rank 2 (found %d)", len(bias))
        Check(bias[1] == filter[0],
            "bias channels (%d) does not match filter count (%d)", bias[1], filter[0])
        }
    return Shape{input[0], filter[0]}
}

func UpdateShape(variable Shape, value Shape) Shape {
    Check(value.Eq(variable), 
        "updated shape %s does not equal variable shape %s", 
            value.String(), variable.String())
    return variable.Clone()
}

func SoftmaxShape(inputShape Shape, axes Value) Shape {
    CheckAxesCompatibleWithRank(axes, len(inputShape))
    return inputShape.Clone()
}

func CopyNShape(shape Shape, times Value) []Shape {
    CheckRange("times", times, 1);
    size := times.Integer()
    result := make([]Shape, size)
    for i := 0; i < size; i++ {
        result[i] = shape.Clone()
    }
    return result
}

func AddNShape(inputs []Shape) Shape {
    size := len(inputs)
    Check(size != 0, "input array must be non-empty")
    shape := inputs[0]
    for i := 1; i < size; i++ {
        Check(inputs[i].Eq(shape), 
            "incompatible item shapes in array (%s vs %s)", 
                shape.String(), inputs[i].String())
    }
    return shape.Clone()
}

func QuantizeShape(input Shape, min Shape, max Shape, bits Value) Shape {
    Check(Broadcastable(min, input), 
        "cannot broadcast 'min' shape (%s) to 'input' shape (%s)",
            min.String(), input.String())
    Check(Broadcastable(max, input), 
        "cannot broadcast 'max' shape (%s) to 'input' shape (%s)",
            max.String(), input.String())
    CheckRange("bits", bits, 0)
    return input.Clone()
}
    
func LinearQuantizeShape(input Shape, min Shape, max Shape, bits Value) Shape {
    return QuantizeShape(input, min, max, bits)
}
    
func LogarithmicQuantizeShape(input Shape, max Shape, bits Value) Shape {
    return QuantizeShape(input, nil, max, bits)
}

