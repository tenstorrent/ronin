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
// Ported from Python to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package input

import (
    "fmt"
    "image"
    _ "image/gif"
    _ "image/jpeg"
    _ "image/png"
    "path/filepath"
    "os"
    "sort"
    "strings"
    "nfnt/resize"
)

//
//    ImageInput
//

const (
    ColorFormatRGB = "RGB"
    ColorFormatBGR = "BGR"
    DataFormatNCHW = "NCHW"
    DataFormatNHWC = "NHWC"
)

type ImageInput struct {
    filenames []string
    colorFormat string
    dataFormat string
    rng []float32
    norm [][3]float32
}

func NewImageInput(
        filenames []string, 
        colorFormat string, 
        dataFormat string, 
        rng []float32, 
        norm [][3]float32) *ImageInput {
    s := new(ImageInput)
    s.Init(filenames, colorFormat, dataFormat, rng, norm)
    return s
}

func(s *ImageInput) Init(
        filenames []string, 
        colorFormat string, 
        dataFormat string, 
        rng []float32, 
        norm [][3]float32) {
    s.filenames = filenames
    s.colorFormat = strings.ToUpper(colorFormat)
    s.dataFormat = strings.ToUpper(dataFormat)
    s.rng = rng
    s.norm = norm
}

func(s *ImageInput) CreateInput(
        dtype string, shape []int, allowBiggerBatch bool) (interface{}, []int, error) {
    if !(s.colorFormat == ColorFormatRGB || s.colorFormat == ColorFormatBGR) {
        err := fmt.Errorf("Invalid color format '%s'", s.colorFormat)
        return nil, nil, err
    }
    if !(s.dataFormat == DataFormatNCHW || s.dataFormat == DataFormatNHWC) {
        err := fmt.Errorf("Invalid data format '%s'", s.dataFormat)
        return nil, nil, err
    }
    if !(len(s.rng) == 0 || len(s.rng) == 2) {
        err := fmt.Errorf("Invalid range")
        return nil, nil, err
    }
    if !(len(s.norm) == 0 || len(s.norm) == 2) {
        err := fmt.Errorf("Invalid norm")
        return nil, nil, err
    }
    if !(len(shape) == 0 || len(shape) == 4) {
        err := fmt.Errorf("ImageInput can only produce tensors with rank 4")
        return nil, nil, err
    }
    var imgs [][]float32
    height := -1
    width := -1
    for _, pattern := range s.filenames {
        filenames, err := filepath.Glob(pattern)
        if err != nil {
            return nil, nil, err
        }
        if len(filenames) == 0 {
            err := fmt.Errorf("No files found for path %s", pattern)
            return nil, nil, err
        }
        sort.Strings(filenames)
        for _, filename := range filenames {
            img, h, w, err := s.getImage(filename, shape)
            if err != nil {
                return nil, nil, err
            }
            if height < 0 {
                height = h
                width = w
            } else {
                if h != height || w != width {
                    err := 
                        fmt.Errorf(
                            "The size of all images must be the same, or --size must be specified")
                    return nil, nil, err
                }
            }
            imgs = append(imgs, img)
        }
    }
    imgCount := len(imgs)
    imgSize := len(imgs[0])
    if len(shape) != 0 {
        batchCount := shape[0]
        if imgCount < batchCount {
            // Network batch size is bigger than supplied data, repeating it
            extImgs := make([][]float32, batchCount)
            for i := 0; i < batchCount; i++ {
                extImgs[i] = imgs[i%imgCount]
            }
            imgs = extImgs
        }
        if len(imgs) != batchCount && !allowBiggerBatch {
            err := fmt.Errorf("Network batch size is smaller than supplied data")
            return nil, nil, err
        }
    }
    switch dtype {
    case "scalar":
        batchCount := len(imgs)
        result := make([]float32, batchCount*imgSize)
        for b := 0; b < batchCount; b++ {
            for p := 0; p < imgSize; p++ {
                result[b*imgSize+p] = imgs[b][p]
            }
        }
        resultShape := makeShape(s.dataFormat, batchCount, height, width)
        return result, resultShape, nil
    case "integer":
        batchCount := len(imgs)
        result := make([]int, batchCount*imgSize)
        for b := 0; b < batchCount; b++ {
            for p := 0; p < imgSize; p++ {
                result[b*imgSize+p] = int(imgs[b][p])
            }
        }
        resultShape := makeShape(s.dataFormat, batchCount, height, width)
        return result, resultShape, nil
    default:
        err := fmt.Errorf("Image does not support dtype %s", dtype)
        return nil, nil, err
    }
}

// implementation

func(s *ImageInput) getImage(filename string, shape []int) ([]float32, int, int, error) {
    var targetSize []int
    if len(shape) != 0 {
        if s.dataFormat == DataFormatNCHW {
            if shape[1] != 3 {
                err := 
                    fmt.Errorf(
                        "NCHW image is specified as input, "+
                        "but channel dimension of input tensor is not 3")
                return nil, 0, 0, err
            }
            targetSize = []int{shape[2], shape[3]}
        } else {
            if shape[3] != 3 {
                err := 
                    fmt.Errorf(
                        "NHWC image is specified as input, "+
                        "but channel dimension of input tensor is not 3")
                return nil, 0, 0, err
            }
            targetSize = []int{shape[1], shape[2]}
        }
    }
    img, err := readImage(filename)
    if err != nil {
        return nil, 0, 0, err
    }
    if len(targetSize) != 0 {
        img = resizeImage(img, targetSize)
    }
    data, height, width := s.extractImage(img)
    size := len(data)
    if len(s.rng) != 0 {
        scale := (s.rng[1] - s.rng[0]) / float32(255.0)
        bias := s.rng[0]
        for i := 0; i < size; i++ {
            data[i] = scale * data[i] + bias
        }
    }
    if len(s.norm) != 0 {
        mean := s.norm[0]
        std := s.norm[1]
        hw := size / 3
        if s.dataFormat == DataFormatNCHW {
            for c := 0; c < 3; c++ {
                for i := 0; i < hw; i++ {
                    data[c*hw+i] = (data[c*hw+i] - mean[c]) / std[c]
                }
            }
        } else {
            for i := 0; i < hw; i++ {
                for c := 0; c < 3; c++ {
                    data[i*3+c] = (data[i*3+c] - mean[c]) / std[c]
                }
            }
        }
    }
    return data, height, width, nil
}

func(s *ImageInput) extractImage(img image.Image) ([]float32, int, int) {
    bounds := img.Bounds()
    height := bounds.Max.Y - bounds.Min.Y
    width := bounds.Max.X - bounds.Min.X
    hw := height * width
    data := make([]float32, 3*hw)
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            r, g, b, _ := img.At(x, y).RGBA()
            // scale 16 bits to 8 bits
            r >>= 8
            g >>= 8
            b >>= 8
            fr := float32(r)
            fg := float32(g)
            fb := float32(b)
            if s.dataFormat == DataFormatNCHW {
                p := y * width + x
                if s.colorFormat == ColorFormatRGB {
                    data[p] = fr
                    data[hw+p] = fg
                    data[2*hw+p] = fb 
                } else {
                    data[p] = fb
                    data[hw+p] = fg
                    data[2*hw+p] = fr
                }
            } else {
                p := 3 * (y * width + x)
                if s.colorFormat == ColorFormatRGB {
                    data[p] = fr
                    data[p+1] = fg
                    data[p+2] = fb 
                } else {
                    data[p] = fb
                    data[p+1] = fg
                    data[p+2] = fr
                }
            }
        }
    }
    return data, height, width
}

func readImage(filename string) (image.Image, error) {
    fp, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer fp.Close()
    img, _, err := image.Decode(fp)
    return img, err
}

func resizeImage(img image.Image, targetSize []int) image.Image {
    return resize.Resize(uint(targetSize[0]), uint(targetSize[1]), img, resize.Bilinear)
}

func makeShape(format string, batchCount int, height int, width int) []int {
    if format == DataFormatNCHW {
        return []int{batchCount, 3, height, width}
    } else {
        return []int{batchCount, height, width, 3}
    }
}

