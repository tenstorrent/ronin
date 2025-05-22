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

package main

import (
    "fmt"
    "os"
    "path/filepath"
    "strings"
    src "fragata/arhat/nnef/tools/io/input"
    "fragata/arhat/nnef/tools/util"
)

func main() {
    err := run()
    if err != nil {
        fmt.Fprintf(os.Stderr, "%s\n", err.Error())
        os.Exit(1)
    }
}

func run() error {
    var err error
    args, err := getArgs()
    if err != nil {
        return err
    }
    input, err := expandInput(args.input)
    if err != nil {
        return err
    }
    var norm [][3]float32
    if len(args.mean) != 0 {
        norm = [][3]float32 {
            [3]float32{args.mean[0], args.mean[1], args.mean[2]},
            [3]float32{args.std[0], args.std[1], args.std[2]},
        }
    }
    source := src.NewImageInput(input, args.color, args.format, args.rng, norm)
    var shape []int
    if len(args.size) != 0 {
        if args.format == src.DataFormatNCHW {
            shape = []int{1, 3, args.size[1], args.size[0]}
        } else {
            shape = []int{1, args.size[1], args.size[0], 3}
        }
    }
    data, shape, err := source.CreateInput(args.dtype, shape, true)
    if err != nil {
        return err
    }
    err = util.WriteTensorFile(args.output, args.dtype, shape, data)
    if err != nil {
        return err
    }
    return nil
}

//
//    Local functions
//

func expandInput(input []string) ([]string, error) {
    var result []string
    for _, path := range input {
        if isDir(path) {
            path = filepath.Join(path, "*")
        }
        result = append(result, path)
    }
    return result, nil
}

func isDir(path string) bool {
    info, err := os.Stat(path)
    if err != nil {
        return false
    }
    return info.Mode().IsDir()
}

//
//    Argument parser
//

type Args struct {
    input []string
    output string
    color string
    format string
    rng []float32
    mean []float32
    std []float32
    size []int
    dtype string
}

func getArgs() (*Args, error) {
    var err error
    var input []string
    var output string
    color := src.ColorFormatRGB
    format := src.DataFormatNCHW
    var rng []float32
    var mean []float32
    var std []float32
    var size []int
    dtype := "scalar"
    parser := util.NewArgParser(os.Args)
    input, err = parser.GetStrings(1, -1)
    if err != nil {
        return nil, err
    }
    for !parser.Done() {
        switch option := parser.Option(); option {
        case "--output":
            output, err = parser.GetString()
            if err != nil {
                return nil, err
            }
        case "--color":
            color, err = parser.GetString()
            if err != nil {
                return nil, err
            }
        case "--format":
            format, err = parser.GetString()
            if err != nil {
                return nil, err
            }
        case "--range":
            rng, err = parser.GetFloats(2, 2)
            if err != nil {
                return nil, err
            }
        case "--mean":
            mean, err = parser.GetFloats(1, 3)
            if err != nil {
                return nil, err
            }
        case "--std":
            std, err = parser.GetFloats(1, 3)
            if err != nil {
                return nil, err
            }
        case "--size":
            size, err = parser.GetInts(1, 2)
            if err != nil {
                return nil, err
            }
        case "--dtype":
            dtype, err = parser.GetString()
            if err != nil {
                return nil, err
            }
        default:
            err = fmt.Errorf("Invalid option: %s", option)
            return nil, err
        }
    }
    if output == "" {
        err = fmt.Errorf("Missing '--output' option")
        return nil, err
    }
    color = strings.ToUpper(color)
    if color != src.ColorFormatRGB && color != src.ColorFormatBGR {
        err = fmt.Errorf("Invalid '--color' value: %s", color)
        return nil, err
    }
    format = strings.ToUpper(format)
    if format != src.DataFormatNCHW && format != src.DataFormatNHWC {
        err = fmt.Errorf("Invalid '--format' value: %s", format)
        return nil, err
    }
    // ACHTUNG: Arguments 'rng' and 'mean/std' are optional in this version
    haveNorm := (len(mean) != 0 || len(std) != 0)
    switch len(mean) {
    case 0:
        if haveNorm {
            mean = []float32{0.0, 0.0, 0.0}
        }
    case 1:
        mean = []float32{mean[0], mean[0], mean[0]}
    case 3:
        // ok
    default:
        err = fmt.Errorf("Invalid '--mean' length: %d (must be 1 or 3)", len(mean))
        return nil, err
    }
    switch len(std) {
    case 0:
        if haveNorm {
            std = []float32{1.0, 1.0, 1.0}
        }
    case 1:
        std = []float32{std[0], std[0], std[0]}
    case 3:
        // ok
    default:
        err = fmt.Errorf("Invalid '--std' length: %d (must be 1 or 3)", len(std))
        return nil, err
    }
    switch len(size) {
    case 0:
        // ok
    case 1:
        size = []int{size[0], size[0]}
    }
    if len(size) != 0 {
        if size[0] <= 0 || size[1] <= 0 {
            err = fmt.Errorf("Invalid '--size' value: [%d, %d]", size[0], size[1])
            return nil, err
        }
    }
    dtype = strings.ToLower(dtype)
    args := &Args{
        input: input,
        output: output,
        color: color,
        format: format,
        rng: rng,
        mean: mean,
        std: std,
        size: size,
        dtype: dtype,
    }
    return args, nil
}

