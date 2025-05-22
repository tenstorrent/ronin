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

package util

import (
    "fmt"
    "io"
    "os"
    nnef "fragata/arhat/nnef/core"
)

//
//    Tensor I/O
//

// interface

func WriteTensorFile(
        filename string, 
        dtype string, 
        shape []int, 
        data interface{}) (err error) {
    fp, err := os.Create(filename)
    if err != nil {
        return err
    }
    return WriteTensor(fp, dtype, shape, data)
}

func WriteTensor(
        fp io.Writer, 
        dtype string, 
        shape []int, 
        data interface{}) (err error) {
    defer func() {
        if r := recover(); r != nil {
            if v, ok := r.(error); ok {
                err = v
            } else {
                panic(r)
            }
        }
    }()
    rank := len(shape)
    if rank > nnef.MaxRank {
        return fmt.Errorf("Tensor rank %d exceeds maximum allowed rank (%d)", rank, nnef.MaxRank)
    }
/* TODO: Revise this
    var quantCode nnef.QuantCode
    if dtype == "scalar" {
        quantCode = nnef.QuantCodeFloat
    } else {
        quantCode = nnef.QuantCodeInteger
    }   
*/
    var itemType nnef.ItemType
    if dtype == "scalar" {
        itemType = nnef.ItemTypeFloat
    } else {
        itemType = nnef.ItemTypeInt
    }   
    var header nnef.TensorHeader
    version := [2]int{1, 0}
/* TODO: Revise this
    header.Fill(version, shape, itemBits(dtype), quantCode)
*/
    header.Fill(version, shape, itemBits(dtype), itemType)
    header.Write(fp)
    switch dtype {
    case "scalar":
        nnef.WriteScalarData(fp, data.([]float32))
    case "integer":
        nnef.WriteIntegerData(fp, data.([]int), true) // write as signed - is this correct?
    case "logical":
        nnef.WriteLogicalData(fp, data.([]bool))
    default:
        return fmt.Errorf("Invalid tensor data type: %s", dtype)
    }
    return
}

// local functions

func itemBits(dtype string) int {
    switch dtype {
    case "scalar":
        return 32
    case "integer":
        return 32
    case "logical":
        return 1
    default:
        return 0
    }
}

