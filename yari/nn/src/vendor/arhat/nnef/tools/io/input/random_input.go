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
    "math"
    "math/rand"
)

//
//    InputSource
//

type InputSource interface {
    CreateInput(dtype string, shape []int, allowBiggerBatch bool) (interface{}, []int, error)
}

//
//    DefaultInput
//

type DefaultInput struct {}

func NewDefaultInput() *DefaultInput {
    return new(DefaultInput)
}

var (
    defaultScalar = NewNormalInput(0.0, 1.0)
    defaultInteger = NewBinomialInput(255, 0.5)
    defaultLogical = NewBernoulliInput(0.5)
)

func(s *DefaultInput) CreateInput(
        dtype string, shape []int, allowBiggerBatch bool) (interface{}, []int, error) {
    switch dtype {
    case "scalar":
        return defaultScalar.CreateInput(dtype, shape, allowBiggerBatch)
    case "integer":
        return defaultInteger.CreateInput(dtype, shape, allowBiggerBatch)
    case "logical":
        return defaultLogical.CreateInput(dtype, shape, allowBiggerBatch)
    default:
        err := fmt.Errorf("Random does not support dtype %s", dtype)
        return nil, nil, err
    }
}

//
//    UniformInput
//

type UniformInput struct {
    min float32
    max float32
}

func NewUniformInput(min float32, max float32) *UniformInput {
    return &UniformInput{min, max}
}

func(s *UniformInput) CreateInput(
        dtype string, shape []int, allowBiggerBatch bool) (interface{}, []int, error) {
    switch dtype {
    case "scalar":
        n := volumeOf(shape)
        data := make([]float32, n)
        a := s.max - s.min
        b := s.min
        for i := 0; i < n; i++ {
            data[i] = a * rand.Float32() + b
        }
        return data, shape, nil
    case "integer":
        n := volumeOf(shape)
        data := make([]int, n)
        smin := int(math.Ceil(float64(s.min)))
        smax := int(math.Floor(float64(s.max))) + 1
        a := smax - smin
        b := smin
        for i := 0; i < n; i++ {
            data[i] = a * rand.Intn(1) + b
        }
        return data, shape, nil
    default:
        err := fmt.Errorf("Random 'uniform' can not be applied to %s", dtype)
        return nil, nil, err
    }
}

//
//    NormalInput
//

type NormalInput struct {
    mean float32
    std float32
}

func NewNormalInput(mean float32, std float32) *NormalInput {
    return &NormalInput{mean, std}
}

func(s *NormalInput) CreateInput(
        dtype string, shape []int, allowBiggerBatch bool) (interface{}, []int, error) {
    if dtype == "scalar" {
        n := volumeOf(shape)
        data := make([]float32, n)
        a := s.std
        b := s.mean
        for i := 0; i < n; i++ {
            data[i] = a * float32(rand.NormFloat64()) + b
        }
        return data, shape, nil
    } else {
        err := fmt.Errorf("Random 'normal' can not be applied to %d", dtype)
        return nil, nil, err
    }
}

//
//    BinomialInput
//

type BinomialInput struct {
    num int
    trueProb float32
}

func NewBinomialInput(num int, trueProb float32) *BinomialInput {
    return &BinomialInput{num, trueProb}
}

func(s *BinomialInput) CreateInput(
        dtype string, shape []int, allowBiggerBatch bool) (interface{}, []int, error) {
    if dtype == "integer" {
        n := volumeOf(shape)
        data := make([]int, n)
        sn := float64(s.num)
        sp := float64(s.trueProb)
        for i := 0; i < n; i++ {
            data[i] = int(Binomial(sn, sp))
        }
        return data, shape, nil
    } else {
        err := fmt.Errorf("Random 'binomial' can not be applied to %s", dtype)
        return nil, nil, err
    }
}

//
//    BernoulliInput
//

type BernoulliInput struct {
    trueProb float32
}

func NewBernoulliInput(trueProb float32) *BernoulliInput {
    return &BernoulliInput{trueProb}
}

func(s *BernoulliInput) CreateInput(
        dtype string, shape []int, allowBiggerBatch bool) (interface{}, []int, error) {
    if dtype == "logical" {
        n := volumeOf(shape)
        data := make([]bool, n)
        sp := float64(s.trueProb)
        for i := 0; i < n; i++ {
            data[i] = (rand.NormFloat64() <= sp)
        }
        return data, shape, nil
    } else {
        err := fmt.Errorf("Random 'bernoulli' can not be applied to %s", dtype)
        return nil, nil, err
    }
}

//
//    Local functions
//

func volumeOf(shape []int) int {
    volume := 1
    for _, dim := range shape {
        volume *= dim
    }
    return volume
}

