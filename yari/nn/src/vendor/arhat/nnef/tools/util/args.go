//
// Copyright (c) 2019-2020 FRAGATA COMPUTER SYSTEMS AG
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

package util

import (
    "fmt"
    "strconv"
)

//
//    ArgParser
//

type ArgParser struct {
    argv []string
    index int
    positional bool
}

// construction/destruction

func NewArgParser(argv []string) *ArgParser {
    return &ArgParser{argv, 0, true}
}

// interface

func(p *ArgParser) Done() bool {
    return p.index >= len(p.argv)
}

func(p *ArgParser) Option() string {
    return p.argv[p.index]
}

func(p *ArgParser) GetInt() (int, error) {
    err := p.checkArg()
    if err != nil {
        return 0, err
    }
    next := p.index + 1
    value, err := strconv.Atoi(p.argv[next])
    if err != nil {
        err = p.invalidValue(next)
        return 0, err
    }
    p.index = next + 1
    return value, nil
}

func(p *ArgParser) GetInts(minCount int, maxCount int) ([]int, error) {
    count, err := p.checkArgs(minCount, maxCount)
    if err != nil {
        return nil, err
    }
    result := make([]int, count)
    next := p.index + 1
    for i := 0; i < count; i++ {
        value, err := strconv.Atoi(p.argv[next])
        if err != nil {
            err = p.invalidValue(next)
            return nil, err
        }
        result[i] = value
        next++
    }
    p.index = next
    return result, nil
}

func(p *ArgParser) GetFloat() (float32, error) {
    err := p.checkArg()
    if err != nil {
        return 0, err
    }
    next := p.index + 1
    value, err := strconv.ParseFloat(p.argv[next], 32)
    if err != nil {
        err = p.invalidValue(next)
        return 0.0, err
    }
    p.index = next + 1
    return float32(value), nil
}

func(p *ArgParser) GetFloats(minCount int, maxCount int) ([]float32, error) {
    count, err := p.checkArgs(minCount, maxCount)
    if err != nil {
        return nil, err
    }
    result := make([]float32, count)
    next := p.index + 1
    for i := 0; i < count; i++ {
        value, err := strconv.ParseFloat(p.argv[next], 32)
        if err != nil {
            err = p.invalidValue(next)
            return nil, err
        }
        result[i] = float32(value)
        next++
    }
    p.index = next
    return result, nil
}

func(p *ArgParser) GetString() (string, error) {
    err := p.checkArg()
    if err != nil {
        return "", err
    }
    next := p.index + 1
    value := p.argv[next]
    p.index = next + 1
    return value, nil
}

func(p *ArgParser) GetStrings(minCount int, maxCount int) ([]string, error) {
    count, err := p.checkArgs(minCount, maxCount)
    if err != nil {
        return nil, err
    }
    result := make([]string, count)
    next := p.index + 1
    for i := 0; i < count; i++ {
        result[i] = p.argv[next]
        next++
    }
    p.index = next
    return result, nil
}

// implementation

func(p *ArgParser) checkArg() error {
    p.checkPositional()
    argc := len(p.argv)
    next := p.index + 1
    if next >= argc {
        return fmt.Errorf("%s: missing value", p.location())
    }
    if !p.positional {
        next++
        if next < argc && p.argv[next][0] != '-' {
            return fmt.Errorf("%s: invalid number of values", p.location())
        }
    }
    return nil
}

func(p *ArgParser) checkArgs(minCount int, maxCount int) (int, error) {
    p.checkPositional()
    argc := len(p.argv)    
    next := p.index + 1
    for next < argc && p.argv[next][0] != '-' {
        next++
    }
    count := next - p.index - 1
    if count < minCount || (maxCount >= 0 && count > maxCount) {
        err := fmt.Errorf("%s: invalid number of values", p.location())
        return 0, err
    }
    return count, nil
}

func(p *ArgParser) checkPositional() {
    if p.argv[p.index][0] == '-' {
        p.positional = true
    }
}

func(p *ArgParser) invalidValue(index int) error {
    return fmt.Errorf("%s: invalid value: %s", p.location(), p.argv[index])
}

func(p *ArgParser) location() string {
    if p.positional {
        return fmt.Sprintf("Argument %d", p.index)
    } else {
        return fmt.Sprintf("Option %s", p.argv[p.index])
    }
}

