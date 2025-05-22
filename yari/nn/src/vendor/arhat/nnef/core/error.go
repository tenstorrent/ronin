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

import "fmt"

//
//    standard errors
//

func InvalidArgument(format string, args ...interface{}) {
    panic(makeError("InvalidArgument", format, args))
}

func LogicError(format string, args ...interface{}) {
    panic(makeError("LogicError", format, args))
}

func RuntimeError(format string, args ...interface{}) {
    panic(makeError("RuntimeError", format, args))
}

func Raise(err error) {
    panic(err)
}

func Assert(cond bool) {
    if !cond {
        panic("AssertionError")
    }
}

func makeError(kind string, format string, args []interface{}) error {
    message := fmt.Sprintf(format, args...)
    return fmt.Errorf("%s: %s", kind, message)
}

//
//    Position
//

type Position struct {
    Line uint
    Column uint
    Filename string
    Origin *Position
}

func(p *Position) Clone() *Position {
    r := new(Position)
    *r = *p
    return r
}

//
//    Error
//

type Error struct {
    position *Position
    message string
}

func NewError(position *Position, format string, args ...interface{}) *Error {
    e := new(Error)
    if position != nil {
        e.position = position.Clone()
    } else {
        e.position = &Position{}
    }
    e.message = fmt.Sprintf(format, args...)
    return e
}

func(e *Error) Position() *Position {
    return e.position
}

func(e *Error) What() string {
    return e.message
}

func RaiseError(position *Position, format string, args ...interface{}) {
    panic(NewError(position, format, args...))
}

