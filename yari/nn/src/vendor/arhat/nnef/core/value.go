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
    "strings"
)

//
//    Items
//

type Items []Value

func(v Items) Clone() Items {
    n := len(v)
    r := make(Items, n)
    for i := 0; i < n; i++ {
        r[i] = v[i].Clone()
    }
    return r
}

func(v Items) Eq(other Items) bool {
    n := len(v)
    if len(other) != n {
        return false
    }
    for i := 0; i < n; i++ {
        if !v[i].Eq(other[i]) {
            return false
        }
    }
    return true
}

func(v Items) Repr() string {
    r := ""
    for i, p := range v {
        if i != 0 {
            r += ","
        }
        r += p.Repr()
    }
    return r
}

//
//    ValueKind
//

type ValueKind int

const (
    ValueKindNone ValueKind = iota
    ValueKindInteger
    ValueKindScalar
    ValueKindLogical
    ValueKindString
    ValueKindIdentifier
    ValueKindArray
    ValueKindTuple
)

//
//    Value
//

type Value interface {
    Kind() ValueKind
    IsNone() bool
    Integer() int
    Scalar() float32
    Logical() bool
    String() string
    Identifier() string
    Array() Items
    Tuple() Items
    Items() Items
    Clone() Value
    Size() int
    At(idx int) Value
    Eq(other Value) bool
    Ne(other Value) bool
    Repr() string
}

//
//    ValueBase
//

type ValueBase struct {}

func(v *ValueBase) Integer() int {
    kindMismatch()
    return 0
}

func(v *ValueBase) IsNone() bool {
    return false
}

func(v *ValueBase) Scalar() float32 {
    kindMismatch()
    return 0.0
}

func(v *ValueBase) Logical() bool {
    kindMismatch()
    return false
}

func(v *ValueBase) String() string {
    kindMismatch()
    return ""
}

func(v *ValueBase) Identifier() string {
    kindMismatch()
    return ""
}

func(v *ValueBase) Array() Items {
    kindMismatch()
    return nil
}

func(v *ValueBase) Tuple() Items {
    kindMismatch()
    return nil
}

func(v *ValueBase) Items() Items {
    expectedItems()
    return nil
}

func(v *ValueBase) Size() int {
    expectedItems()
    return 0
}

func(v *ValueBase) At(idx int) Value {
    expectedItems()
    return nil
}

func kindMismatch() {
    InvalidArgument("Value: kind mismatch")
}

func expectedItems() {
    InvalidArgument("Value: expected items")
}

//
//    NoneValue
//

type NoneValue struct {
    ValueBase
}

var none = &NoneValue{}

func None() *NoneValue {
    return none
}

func(v *NoneValue) Kind() ValueKind {
    return ValueKindNone
}

func(v *NoneValue) IsNone() bool {
    return true
}

func(v *NoneValue) Clone() Value {
    return none
}

func(v *NoneValue) Eq(other Value) bool {
    _, ok := other.(*NoneValue)
    return ok
}

func(v *NoneValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *NoneValue) Repr() string {
    return "none"
}

//
//    IntegerValue
//

type IntegerValue struct {
    ValueBase
    value int
}

func NewIntegerValue(value int) *IntegerValue {
    return &IntegerValue{value: value}
}

func(v *IntegerValue) Kind() ValueKind {
    return ValueKindInteger
}

func(v *IntegerValue) Integer() int {
    return v.value
}

func(v *IntegerValue) Clone() Value {
    return NewIntegerValue(v.value)
}

func(v *IntegerValue) Eq(other Value) bool {
    p, ok := other.(*IntegerValue)
    if !ok {
        return false
    }
    return (v.value == p.value)
}

func(v *IntegerValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *IntegerValue) Repr() string {
    return fmt.Sprintf("%d", v.value)
}

//
//    ScalarValue
//

type ScalarValue struct {
    ValueBase
    value float32
}

func NewScalarValue(value float32) *ScalarValue {
    return &ScalarValue{value: value}
}

func(v *ScalarValue) Kind() ValueKind {
    return ValueKindScalar
}

func(v *ScalarValue) Scalar() float32 {
    return v.value
}

func(v *ScalarValue) Clone() Value {
    return NewScalarValue(v.value)
}

func(v *ScalarValue) Eq(other Value) bool {
    p, ok := other.(*ScalarValue)
    if !ok {
        return false
    }
    return (v.value == p.value)
}

func(v *ScalarValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *ScalarValue) Repr() string {
    r := fmt.Sprintf("%g", v.value)
    if !strings.ContainsAny(r, ".e") {
        r += ".0"
    }
    return r 
}

//
//    LogicalValue
//

type LogicalValue struct {
    ValueBase
    value bool
}

func NewLogicalValue(value bool) *LogicalValue {
    return &LogicalValue{value: value}
}

func(v *LogicalValue) Kind() ValueKind {
    return ValueKindLogical
}

func(v *LogicalValue) Logical() bool {
    return v.value
}

func(v *LogicalValue) Clone() Value {
    return NewLogicalValue(v.value)
}

func(v *LogicalValue) Eq(other Value) bool {
    p, ok := other.(*LogicalValue)
    if !ok {
        return false
    }
    return (v.value == p.value)
}

func(v *LogicalValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *LogicalValue) Repr() string {
    if v.value {
        return "true"
    } else {
        return "false"
    }
}

//
//    StringValue
//

type StringValue struct {
    ValueBase
    value string
}

func NewStringValue(value string) *StringValue {
    return &StringValue{value: value}
}

func(v *StringValue) Kind() ValueKind {
    return ValueKindString
}

func(v *StringValue) String() string {
    return v.value
}

func(v *StringValue) Clone() Value {
    return NewStringValue(v.value)
}

func(v *StringValue) Eq(other Value) bool {
    p, ok := other.(*StringValue)
    if !ok {
        return false
    }
    return (v.value == p.value)
}

func(v *StringValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *StringValue) Repr() string {
    return "'" + v.value + "'"
}

//
//    IdentifierValue
//

type IdentifierValue struct {
    ValueBase
    value string
}

func NewIdentifierValue(value string) *IdentifierValue {
    return &IdentifierValue{value: value}
}

func(v *IdentifierValue) Kind() ValueKind {
    return ValueKindIdentifier
}

func(v *IdentifierValue) Identifier() string {
    return v.value
}

func(v *IdentifierValue) Clone() Value {
    return NewIdentifierValue(v.value)
}

func(v *IdentifierValue) Eq(other Value) bool {
    p, ok := other.(*IdentifierValue)
    if !ok {
        return false
    }
    return (v.value == p.value)
}

func(v *IdentifierValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *IdentifierValue) Repr() string {
    return v.value
}

//
//    ItemsValue
//

type ItemsValue struct {
    ValueBase
    value Items
}

func(v *ItemsValue) Init(value Items, clone bool) {
    if clone && value != nil {
        v.value = value.Clone()
    } else {
        v.value = value
    }
}

func(v *ItemsValue) Items() Items {
    return v.value
}

func(v *ItemsValue) Size() int {
    return len(v.value)
}

func(v *ItemsValue) At(idx int) Value {
    return v.value[idx]
}

//
//    ArrayValue
//

type ArrayValue struct {
    ItemsValue
}

func NewArrayValue(value Items, clone bool) *ArrayValue {
    v := new(ArrayValue)
    v.ItemsValue.Init(value, clone)
    return v
}

func(v *ArrayValue) Kind() ValueKind {
    return ValueKindArray
}

func(v *ArrayValue) Array() Items {
    return v.value
}

func(v *ArrayValue) Clone() Value {
    return NewArrayValue(v.value, true)
}

func(v *ArrayValue) Eq(other Value) bool {
    if v == other {
        return true
    }
    p, ok := other.(*ArrayValue)
    if !ok {
        return false
    }
    return v.value.Eq(p.value)
}

func(v *ArrayValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *ArrayValue) Repr() string {
    return "[" + v.value.Repr() + "]"
}

//
//    TupleValue
//

type TupleValue struct {
    ItemsValue
}

func NewTupleValue(value Items, clone bool) *TupleValue {
    v := new(TupleValue)
    v.ItemsValue.Init(value, clone)
    return v
}

func(v *TupleValue) Kind() ValueKind {
    return ValueKindTuple
}

func(v *TupleValue) Tuple() Items {
    return v.value
}

func(v *TupleValue) Clone() Value {
    return NewTupleValue(v.value, true)
}

func(v *TupleValue) Eq(other Value) bool {
    if v == other {
        return true
    }
    p, ok := other.(*TupleValue)
    if !ok {
        return false
    }
    return v.value.Eq(p.value)
}

func(v *TupleValue) Ne(other Value) bool {
    return !v.Eq(other)
}

func(v *TupleValue) Repr() string {
    return "(" + v.value.Repr() + ")"
}

