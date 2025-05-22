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

//
//    Typename
//

type Typename int

const (
    TypenameInteger Typename = iota
    TypenameScalar
    TypenameLogical
    TypenameString
    TypenameGeneric
)

var typenameStrings = []string{
    TypenameInteger: "integer",
    TypenameScalar: "scalar",
    TypenameLogical: "logical",
    TypenameString: "string",
    TypenameGeneric: "?",
}

func(t Typename) String() string {
    return typenameStrings[t]
}

//
//    TypeKind
//

type TypeKind int

const (
    TypeKindPrimitive TypeKind = iota
    TypeKindTensor
    TypeKindArray
    TypeKindTuple
)

//
//    Type
//

type Type interface {
    Kind() TypeKind
    IsAttribute() bool
    IsGeneric() bool
    String() string
}

//
//    PrimitiveType
//

type PrimitiveType struct {
    name Typename
}

func NewPrimitiveType(name Typename) *PrimitiveType {
    return &PrimitiveType{name}
}

func(t *PrimitiveType) Name() Typename {
    return t.name
}

func(t *PrimitiveType) Kind() TypeKind {
    return TypeKindPrimitive
}

func(t *PrimitiveType) IsAttribute() bool {
    return true
}

func(t *PrimitiveType) IsGeneric() bool {
    return (t.name == TypenameGeneric)
}

func(t *PrimitiveType) String() string {
    return t.name.String()
}

//
//    TensorType
//

type TensorType struct {
    dataType Type
}

func NewTensorType(dataType Type) *TensorType {
    return &TensorType{dataType}
}

func(t *TensorType) DataType() Type {
    return t.dataType
}

func(t *TensorType) Kind() TypeKind {
    return TypeKindTensor
}

func(t *TensorType) IsAttribute() bool {
    return false
}

func(t *TensorType) IsGeneric() bool {
    return (t.dataType != nil && t.dataType.IsGeneric())
}

func(t *TensorType) String() string {
    if t.dataType != nil {
        return "tensor<" + t.dataType.String() + ">"
    } else {
        return "tensor<>"
    }
}

//
//    ArrayType
//

type ArrayType struct {
    itemType Type
}

func NewArrayType(itemType Type) *ArrayType {
    return &ArrayType{itemType}
}

func(t *ArrayType) ItemType() Type {
    return t.itemType
}

func(t *ArrayType) Kind() TypeKind {
    return TypeKindArray
}

func(t *ArrayType) IsAttribute() bool {
    return (t.itemType != nil && t.itemType.IsAttribute())
}

func(t *ArrayType) IsGeneric() bool {
    return (t.itemType != nil && t.itemType.IsGeneric())
}

func(t *ArrayType) String() string {
    if t.itemType != nil {
        return t.itemType.String() + "[]"
    } else {
        return "[]"
    }
}

//
//    TupleType
//

type TupleType struct {
    itemTypes []Type
}

func NewTupleType(itemTypes []Type) *TupleType {
    return &TupleType{itemTypes}
}

func(t *TupleType) Size() int {
    return len(t.itemTypes)
}

func(t *TupleType) ItemType(idx int) Type {
    return t.itemTypes[idx]
}

func(t *TupleType) Kind() TypeKind {
    return TypeKindTuple
}

func(t *TupleType) IsAttribute() bool {
    for _, p := range t.itemTypes {
        if !p.IsAttribute() {
            return false
        }
    }
    return true
}

func(t *TupleType) IsGeneric() bool {
    for _, p := range t.itemTypes {
        if p.IsGeneric() {
            return true
        }
    }
    return false
}

func(t *TupleType) String() string {
    return "(" + itemTypesString(t.itemTypes) + ")"
}

func itemTypesString(itemTypes []Type) string {
    s := ""
    for i, p := range itemTypes {
        if i != 0 {
            s += ","
        }
        if p != nil {
            s += p.String()
        } else {
            // special case of lvalue tuples
            // occurs when computing hashes for GetTupleType
            s += "*"
        }
    }
    return s
}

//
//    Functions
//

var primitiveTypes = []*PrimitiveType{
    TypenameInteger: NewPrimitiveType(TypenameInteger),
    TypenameScalar: NewPrimitiveType(TypenameScalar),
    TypenameLogical: NewPrimitiveType(TypenameLogical),
    TypenameString: NewPrimitiveType(TypenameString),
    TypenameGeneric: NewPrimitiveType(TypenameGeneric),
}

func GetPrimitiveType(name Typename) *PrimitiveType {
    return primitiveTypes[name]
}

var tensorTypes = []*TensorType{
    TypenameInteger: NewTensorType(primitiveTypes[TypenameInteger]),
    TypenameScalar: NewTensorType(primitiveTypes[TypenameScalar]),
    TypenameLogical: NewTensorType(primitiveTypes[TypenameLogical]),
    TypenameString: NewTensorType(primitiveTypes[TypenameString]),
    TypenameGeneric: NewTensorType(primitiveTypes[TypenameGeneric]),    
}

func GetTensorType(name Typename) *TensorType {
    return tensorTypes[name]
}

var unboundTensorType = NewTensorType(nil)

func UnboundTensorType() *TensorType {
    return unboundTensorType
}

var arrayTypes = make(map[Type]*ArrayType)

func GetArrayType(itemType Type) *ArrayType {
    t, ok := arrayTypes[itemType]
    if !ok {
        t = NewArrayType(itemType)
        arrayTypes[itemType] = t
    }
    return t
}

var tupleTypes = make(map[string]*TupleType)

func GetTupleType(itemTypes []Type) *TupleType {
    s := itemTypesString(itemTypes)
    t, ok := tupleTypes[s]
    if !ok {
        t = NewTupleType(itemTypes)
        tupleTypes[s] = t
    }
    return t
}

