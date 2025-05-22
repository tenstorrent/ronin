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

 func IsCastable(type1 Type, type2 Type, allowPrimitiveToTensor bool) bool {
    if type1 == type2 {
        return true
    }
    if type1.Kind() == type2.Kind() {
        switch type1.Kind() {
        case TypeKindPrimitive:
            primitiveType1 := type1.(*PrimitiveType)
            primitiveType2 := type2.(*PrimitiveType)
            return (primitiveType1.Name() == primitiveType2.Name() || 
                primitiveType2.Name() == TypenameGeneric)
        case TypeKindTensor:
            tensorType1 := type1.(*TensorType)
            tensorType2 := type2.(*TensorType)
            dataType1 := tensorType1.DataType()
            dataType2 := tensorType2.DataType()
            if dataType1 != nil && dataType2 != nil {
                return IsCastable(dataType1, dataType2, allowPrimitiveToTensor)
            } else {
                return (dataType2 == nil)
            }
        case TypeKindArray:
            arrayType1 := type1.(*ArrayType)
            arrayType2 := type2.(*ArrayType)
            itemType1 := arrayType1.ItemType()
            itemType2 := arrayType2.ItemType()
            if itemType1 != nil && itemType2 != nil {
                return IsCastable(itemType1, itemType2, allowPrimitiveToTensor)
            } else {
                return (itemType1 == nil)
            }
        case TypeKindTuple:
            tupleType1 := type1.(*TupleType)
            tupleType2 := type2.(*TupleType)
            if tupleType1.Size() != tupleType2.Size() {
                return false
            }
            for i := 0; i < tupleType1.Size(); i++ {
                if !IsCastable(tupleType1.ItemType(i), tupleType2.ItemType(i), allowPrimitiveToTensor) {
                    return false
                }
            }
            return true
        }
    } else if type1.Kind() == TypeKindPrimitive && 
            type2.Kind() == TypeKindTensor && allowPrimitiveToTensor {
        tensorType := type2.(*TensorType)
        dataType := tensorType.DataType()
        return (dataType == nil || IsCastable(type1, dataType, true))
    }
    return false
}

func CommonType(type1 Type, type2 Type, allowPrimitiveToTensor bool) Type {
    if IsCastable(type1, type2, allowPrimitiveToTensor) {
        return type2
    }
    if IsCastable(type2, type1, allowPrimitiveToTensor) {
        return type1
    }
    return nil
}

 func BindDataType(paramType Type, dataType *PrimitiveType) Type {
    generic := GetPrimitiveType(TypenameGeneric)
    if !paramType.IsGeneric() || dataType == generic {
        return paramType
    }    
    switch paramType.Kind() {
    case TypeKindPrimitive:
        if paramType == generic {
            return dataType
        } else {
            return paramType
        }
    case TypeKindTensor:
        tensor := paramType.(*TensorType)
        if tensor.DataType() == generic {
            return GetTensorType(dataType.Name())
        } else {
            return paramType
        }
    case TypeKindArray:
        array := paramType.(*ArrayType)
        itemType := array.ItemType()
        if itemType != nil {
            return GetArrayType(BindDataType(itemType, dataType))
        } else {
            return paramType
        }
    case TypeKindTuple:
        tuple := paramType.(*TupleType)
        size := tuple.Size()
        itemTypes := make([]Type, size)
        for i := 0; i < size; i++ {
            itemTypes[i] = BindDataType(tuple.ItemType(i), dataType)
        }
        return GetTupleType(itemTypes)
    default:
        Assert(false)
        return nil
    }
}

func DeduceParamDataType(
        paramType Type, argType Type, dataType *PrimitiveType) (
            *PrimitiveType, []Typename) {
    var mismatch []Typename
    if paramType.Kind() == argType.Kind() {
        switch paramType.Kind() {
        case TypeKindPrimitive:
            if paramType.IsGeneric() {
                primitiveType := argType.(*PrimitiveType)
                if dataType == nil {
                    dataType = primitiveType
                } else if dataType != argType {
                    mismatch = []Typename{dataType.Name(), primitiveType.Name()}
                }
            }
        case TypeKindTensor:
            tensorType1 := paramType.(*TensorType)
            tensorType2 := argType.(*TensorType)
            dataType1 := tensorType1.DataType()
            dataType2 := tensorType2.DataType()
            if dataType1 != nil && dataType2 != nil {
                dataType, mismatch = DeduceParamDataType(dataType1, dataType2, dataType)
            }
        case TypeKindArray:
            arrayType1 := paramType.(*ArrayType)
            arrayType2 := argType.(*ArrayType)
            itemType1 := arrayType1.ItemType()
            itemType2 := arrayType2.ItemType()
            if itemType1 != nil && itemType2 != nil {
                dataType, mismatch = DeduceParamDataType(itemType1, itemType2, dataType)
            }
        case TypeKindTuple:
            tupleType1 := paramType.(*TupleType)
            tupleType2 := argType.(*TupleType)
            Assert(tupleType1.Size() == tupleType2.Size())
            for i := 0; i < tupleType1.Size(); i++ {
                dataType, mismatch = 
                    DeduceParamDataType(tupleType1.ItemType(i), tupleType2.ItemType(i), dataType)
                if mismatch != nil {
                    break
                }
            }
        }
    } else if paramType.Kind() == TypeKindTensor && argType.Kind() == TypeKindPrimitive {
        tensorType := paramType.(*TensorType)
        dataType, mismatch = DeduceParamDataType(tensorType.DataType(), argType, dataType)
    }
    return dataType, mismatch
}

func DeduceDataType(
        proto *Prototype, types map[string]Type, dataType *PrimitiveType) (
            *PrimitiveType, []Typename) {
    var mismatch []Typename
    for i := 0; i < proto.ParamCount(); i++ {
        param := proto.ParamAt(i)
        if param.Type().IsGeneric() {
            argType := types[param.Name()]
            dataType, mismatch = DeduceParamDataType(param.Type(), argType, dataType)
            if mismatch != nil {
                break
            }
        }
    }
    return dataType, mismatch
}

