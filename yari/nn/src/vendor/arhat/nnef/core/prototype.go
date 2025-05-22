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
//    Typed
//

type Typed struct {
    name string
    typ Type
}

func(t *Typed) Init(name string, typ Type) {
    t.name = name
    t.typ = typ
}

func(t *Typed) Name() string {
    return t.name
}

func(t *Typed) Type() Type {
    return t.typ
}

func(t *Typed) String() string {
    return t.name + ": " + t.typ.String()
}

//
//    Param
//

type Param struct {
    Typed
    defaultValue Value
}

func NewParam(name string, typ Type, defaultValue Value) *Param {
    p := new(Param)
    p.Init(name, typ)
    p.defaultValue = defaultValue
    return p
}

func(p *Param) DefaultValue() Value {
    return p.defaultValue
}

//
//    Result
//

type Result struct {
    Typed
}

func NewResult(name string, typ Type) *Result {
    r := new(Result)
    r.Init(name, typ)
    return r
}

//
//    Prototype
//

type Prototype struct {
    name string
    params []*Param
    results []*Result
    hasGenericParams bool
    hasGenericResults bool
    genericParamDefault *PrimitiveType
}

func NewPrototype(
        name string, 
        params []*Param, 
        results []*Result, 
        genericParamDefault *PrimitiveType) *Prototype {
    p := new(Prototype)
    p.name = name
    n := len(params)
    if n != 0 {
        p.params = make([]*Param, n)
        copy(p.params, params)
    }
    n = len(results)
    if n != 0 {
        p.results = make([]*Result, n)
        copy(p.results, results)
    }
    p.genericParamDefault = genericParamDefault
    p.initGeneric()
    return p
}

func(p *Prototype) initGeneric() {
    p.hasGenericParams = false
    for _, v := range p.params {
        if v.Type().IsGeneric() {
            p.hasGenericParams = true
            break
        }
    }
    p.hasGenericResults = false
    for _, v := range p.results {
        if v.Type().IsGeneric() {
            p.hasGenericResults = true
            break
        }
    }
}

func(p *Prototype) Name() string {
    return p.name
}

func(p *Prototype) ParamCount() int {
    return len(p.params)
}

func(p *Prototype) ParamAt(idx int) *Param {
    return p.params[idx]
}

func(p *Prototype) GetParam(name string) *Param {
    for _, param := range p.params {
        if param.Name() == name {
            return param
        }
    }
    return nil
}

func(p *Prototype) ResultCount() int {
    return len(p.results)
}

func(p *Prototype) ResultAt(idx int) *Result {
    return p.results[idx]
}

func(p *Prototype) GetResult(name string) *Result {
    for _, result := range p.results {
        if result.Name() == name {
            return result
        }
    }
    return nil
}

func(p *Prototype) GenericParamDefault() *PrimitiveType {
    return p.genericParamDefault
}

func(p *Prototype) HasGenericParams() bool {
    return p.hasGenericParams
}

func(p *Prototype) HasGenericResults() bool {
    return p.hasGenericResults
}

func(p *Prototype) IsGeneric() bool {
    return p.hasGenericParams || p.hasGenericResults
}

func(p *Prototype) String() string {
    s := p.name
    if p.IsGeneric() {
        s += "<?"
        if p.genericParamDefault != nil{
            s += " = " + p.genericParamDefault.String()
        }
        s += ">"
    }
    s += "("
    for i, v := range p.params {
        if i != 0 {
            s += ", "
        }
        s += v.String()
    }
    s += ")"
    s += " -> "
    s += "("
    for i, v := range p.results {
        if i != 0 {
            s += ", "
        }
        s += v.String()
    }
    s += ")"
    return s
}

