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
//    ValueDict
//

//
// Ordered key-value pairs of arbitrary typed parameter values used for operation attributes
//

type ValueDictItem struct {
    key string
    value Value
}

type ValueDict struct {
    items []ValueDictItem
}

func(d *ValueDict) Add(key string, value Value) {
    d.items = append(d.items, ValueDictItem{key, value})
}

func(d *ValueDict) Contains(key string) bool {
    n := len(d.items)
    for i := 0; i < n;  i++ {
        if d.items[i].key == key {
            return true
        }
    }
    return false
}

func(d *ValueDict) Get(key string, def Value) Value {
    n := len(d.items)
    for i := 0; i < n;  i++ {
        if d.items[i].key == key {
            return d.items[i].value
        }
    }
    return def
}

func(d *ValueDict) Size() int {
    return len(d.items)
}

func(d *ValueDict) At(idx int) Value {
    return d.items[idx].value
}

//
//    Tensor
//

//
// Tensor data structure used both for activation and variable tensors
//
type Tensor struct {
    name string       // name of the tensor in the graph
    dtype string      // data type of the tensor (such as "scalar", "integer", "logical")
    shape []int       // shape of the tensor, filled if shape propagation is in effect
    data interface{}  // array of the data of the tensor, filled in if tensor is a variable
    // quantization algorithm info for both activation and variable tensors
    // used keys: "op-name" (string), attribute names depending on op-name
    quantization ValueDict 
}

func(t *Tensor) SetName(name string) {
    t.name = name
}

func(t *Tensor) Name() string {
    return t.name
}

func(t *Tensor) SetDtype(dtype string) {
    t.dtype = dtype
}

func(t *Tensor) Dtype() string {
    return t.dtype
}

func(t *Tensor) SetShape(shape []int) {
    // unique shape instance is expected (caller shall clone if necessary)
    t.shape = shape
}

func(t *Tensor) Shape() []int {
    return t.shape
}

func(t *Tensor) ResizeData() {
    volume := Shape(t.shape).VolumeOf()
    switch t.dtype {
    case "scalar":
        t.data = make([]float32, volume)
    case "integer":
        t.data = make([]int, volume)
    case "bool":
        t.data = make([]bool, volume)
    default:
        t.data = nil
    }
}

func(t *Tensor) ResizeQuantData() {
    volume := Shape(t.shape).VolumeOf()
    t.data = make([]byte, volume)
}

func(t *Tensor) Data() interface{} {
    return t.data
}

func(t *Tensor) ScalarData() []float32 {
    return t.data.([]float32)
}

func(t *Tensor) IntegerData() []int {
    return t.data.([]int)
}

func(t *Tensor) LogicalData() []bool {
    return t.data.([]bool)
}

func(t *Tensor) QuantData() []byte {
    return t.data.([]byte)
}

func(t *Tensor) Quantization() ValueDict {
    return t.quantization
}

func(t *Tensor) AddQuantization(key string, value Value) {
    t.quantization.Add(key, value)
}

//
//    Operation
//

//
// Operation data structure to represent a single operation in the graph
//
type Operation struct {
    name string       // name (kind) of the operation
    dtype string      // data type in case the operation is generic (such as "scalar", "integer", "logical")
    attribs ValueDict // ordered dictionary of non-tensor attributes of the operation (declaration order)
    inputs ValueDict  // ordered dictionary of tensor inputs of the operation (may also contain constants)
    outputs ValueDict // ordered dictionary tensor outputs of the operation
}

func(o *Operation) SetName(name string) {
    o.name = name
}

func(o *Operation) Name() string {
    return o.name
}

func(o *Operation) SetDtype(dtype string) {
    o.dtype = dtype
}

func(o *Operation) Dtype() string {
    return o.dtype
}

func(o *Operation) AddAttrib(name string, value Value) {
    o.attribs.Add(name, value)
}

func(o *Operation) GetAttrib(name string) Value {
    return o.attribs.Get(name, nil)
}

func(o *Operation) AttribCount() int {
    return o.attribs.Size()
}

func(o *Operation) AttribAt(idx int) Value {
    return o.attribs.At(idx)
}

func(o *Operation) AddInput(name string, value Value) {
    o.inputs.Add(name, value)
}

func(o *Operation) GetInput(name string) Value {
    return o.inputs.Get(name, nil)
}

func(o *Operation) InputCount() int {
    return o.inputs.Size()
}

func(o *Operation) InputAt(idx int) Value {
    return o.inputs.At(idx)
}

func(o *Operation) AddOutput(name string, value Value) {
    o.outputs.Add(name, value)
}

func(o *Operation) GetOutput(name string) Value {
    return o.outputs.Get(name, nil)
}

func(o *Operation) OutputCount() int {
    return o.outputs.Size()
}

func(o *Operation) OutputAt(idx int) Value {
    return o.outputs.At(idx)
}

//
//    Graph
//

//
// Graph data structure, list of tensors and operations
//
type Graph struct {
    name string                  // name of the graph
    tensorMap map[string]*Tensor // table of tensors in the graph
    tensors []*Tensor            // list of tensors in the graph
    operations []*Operation      // list of operations in the graph, in topograpic order
    inputs []string              // list of input tensor ids
    outputs []string             // list of output tensor ids
}

func(g *Graph) SetName(name string) {
    g.name = name
}

func(g *Graph) Name() string {
    return g.name
}

func(g *Graph) ClearTensors() {
    g.tensorMap = make(map[string]*Tensor)
    g.tensors = nil
}

func(g *Graph) AddTensor(name string, tensor *Tensor) {
    g.tensorMap[name] = tensor
    g.tensors = append(g.tensors, tensor)
}

func(g *Graph) GetTensor(name string) *Tensor {
    return g.tensorMap[name]
}

func(g *Graph) TensorCount() int {
    return len(g.tensors)
}

func(g *Graph) TensorAt(idx int) *Tensor {
    return g.tensors[idx]
}

func(g *Graph) AddOperation(operation *Operation) {
    g.operations = append(g.operations, operation)
}

func(g *Graph) ClearOperations() {
    g.operations = nil
}

func (g *Graph) OperationCount() int {
    return len(g.operations)
}

func (g *Graph) OperationAt(idx int) *Operation {
    return g.operations[idx]
}

func(g *Graph) ResizeInputs(size int) {
    g.inputs = make([]string, size)
}

func(g *Graph) SetInput(idx int, input string) {
    g.inputs[idx] = input
}

func(g *Graph) InputCount() int {
    return len(g.inputs)
}

func(g *Graph) InputAt(idx int) string {
    return g.inputs[idx]
}

func(g *Graph) ResizeOutputs(size int) {
    g.outputs = make([]string, size)
}

func(g *Graph) SetOutput(idx int, output string) {
    g.outputs[idx] = output
}

func(g *Graph) OutputCount() int {
    return len(g.outputs)
}

func(g *Graph) OutputAt(idx int) string {
    return g.outputs[idx]
}

