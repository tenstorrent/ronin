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
    "bufio"
    "io"
    "strconv"
    "strings"
)

//
//    Token
//

type Token int

const (
    TokenEof Token = iota
    TokenVersion
    TokenExtension
    TokenIdentifier
    TokenCharacters
    TokenDecimal
    TokenFractional
    TokenGraph
    TokenFragment
    TokenTensor
    TokenInteger
    TokenScalar
    TokenLogical
    TokenString
    TokenTrue
    TokenFalse
    TokenFor
    TokenIn
    TokenIf
    TokenElse
    TokenYield
    TokenLengthOf
    TokenShapeOf
    TokenRangeOf
    TokenArrow
    TokenAnd
    TokenOr
    TokenLe
    TokenGe
    TokenEq
    TokenNe
)

var tokenStrings = []string{
    TokenEof: "eof",
    TokenVersion: "version",
    TokenExtension: "extension",
    TokenIdentifier: "identifier",
    TokenCharacters: "literal",
    TokenDecimal: "decimal",
    TokenFractional: "fractional",
    TokenGraph: "graph",
    TokenFragment: "fragment",
    TokenTensor: "tensor",
    TokenInteger: "integer",
    TokenScalar: "scalar",
    TokenLogical: "logical",
    TokenString: "string",
    TokenTrue: "true",
    TokenFalse: "false",
    TokenFor: "for",
    TokenIn: "in",
    TokenIf: "if",
    TokenElse: "else",
    TokenYield: "yield",
    TokenLengthOf: "length_of",
    TokenShapeOf: "shape_of",
    TokenRangeOf: "range_of",
    TokenArrow: "->",
    TokenAnd: "&&",
    TokenOr: "||",
    TokenLe: "<=",
    TokenGe: ">=",
    TokenEq: "==",
    TokenNe: "!=",
}

func(t Token) String() string {
    if int(t) < len(tokenStrings) {
        return tokenStrings[t]
    }
    return string([]byte{byte(t)})
}

func(t Token) IsType() bool {
    return (t >= TokenTensor && t <= TokenString)
}

func(t Token) IsKeyword() bool {
    return (t >= TokenFragment && t <= TokenFalse)
}

func(t Token) IsOperator() bool {
    return (t >= TokenLengthOf)
}

//
//    Lexer
//

type Lexer struct {
    input *bufio.Reader
    str strings.Builder
    position *Position
    token Token
    nextCh int
}

func NewLexer(input io.Reader, filename string) *Lexer {
    l := new(Lexer)
    l.Init(input, filename)
    return l
}

func(l *Lexer) Init(input io.Reader, filename string) {
    l.input = bufio.NewReader(input)
    l.position = &Position{1, 1, filename, nil}
    l.token = TokenEof
    l.nextCh = l.readByte()
}

func(l *Lexer) Next() {
    l.position.Column += uint(l.str.Len())
    if l.token == TokenCharacters {
        l.position.Column += 2
    }     
    l.skipSpace()
    l.skipComment()
    l.str.Reset()
    switch {
    case l.eof():
        l.token = TokenEof
    case l.nextCh == '\'' || l.nextCh == '"':
        l.token = l.getCharacters()
    case isAlpha(l.nextCh) || l.nextCh == '_':
        l.token = l.getIdentifier()
    case isDigit(l.nextCh):
        l.token = l.getNumber()
    default:
        l.token = l.getOperator()
    }
}

func(l *Lexer) Token() Token {
    return l.token
}

func(l *Lexer) Str() string {
    return l.str.String()
}

func(l *Lexer) Position() *Position {
    return l.position
}

func(l *Lexer) ReadToken(token Token) {
    if l.token != token {
        RaiseError(l.position, "expected token '%s', found '%s'", token.String(), l.token.String())
    }
    l.Next()
}

func(l *Lexer) ReadIfToken(token Token) bool {
    if l.token == token {
        l.Next()
        return true
    }
    return false
}

func(l *Lexer) getCharacters() Token {
    delim := l.advance()
    for l.nextCh != delim && !l.eof() {
        l.writeByte(l.advance())
    }
    if l.eof() {
        position := Position{
            l.position.Line, 
            l.position.Column+uint(l.str.Len())+1, 
            l.position.Filename, 
            nil,
        }
        RaiseError(&position, "expected %c", delim)
    }
    l.advance()
    return TokenCharacters
}

var keywords = map[string]Token{
    "version": TokenVersion,
    "extension": TokenExtension,
    "graph": TokenGraph,
    "fragment": TokenFragment,
    "tensor": TokenTensor,
    "integer": TokenInteger,
    "scalar": TokenScalar,
    "logical": TokenLogical,
    "string": TokenString,
    "true": TokenTrue,
    "false": TokenFalse,
    "for": TokenFor,
    "in": TokenIn,
    "if": TokenIf,
    "else": TokenElse,
    "yield": TokenYield,
    "length_of": TokenLengthOf,
    "shape_of": TokenShapeOf,
    "range_of": TokenRangeOf,
}

func(l *Lexer) getIdentifier() Token {
    for {
        l.writeByte(l.advance())
        if !(isAlpha(l.nextCh) || isDigit(l.nextCh) || l.nextCh == '_') {
            break
        }
    }
    if kwd, ok := keywords[l.str.String()]; ok {
        return kwd
    }
    return TokenIdentifier
}

func(l *Lexer) getNumber() Token {
    real := false  
    for {
        l.writeByte(l.advance())
        if l.nextCh == '.' && !real {
            l.writeByte(l.advance())
            real = true
        }
        if !isDigit(l.nextCh) {
            break
        }
    }
    if l.nextCh == 'e' || l.nextCh == 'E' {
        l.writeByte(l.advance())
        if l.nextCh == '+' || l.nextCh == '-' {
            l.writeByte(l.advance())
        }
        if !isDigit(l.nextCh) {
            position := Position{
                l.position.Line, 
                l.position.Column+uint(l.str.Len()), 
                l.position.Filename, 
                nil,
            }
            RaiseError(&position, "expected digit")
        }
        for isDigit(l.nextCh) {
            l.writeByte(l.advance())
        }
        real = true
    }
    if real {
        return TokenFractional
    } else {
        return TokenDecimal
    }
}

func(l *Lexer) getOperator() Token {
    ch := l.advance()
    l.writeByte(ch)        
    token := Token(ch)
    if l.nextCh == '=' {
        switch ch {
        case '<':
            l.writeByte(l.advance())
            token = TokenLe
        case '>':
            l.writeByte(l.advance())
            token = TokenGe
        case '=':
            l.writeByte(l.advance())
            token = TokenEq
        case '!':
            l.writeByte(l.advance())
            token = TokenNe
        }
    }
    switch {
    case ch == '&' && l.nextCh == '&':
        l.writeByte(l.advance())
        token = TokenAnd
    case ch == '|' && l.nextCh == '|':
        l.writeByte(l.advance())
        token = TokenOr
    case ch == '-' && l.nextCh == '>':
        l.writeByte(l.advance())
        token = TokenArrow
    }
    return token
}

func(l *Lexer) skipSpace() {
    for isSpace(l.nextCh) {
        l.position.Column++
        ch := l.advance()
        if ch == '\r' || ch == '\n' {
            l.position.Line++
            l.position.Column = 1
        }
        if ch == '\r' && l.nextCh == '\n' {
            l.advance()
        }
    }
}

func(l *Lexer) skipComment() {
    for l.nextCh == '#' {
        for l.nextCh != '\n' && l.nextCh != '\r' && !l.eof() {
            l.advance()
            l.position.Column++
        }
        l.skipSpace()
    }
}

func(l *Lexer) writeByte(ch int) {
    l.str.WriteByte(byte(ch))
}

func(l *Lexer) advance() int {
    ch := l.nextCh
    l.nextCh = l.readByte()
    return ch
}

func(l *Lexer) readByte() int {
    ch, err := l.input.ReadByte()
    if err != nil {
        return -1
    }
    return int(ch)
}

func(l *Lexer) eof() bool {
    return (l.nextCh < 0)
}

func isAlpha(ch int) bool {
    return ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'))
}

func isDigit(ch int) bool {
    return (ch >= '0' && ch <= '9')
}

func isSpace(ch int) bool {
    switch ch {
    case ' ', '\f', '\n', '\r', '\t', '\v':
        return true
    default:
        return false
    }
}

//
//    functions
//

func GetScalarValue(lexer *Lexer) float32 {
    val, err := strconv.ParseFloat(lexer.Str(), 32)
    if err != nil {
        RaiseError(lexer.Position(), "%s", err.Error())
    }
    return float32(val)
}

func GetIntegerValue(lexer *Lexer) int {
    val, err := strconv.Atoi(lexer.Str())
    if err != nil {
        RaiseError(lexer.Position(), "%s", err.Error())
    }
    return val
}

