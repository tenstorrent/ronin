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
     "io"
    "strconv"
    "strings"
)

//
//    minor types
//

type Version [2]int
type Extensions []string

type Flags int

const (
    KhrEnableFragmentDefinitions Flags = 0x1
    KhrEnableOperatorExpressions = 0x2
)

//
//    constants
//

var MaxSupportedVersion = Version{1, 0}

//
//    ParserCallback
//

type ParserCallback interface {
    BeginDocument(filename string, version Version)
    EndDocument(filename string)
    HandleExtension(extension string) bool
    BeginGraph(proto *Prototype, fragments map[string]*Prototype)
    EndGraph(proto *Prototype, dtypes map[string]Typename)
    Operation(proto *Prototype, args map[string]Value, dtypes map[string]Typename)
}

//
//    ParserCallbackBase
//

type ParserCallbackBase struct {}

func(c *ParserCallbackBase) BeginDocument(filename string, version Version) { }

func(c *ParserCallbackBase) EndDocument(filename string) { }

func(c *ParserCallbackBase) HandleExtension(extension string) bool {
    return false
}

func(c *ParserCallbackBase) BeginGraph(proto *Prototype, fragments map[string]*Prototype) { }

func(c *ParserCallbackBase) EndGraph(proto *Prototype, dtypes map[string]Typename) { }

//
//    Parser
//

type Parser interface {
    Parse(is io.Reader, filename string, callback ParserCallback)
}

//
//    functions
//

func GetTypename(lexer *Lexer) Typename {
    switch lexer.Token() {
    case TokenInteger:
        return TypenameInteger
    case TokenScalar:
        return TypenameScalar
    case TokenLogical:
        return TypenameLogical
    case TokenString:
        return TypenameString
    case '?':
        return TypenameGeneric
    default:
        RaiseError(lexer.Position(), 
            "expected type name, found '%s'", lexer.Token().String())
        return 0
    }
}

func ReadVersion(lexer *Lexer) Version {
    lexer.ReadToken(TokenVersion)
    if lexer.Token() != TokenFractional {
        RaiseError(lexer.Position(), "expected version number")
    }
    str := lexer.Str()
    dots := strings.Count(str, ".")
    isDigits := true
    for _, c := range str {
        if !((c >= '0' && c <= '9') || c == '.') {
            isDigits = false
            break
        }
    }
    if !isDigits || dots != 1 {
        RaiseError(lexer.Position(), "invalid version number format: %s", str)
    }
    lexer.Next()
    parts := strings.SplitN(str, ".", 2)
    major, err := strconv.Atoi(parts[0])
    if err != nil {
        RaiseError(lexer.Position(), "%s", err.Error())
    }
    minor, err := strconv.Atoi(parts[1])
    if err != nil {
        RaiseError(lexer.Position(), "%s", err.Error())
    }
    if major > MaxSupportedVersion[0] || 
            (major == MaxSupportedVersion[0] && minor > MaxSupportedVersion[1]) {
        RaiseError(lexer.Position(), 
            "unsupported version %d.%d; maximum supported version is %d.%d",
                major, minor, MaxSupportedVersion[0], MaxSupportedVersion[1])
    }
    lexer.ReadToken(Token(';'))
    return Version{major, minor}
}

func ReadExtensions(lexer *Lexer, handler func(string) bool) Extensions {
    var extensions Extensions
    var position Position
    for lexer.ReadIfToken(TokenExtension) {
        for {
            position = *lexer.Position()
            str := lexer.Str()
            extensions = append(extensions, str)
            lexer.ReadToken(TokenIdentifier)
            if !handler(str) {
                RaiseError(&position, "could not handle extension '%s'", str)
            }
            if !lexer.ReadIfToken(Token(',')) {
                break
            }
        }
        lexer.ReadToken(Token(';'))
    }
    return extensions
}

