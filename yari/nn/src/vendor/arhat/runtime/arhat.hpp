//
// Copyright (c) 2019-2025 FRAGATA COMPUTER SYSTEMS AG
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

#pragma once

#include <cstdio>
#include <string>
#include <exception>

namespace arhat {

//
//    Error handling
//

class ArhatError: public std::exception {
public:
    ArhatError(const std::string &msg);
    ~ArhatError();
public:
    const char *what() const noexcept override;
private:
    std::string m_msg;
};

void Error(const char *fmt, ...);

//
//    Tensor
//

class Tensor {
public:
    Tensor();
    ~Tensor();
public:
    enum { MaxRank = 8 };
    enum Dtype {
        Bool,
        Int,
        Float
    };
public:
    void Reset(Dtype aDtype, int aRank, const int *aShape);
    Dtype Type() const {
        return dtype;
    }
    int Volume() const {
        return volume;
    }
    int Rank() const {
        return rank;
    }
    const int *Shape() const {
        return shape;
    }
    void *Data() const {
        return data;
    }
    int Bytes() const {
        return volume * ItemBytes(dtype);
    }
    void Read(const std::string &path);
    void Write(const std::string &path);
    void Read(FILE *fp);
    void Write(FILE *fp);
private:
    void ResizeData(int size);
    static int ItemBytes(Dtype dtype);
    static int ItemBits(Dtype dtype);
private:
    Dtype dtype;
    int volume;
    int rank;
    int shape[MaxRank];
    char *data;
    int dataSize;
};

//
//    Metrics
//

void TopK(int count, const float *data, int k, int *pos, float *val);

} // namespace arhat

