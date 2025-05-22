//
// Copyright (c) 2019-2025 FRAGATA COMPUTER SYSTEMS AG
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

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>

#include "runtime/arhat.hpp"

namespace arhat {

namespace {

//
//    TensorHeader
//

struct TensorHeader {
    enum { MaxRank = 8 };
    enum QuantCode { 
        Float = 0x00, 
        Integer = 0x01, 
        Linear = 0x10, 
        Logarithmic = 0x11 
    };
    uint8_t magic[2];
    uint8_t version[2];
    uint32_t dataLength;
    uint32_t rank;
    uint32_t extents[MaxRank];
    uint32_t bitsPerItem;
    uint32_t quantCode;
    uint32_t quantParams[8];
    uint8_t reserved[44];
};

// interface

void FillTensorHeader(
        TensorHeader &header,
        int *version, 
        int rank, 
        int *extents, 
        int bitsPerItem,
        TensorHeader::QuantCode quantCode) {
    const char *magic = "N\xEF";
    memset(&header, 0, sizeof(header));
    header.magic[0] = uint8_t(magic[0]);
    header.magic[1] = uint8_t(magic[1]);
    header.version[0] = uint8_t(version[0]);
    header.version[1] = uint8_t(version[1]);
    if (rank > TensorHeader::MaxRank) {
        Error("Tensor rank %d exceeds maximum possible value (%d)", rank, TensorHeader::MaxRank);
    }
    uint32_t itemCount = 1;
    for (int i = 0; i < rank; i++) {
        itemCount *= uint32_t(extents[i]);
    }    
    header.dataLength = uint32_t((itemCount * bitsPerItem + 7) / 8);
    header.bitsPerItem = uint32_t(bitsPerItem);
    header.rank = uint32_t(rank);
    header.quantCode = quantCode;
    for (int i = 0; i < rank; i++) {
        header.extents[i] = extents[i];
    }
}

void ValidateTensorHeader(const TensorHeader &header) {
    if (header.magic[0] != 'N' || header.magic[1] != 0xEF) {
        Error("Invliad magic number in tensor binary");
    }
    if (header.version[0] != 1 || header.version[1] != 0) {
        Error("Unknown version number %d.%d", int(header.version[0]), int(header.version[1]));
    }
    if (header.rank > TensorHeader::MaxRank) {
        Error("Tensor rank %d exceeds maximum allowed rank (%d)", 
            int(header.rank), int(TensorHeader::MaxRank));
    }
    uint32_t itemCount = 1;
    for (int i = 0; i < int(header.rank); i++) {
        itemCount *= header.extents[i];
    }
    if (header.dataLength != (itemCount * header.bitsPerItem + 7) / 8) {
        Error("Data length is not compatible with extents and bits per item");
    }
    if ((header.quantCode & 0xffff0000) == 0) {
        // Khronos-defined item type
        uint32_t code = (header.quantCode & 0x0000ffff);
        switch (code) {
        case TensorHeader::Float:
            if (header.bitsPerItem != 16 && header.bitsPerItem != 32 && header.bitsPerItem != 64) {
                Error("Invalid bits per item for float item type: %d", int(header.bitsPerItem));
            }
            break;
        case TensorHeader::Integer:
        case TensorHeader::Linear:
        case TensorHeader::Logarithmic:
            if (header.bitsPerItem > 64) {
                Error("Invalid bits per item for integer item type: %d", int(header.bitsPerItem));
            }
            break;
        default:
            Error("Unkown Khronos-defined item type code: %x", int(code));
            break;
        }
    }
}

void ReadTensorHeader(TensorHeader &header, FILE *fp) {
    // ACHTUNG: Little endian host assumed
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        Error("Failed to read tensor header");
    } 
}

void WriteTensorHeader(TensorHeader &header, FILE *fp) {
    // ACHTUNG: Little endian host assumed
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        Error("Failed to write tensor header");
    } 
}

//
//    DataBuffer
//

class DataBuffer {
public:
    DataBuffer(int size);
    ~DataBuffer();
public:
    int size;
    char *data;
};

// construction/destruction

DataBuffer::DataBuffer(int size) {
    this->size = size;
    this->data = new char[size];
}

DataBuffer::~DataBuffer() {
    delete[] data;
}

//
//    Data utiities
//

void PackBits(int n, const bool *data, char *bytes) {
    for (int i = 0; i < n; i++) {
        bytes[i / 8] |= (data[i] << (7 - (i % 8)));
    }
}

void UnpackBits(int n, const char *bytes, bool *data ) {
    for (int i = 0; i < n; i++) {
        data[i] = (bytes[i / 8] >> (7 - (i % 8))) & 0x01;
    }
}

// ACHTUNG: Little endian host assumed

void FloatsFromBytes(const char *bytes, int count, int bitsPerItem, float *data) {
    if (bitsPerItem == 32) {
        const float *p = (const float *)bytes;
        for (int i = 0; i < count; i++) {
            data[i] = p[i];
        }
    } else if (bitsPerItem == 64) {
        const double *p = (const double *)bytes;
        for (int i = 0; i < count; i++) {
            data[i] = float(p[i]);
        }
    } else {
        Error("Cannot load float data of %d bits per item", bitsPerItem);
    }
}

void IntsFromBytes(const char *bytes, int count, int bitsPerItem, int32_t *data) {
    if (bitsPerItem == 8) {
        const int8_t *p = (const int8_t *)bytes;
        for (int i = 0; i < count; i++) {
            data[i] = int32_t(p[i]);
        }
    } else if (bitsPerItem == 16) {
        const int16_t *p = (const int16_t *)bytes;
        for (int i = 0; i < count; i++) {
            data[i] = int32_t(p[i]);
        }
    } else if (bitsPerItem == 32) {
        const int32_t *p = (const int32_t *)bytes;
        for (int i = 0; i < count; i++) {
            data[i] = p[i];
        }
    } else if (bitsPerItem == 64) {
        const int64_t *p = (const int64_t *)bytes;
        for (int i = 0; i < count; i++) {
            data[i] = int32_t(p[i]);
        }
    } else {
        Error("Cannot load int data of %d bits per item", bitsPerItem);
    }
}

void BoolsFromBytes(const char *bytes, int count, int bitsPerItem, bool *data) {
    if (bitsPerItem == 1) {
        UnpackBits(count, bytes, data);
    } else if (bitsPerItem == 8) {
        int8_t *p = (int8_t *)bytes;
        for (int i = 0; i < count; i++) {
            data[i] = (p[i] != 0);
        }
    } else {
        Error("Cannot load bool data of %d bits per item", bitsPerItem);
    }
}

void FloatsToBytes(const float *data, int count, char *bytes) {
    float *p = (float *)bytes;
    for (int i = 0; i < count; i++) {
        p[i] = data[i];
    }
}

void IntsToBytes(const int32_t *data, int count, char *bytes) {
    int32_t *p = (int32_t *)bytes;
    for (int i = 0; i < count; i++) {
        p[i] = data[i];
    }
}

void BoolsToBytes(const bool* data, int count, char *bytes ) {
    PackBits(count, data, bytes);
}

} // namespace

//
//    Tensor
//

// construction/destruction

Tensor::Tensor() {
    data = nullptr;
    dataSize = 0;
}

Tensor::~Tensor() {
    delete[] data;
}

// interface

void Tensor::Reset(Dtype aDtype, int aRank, const int *aShape) {
    assert(aRank <= MaxRank);
    dtype = aDtype;
    rank = aRank;
    memset(shape, 0, sizeof(shape));
    volume = 1;
    for (int i = 0; i < rank; i++) {
        int d = aShape[i];
        shape[i] = d;
        volume *= d;
    }
    ResizeData(volume * ItemBytes(dtype));
}

void Tensor::Read(const std::string &path) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        Error("Failed to open file %s", path.c_str());
    }
    Read(fp);
    fclose(fp);
}

void Tensor::Write(const std::string &path) {
    FILE *fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        Error("Failed to open file %s", path.c_str());
    }
    Write(fp);
    fclose(fp);
}

// TODO: Revise I/O functions for 1.0.3 tensor data format
//     (but how to keep compatibility to legacy model zoo files?)

void Tensor::Read(FILE *fp) {
    TensorHeader header;
    ReadTensorHeader(header, fp);
    ValidateTensorHeader(header);
    rank = int(header.rank);
    assert(rank <= MaxRank);
    memset(shape, 0, sizeof(shape));
    volume = 1;
    for (int i = 0; i < rank; i++) {
        int d = int(header.extents[i]);
        shape[i] = d;
        volume *= d;
    }
    DataBuffer bytes(header.dataLength);
    if (fread(bytes.data, 1, bytes.size, fp) != bytes.size) {
        Error("Failed to read tensor data");
    }
    int bitsPerItem = int(header.bitsPerItem);
    if (header.quantCode == TensorHeader::Float) {
        dtype = Float;
        ResizeData(volume * ItemBytes(dtype));
        FloatsFromBytes(bytes.data, volume, bitsPerItem, (float *)data);
    } else if (header.quantCode == TensorHeader::Integer && bitsPerItem == 1) {
        dtype = Bool;
        ResizeData(volume * ItemBytes(dtype));
        BoolsFromBytes(bytes.data, volume, bitsPerItem, (bool *)data);
    } else if (header.quantCode == TensorHeader::Integer) {
        dtype = Int;
        ResizeData(volume * ItemBytes(dtype));
        IntsFromBytes(bytes.data, volume, bitsPerItem, (int32_t *)data);
    } else {
        Error("Unsupported tensor item type code %x and bits per item %d", 
            int(header.quantCode), bitsPerItem);
    }
}

void Tensor::Write(FILE *fp) {
    if (rank > TensorHeader::MaxRank) {
        Error("Tensor rank %d exceeds maximum allowed rank (%d)", rank, TensorHeader::MaxRank);
    }
    TensorHeader::QuantCode quantCode = 
        (dtype == Float) ? TensorHeader::Float : TensorHeader::Integer;
    TensorHeader header;
    int version[] = { 1, 0 };
    FillTensorHeader(header, version, rank, shape, ItemBits(dtype), quantCode);
    DataBuffer bytes(header.dataLength);
    switch (dtype) {
    case Float:
        FloatsToBytes((const float *)data, volume, bytes.data);
        break;
    case Int:
        IntsToBytes((const int32_t *)data, volume, bytes.data);
        header.quantParams[0] = 1;
        break;
    case Bool:
        BoolsToBytes((const bool *)data, volume, bytes.data);
        break;
    default:
        Error("Invalid tensor data type: %d", dtype);
        break;
    }
    WriteTensorHeader(header, fp);
    if (fwrite(bytes.data, 1, bytes.size, fp) != bytes.size) {
        Error("Failed to write tensor data");
    }
}

// implementation

void Tensor::ResizeData(int size) {
    if (size == dataSize) {
        return;
    }
    delete[] data;
    data = new char[size];
    dataSize = size;
}

int Tensor::ItemBytes(Dtype dtype) {
    switch (dtype) {
    case Float:
        return sizeof(float);
    case Int:
        return sizeof(int32_t);
    case Bool:
        return sizeof(bool);
    default:
        return 0;
    }
}

int Tensor::ItemBits(Dtype dtype) {
    switch (dtype) {
    case Float:
        return 32;
    case Int:
        return 32;
    case Bool:
        return 1;
    default:
        return 0;
    }
}

} // namespace arhat

