#pragma once

#include <cstdint>
#include <vector>

namespace tt {
namespace metal {
namespace device {

class Memory {
public:
    Memory() { }
    virtual ~Memory() { }
public:
    virtual uint32_t size() = 0;
    virtual uint8_t *map_addr(uint32_t addr) = 0;
};

class L1Bank: public Memory {
public:
    L1Bank();
    ~L1Bank();
public:
    void init(uint32_t size);
    uint32_t size() override;
    uint8_t *map_addr(uint32_t addr) override;
private:
    std::vector<uint8_t> m_data;
};

class DramBank: public Memory {
public:
    DramBank();
    ~DramBank();
public:
    void init(uint32_t size);
    uint32_t size() override;
    uint8_t *map_addr(uint32_t addr) override;
private:
    std::vector<uint8_t> m_data;
};

} // namespace device
} // namespace metal
} // namespace tt

