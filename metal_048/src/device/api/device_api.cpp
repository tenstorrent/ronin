
#include <cstdint>

#include "api/device_api.hpp"
#include "api/device_impl.hpp"

namespace tt {
namespace metal {
namespace device {

///
//    Device
//

Device *Device::create(Arch arch) {
    return new DeviceImpl(arch);
}

} // namespace device
} // namespace metal
} // namespace tt

