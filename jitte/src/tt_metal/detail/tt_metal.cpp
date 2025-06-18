
#include <string>
#include <vector>

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {
namespace tt_metal{
namespace detail {

// Profiling is not supported

void ClearProfilerControlBuffer(Device *device) { }

void InitDeviceProfiler(Device *device) { }

void DumpDeviceProfileResults(
        Device *device, 
        std::vector<CoreCoord> &worker_cores, 
        bool last_dump/* = false*/) { }

void DumpDeviceProfileResults(Device *device, bool last_dump/* = false*/) { }

void SetDeviceProfilerDir(std::string output_dir/* = ""*/) { }

void SetHostProfilerDir(std::string output_dir/* = ""*/) { }

void FreshProfilerHostLog() { }

void FreshProfilerDeviceLog() { }

} // namespace detail
} // namespace tt_metal
} // namespace tt

