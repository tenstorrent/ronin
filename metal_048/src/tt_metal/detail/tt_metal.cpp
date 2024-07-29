
#include <string>
#include <vector>

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {
namespace tt_metal{
namespace detail {

// [RONIN] Profiling is not supported

void InitDeviceProfiler(Device *device) { }

void DumpDeviceProfileResults(
        Device *device, 
        std::vector<CoreCoord> &worker_cores, 
        bool free_buffers/* = false*/) { }

void DumpDeviceProfileResults(Device *device, bool free_buffers/* = false*/) { }

void SetDeviceProfilerDir(std::string output_dir/* = ""*/) { }

void SetHostProfilerDir(std::string output_dir/* = ""*/) { }

void FreshProfilerHostLog() { }

void FreshProfilerDeviceLog() { }

} // namespace detail
} // namespace tt_metal
} // namespace tt

