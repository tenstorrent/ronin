// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/utils.hpp"
//#include <mutex> // [RONIN]
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "llrt/rtoptions.hpp"

namespace tt
{
namespace utils
{
    bool run_command(const string &cmd, const string &log_file, const bool verbose)
    {
        ZoneScoped;
        ZoneText( cmd.c_str(), cmd.length());
        int ret;
#if 0 // [RONIN]
        static std::mutex io_mutex;
#endif
        if (getenv("TT_METAL_BACKEND_DUMP_RUN_CMD") or verbose) {
            {
#if 0 // [RONIN]
                std::lock_guard<std::mutex> lk(io_mutex);
#endif
                cout << "===== RUNNING SYSTEM COMMAND:" << std::endl;
                cout << cmd << std::endl << std::endl;
            }
            ret = system(cmd.c_str());
            // {
            //     std::lock_guard<std::mutex> lk(io_mutex);
            //     cout << "===== DONE SYSTEM COMMAND: " << cmd << std::endl;
            // }

        } else {
            string redirected_cmd = cmd + " >> " + log_file + " 2>&1";
            ret = system(redirected_cmd.c_str());
        }

        return (ret == 0);
    }

    void create_file(string file_path_str) {
        fs::path file_path(file_path_str);
        fs::create_directories(file_path.parent_path());

        std::ofstream ofs(file_path);
        ofs.close();
    }

    const std::string& get_reports_dir() {
        static std::string outpath;
        if (outpath == "") {
            outpath = tt::llrt::OptionsG.get_root_dir() + "/generated/reports/";
        }
        return outpath;
    }

}   // namespace utils

}   // namespace tt
