// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

#include "test/util/net.hpp"
#include "test/util/tensor.hpp"
#include "test/util/timer.hpp"
#include "test/util/comp.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

namespace {

bool str_to_int(const char *s, int &v) {
    char *p;
    long t = strtol(s, &p, 10);
    if (*p != '\0') {
        return false;
    }
    int r = int(t);
    if (long(r) != t) {
        return false;
    }
    v = r;
    return true;
}

} // namespace

bool parse_net_cmd_args(int argc, char **argv, NetCmdArgs &args) {
    args.batch = 0;
    args.compare = false;
    args.repeat = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--mode")) {
            i++;
            if (i == argc) {
                fprintf(stderr, "Mode must be provided after --mode\n");
                return false;
            }
            args.mode = argv[i];
        } else if (!strcmp(arg, "--batch")) {
            i++;
            if (i == argc) {
                fprintf(stderr, "Batch size must be provided after --batch\n");
                return false;
            }
            if (!str_to_int(argv[i], args.batch)) {
                fprintf(stderr, "Invalid bath size\n");
                return false;
            }
        } else if (!strcmp(arg, "--data")) {
            i++;
            if (i == argc) {
                fprintf(stderr, "Data directory path must be provided after --data\n");
                return false;
            }
            args.data = argv[i];
        } else if (!strcmp(arg, "--input")) {
            if (i + 1 == argc) {
                fprintf(stderr, "Input file name(s) must be provided after --input\n");
                return false;
            }
            while (i + 1 < argc && argv[i+1][0] != '-') {
                i++;
                args.inputs.push_back(argv[i]);
            }
        } else if (!strcmp(arg, "--output")) {
            if (i + 1 == argc) {
                fprintf(stderr, "Output file name(s) must be provided after --output\n");
                return false;
            }
            while (i + 1 < argc && argv[i+1][0] != '-') {
                i++;
                args.outputs.push_back(argv[i]);
            }
        } else if (!strcmp(arg, "--compare")) {
            args.compare = true;
        } else if (!strcmp(arg, "--repeat")) {
            i++;
            if (i == argc) {
                fprintf(stderr, "Repeat count must be provided after --repeat\n");
                return false;
            }
            if (!str_to_int(argv[i], args.repeat)) {
                fprintf(stderr, "Invalid repeat count\n");
                return false;
            }
        } else {
            fprintf(stderr, "Unrecognized option: %s\n", argv[i]);
            return false;
        }
    }
    if (args.data.empty()) {
        fprintf(stderr, "Missing data directory\n");
        return false;
    }
    if (args.inputs.empty()) {
        fprintf(stderr, "Missing input file name(s)\n");
        return false;
    }
    return true;
}

//
//    NetError
//

NetError::NetError(const std::string &msg):
        m_msg(msg) { }

NetError::~NetError() { }

const char *NetError::what() const noexcept {
    return m_msg.c_str();
}

//
//    NetRunner
//

NetRunner::NetRunner() { }

NetRunner::~NetRunner() { }

void NetRunner::set_iface(NetIface *net) {
    m_net = net;
}

void NetRunner::run(const NetCmdArgs &args) {
    reset(args);
    read_inputs();
    infer_batch_size();
    reorder_inputs();
    create_net();
    validate_input_count();
    validate_output_count();
    init_net(args.data);
    set_inputs();
    if (args.repeat == 0) {
        m_net->run();
    } else {
        m_net->run();
        sync_run();
        Timer timer;
        timer.start();
        for (int iter = 0; iter < m_args->repeat; iter++) {
            m_net->run();
            sync_run();
        }
        timer.stop();
        m_elapsed_time = timer.elapsed();
        print_elapsed_time(m_args->repeat);
    }
    if (args.compare) {
        compare_outputs();
    } else if (!args.outputs.empty()) {
        write_outputs();
    }
    print_outputs();
}

void NetRunner::reorder_inputs() {
    // default: do nothing
}

void NetRunner::infer_batch_size() {
    m_batch_size = 1;
}

void NetRunner::sync_run() {
    // default: do nothing
}

void NetRunner::print_elapsed_time(int repeat) {
    printf("Elapsed time %g ms / %d iterations = %g\n", 
        m_elapsed_time, repeat, m_elapsed_time / float(repeat));
}

void NetRunner::print_outputs() {
    // default: do nothing
}

void NetRunner::reset(const NetCmdArgs &args) {
    m_args = &args;
    m_net = nullptr;
    m_inputs.clear();
    m_batch_size = 0;
    m_elapsed_time = 0.0f;
}

void NetRunner::read_inputs() {
    int input_count = int(m_args->inputs.size());
    m_inputs.resize(input_count);
    for (int i = 0; i < input_count; i++) {
        read_tensor(m_args->inputs[i], m_inputs[i]);
    }
}

void NetRunner::validate_input_count() {
    int input_count = int(m_args->inputs.size());
    int net_input_count = m_net->input_count();
    if (input_count != net_input_count) {
        error(
            "Invalid number of inputs: want %d, got %d", 
                net_input_count, input_count);
    }
}

void NetRunner::validate_output_count() {
    int output_count = int(m_args->outputs.size());
    if (output_count != 0) {
        int net_output_count = m_net->output_count();
        if (output_count != net_output_count) {
            error(
                "Invalid number of outputs: want %d, got %d", 
                    net_output_count, output_count);
        }
    }
}

void NetRunner::set_inputs() {
    int input_count = int(m_inputs.size());
    for (int i = 0; i < input_count; i++) {
        m_net->set_input(i, m_inputs[i]);
    }
}

void NetRunner::compare_outputs() {
    std::vector<float> output;
    std::vector<float> golden;
    int output_count = int(m_args->outputs.size());
    for (int i = 0; i < output_count; i++) {
        m_net->get_output(i, output);
        read_tensor(m_args->outputs[i], golden);
        match_output(i, output, golden);
    }
}

void NetRunner::write_outputs() {
    std::vector<float> output;
    int output_count = int(m_args->outputs.size());
    for (int i = 0; i < output_count; i++) {
        m_net->get_output(i, output);
        write_tensor(m_args->outputs[i], output);
    }
}

void NetRunner::match_output(
        int index,
        const std::vector<float> &output, 
        const std::vector<float> &golden) {
    float rtol = 1.0e-1f;
    float atol = 1.0e-3f;
    float rtol_delta = 0.0f;
    float atol_delta = 0.0f;
    int num_outliers = 0;

    printf("Match output [%d]\n", index);

    bool allclose = 
        comp_allclose(
            golden, 
            output, 
            rtol, 
            atol, 
            rtol_delta, 
            atol_delta, 
            num_outliers);
    printf("All close = %s\n", allclose ? "OK" : "FAIL");
    printf("Max ATOL delta: %g, max RTOL delta: %g, outliers: %d / %zd\n", 
        atol_delta, rtol_delta, num_outliers, output.size());

    float pcc = comp_pcc(golden, output);
    printf("Pcc = %s\n", (pcc >= 0.9999f) ? "OK" : "FAIL"); 
    printf("PCC: %g\n", pcc);
}

void NetRunner::error(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    throw NetError(buf);
}

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

