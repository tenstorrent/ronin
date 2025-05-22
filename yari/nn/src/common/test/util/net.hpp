// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <exception>

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

struct NetCmdArgs {
    std::string mode;
    int batch;
    std::string data;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    bool compare;
    int repeat;
};

bool parse_net_cmd_args(int argc, char **argv, NetCmdArgs &args);

class NetIface {
public:
    NetIface() { }
    virtual ~NetIface() { }
public:
    virtual int input_count() = 0;
    virtual void set_input(int index, const std::vector<float> &data) = 0;
    virtual int output_count() = 0;
    virtual void get_output(int index, std::vector<float> &data) = 0;
    virtual void run() = 0;
};

template<typename NET>
class NetWrap: public NetIface {
public:
    NetWrap(NET *net):
        m_net(net) { }
    ~NetWrap() { }
public:
    int input_count() override {
        return m_net->input_count();
    }
    void set_input(int index, const std::vector<float> &data) override {
        return m_net->set_input(index, data);
    }
    int output_count() override {
        return m_net->output_count();
    }
    void get_output(int index, std::vector<float> &data) override {
        m_net->get_output(index, data);
    }
    void run() override {
        m_net->run();
    }
private:
    NET *m_net = nullptr;
};

class NetError: public std::exception {
public:
    NetError(const std::string &msg);
    ~NetError();
public:
    const char *what() const noexcept override;
private:
    std::string m_msg;
};

class NetRunner {
public:
    NetRunner();
    virtual ~NetRunner();
public:
    void set_iface(NetIface *net);
    void run(const NetCmdArgs &args);
protected:
    virtual void reorder_inputs();
    virtual void infer_batch_size();
    virtual void create_net() = 0;
    virtual void init_net(const std::string &data_dir) = 0;
    virtual void sync_run();
    virtual void print_elapsed_time(int repeat);
    virtual void print_outputs();
protected:
    void reset(const NetCmdArgs &args);
    void read_inputs();
    void validate_input_count();
    void validate_output_count();
    void set_inputs();
    void compare_outputs();
    void write_outputs();
    static void match_output(
        int index,
        const std::vector<float> &output, 
        const std::vector<float> &golden);
    static void error(const char *fmt, ...);
protected:
    const NetCmdArgs *m_args = nullptr;
    NetIface *m_net = nullptr;
    std::vector<std::vector<float>> m_inputs;
    int m_batch_size = 0;
    float m_elapsed_time = 0.0f;
};

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

