// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <memory>
#include <exception>

#include "host/ref/resnet50_v1_7_ref.hpp"

#include "test/util/net.hpp"
#include "test/util/reorder.hpp"
#include "test/util/imagenet.hpp"

using ronin::nn::resnet50_v1_7::ref::ResNet50V17Ref;

using namespace ronin::nn::common::test;

namespace {

class ResNet50V17RefRunner: public util::NetRunner {
public:
    ResNet50V17RefRunner();
    ~ResNet50V17RefRunner();
protected:
    void reorder_inputs() override;
    void infer_batch_size() override;
    void create_net() override;
    void init_net(const std::string &data_dir) override;
    void print_outputs() override;
private:
    std::unique_ptr<ResNet50V17Ref> m_net;
    std::unique_ptr<util::NetWrap<ResNet50V17Ref>> m_iface;
};

ResNet50V17RefRunner::ResNet50V17RefRunner() { }

ResNet50V17RefRunner::~ResNet50V17RefRunner() { }

void ResNet50V17RefRunner::reorder_inputs() {
    m_inputs[0] = util::reorder_nchw_to_nhwc(m_inputs[0], m_batch_size, 224, 224, 3);
}

void ResNet50V17RefRunner::infer_batch_size() {
    constexpr int FRAME_SIZE = 224 * 224 * 3;
    if (m_inputs.size() == 0) {
        error("Missing input tensors");
    }
    int volume = int(m_inputs[0].size());
    if (volume % FRAME_SIZE != 0) {
        error(
            "Input #0 volume (%d) is not multiple of frame size (%d)\n",
                volume, FRAME_SIZE);
    }
    m_batch_size = volume / FRAME_SIZE;
}

void ResNet50V17RefRunner::create_net() {
    m_net = std::make_unique<ResNet50V17Ref>(m_batch_size);
    m_iface = std::make_unique<util::NetWrap<ResNet50V17Ref>>(m_net.get());
    set_iface(m_iface.get());
}

void ResNet50V17RefRunner::init_net(const std::string &data_dir) {
    m_net->init(data_dir);
}

void ResNet50V17RefRunner::print_outputs() {
    std::vector<float> output;
    m_net->get_output(0, output);
    util::print_imagenet_classes(output, m_batch_size, 5);
}

void run(const util::NetCmdArgs &args) {
    ResNet50V17RefRunner runner;
    runner.run(args);
}

} // namespace

//
//    Main program
//

int main(int argc, char **argv) {
    util::NetCmdArgs args;
    if (!util::parse_net_cmd_args(argc, argv, args)) {
        return 1;
    }
    try {
        run(args);
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

