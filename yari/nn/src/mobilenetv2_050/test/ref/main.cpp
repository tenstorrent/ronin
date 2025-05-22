// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <memory>
#include <exception>

#include "host/ref/mobilenetv2_050_ref.hpp"

#include "test/util/net.hpp"
#include "test/util/reorder.hpp"
#include "test/util/imagenet.hpp"

using ronin::nn::mobilenetv2_050::ref::MobileNetV2_050_Ref;

using namespace ronin::nn::common::test;

namespace {

class MobileNetV2_050_RefRunner: public util::NetRunner {
public:
    MobileNetV2_050_RefRunner();
    ~MobileNetV2_050_RefRunner();
protected:
    void reorder_inputs() override;
    void infer_batch_size() override;
    void create_net() override;
    void init_net(const std::string &data_dir) override;
    void print_outputs() override;
private:
    std::unique_ptr<MobileNetV2_050_Ref> m_net;
    std::unique_ptr<util::NetWrap<MobileNetV2_050_Ref>> m_iface;
};

MobileNetV2_050_RefRunner::MobileNetV2_050_RefRunner() { }

MobileNetV2_050_RefRunner::~MobileNetV2_050_RefRunner() { }

void MobileNetV2_050_RefRunner::reorder_inputs() {
    m_inputs[0] = util::reorder_nchw_to_nhwc(m_inputs[0], m_batch_size, 224, 224, 3);
}

void MobileNetV2_050_RefRunner::infer_batch_size() {
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

void MobileNetV2_050_RefRunner::create_net() {
    m_net = std::make_unique<MobileNetV2_050_Ref>(m_batch_size);
    m_iface = std::make_unique<util::NetWrap<MobileNetV2_050_Ref>>(m_net.get());
    set_iface(m_iface.get());
}

void MobileNetV2_050_RefRunner::init_net(const std::string &data_dir) {
    m_net->init(data_dir);
}

void MobileNetV2_050_RefRunner::print_outputs() {
    std::vector<float> output;
    m_net->get_output(0, output);
    util::print_imagenet_classes(output, m_batch_size, 5);
}

void run(const util::NetCmdArgs &args) {
    MobileNetV2_050_RefRunner runner;
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

