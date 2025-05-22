// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <cmath>
#include <memory>
#include <algorithm>

#include "host/core/api.hpp"

#include "host/tanto/mobilenetv2_050_global.hpp"

#include "test/util/net.hpp"
#include "test/util/reorder.hpp"
#include "test/util/imagenet.hpp"

#include "test/tanto/run.hpp"

namespace core = ronin::tanto::host;

using namespace ronin::nn::common::test;

namespace {

using ronin::nn::mobilenetv2_050::tanto::MobileNetV2_050_Global;

void copy(float *dst, const float *src, int count) {
    memcpy(dst, src, count * sizeof(float));
}

int round_up(int a, int b) {
    return ((a + b - 1) / b) * b;
}

int pow2_up(int n) {
    return 1 << int(std::ceil(std::log2(double(n))));
}

class MobileNetV2_050_GlobalRunner: public util::NetRunner {
public:
    MobileNetV2_050_GlobalRunner(const core::Device &device, int algo_batch_size);
    ~MobileNetV2_050_GlobalRunner();
protected:
    void reorder_inputs() override;
    void infer_batch_size() override;
    void create_net() override;
    void init_net(const std::string &data_dir) override;
    void sync_run() override;
    void print_outputs() override;
private:
    void pad_inputs_to_batch_size();
private:
    static constexpr int FRAME_SIZE = 224 * 224 * 3;
private:
    core::Device m_device;
    int m_algo_batch_size;
    int m_input_batch_size;
    std::unique_ptr<MobileNetV2_050_Global> m_net;
    std::unique_ptr<util::NetWrap<MobileNetV2_050_Global>> m_iface;
};

MobileNetV2_050_GlobalRunner::MobileNetV2_050_GlobalRunner(
        const core::Device &device, int algo_batch_size):
            m_device(device),
            m_algo_batch_size(algo_batch_size),
            m_input_batch_size(0) { }

MobileNetV2_050_GlobalRunner::~MobileNetV2_050_GlobalRunner() { }

void MobileNetV2_050_GlobalRunner::reorder_inputs() {
    m_inputs[0] = util::reorder_nchw_to_nhwc(m_inputs[0], m_batch_size, 224, 224, 3);
}

void MobileNetV2_050_GlobalRunner::infer_batch_size() {
    if (m_inputs.size() == 0) {
        error("Missing input tensors");
    }
    int volume = int(m_inputs[0].size());
    if (volume % FRAME_SIZE != 0) {
        error(
            "Input #0 volume (%d) is not multiple of frame size (%d)\n",
                volume, FRAME_SIZE);
    }
    // input batch size is used to print results
    m_input_batch_size = volume / FRAME_SIZE;
    m_batch_size = std::max(m_input_batch_size, m_args->batch);
    m_batch_size = round_up(m_batch_size, m_algo_batch_size);
    printf("Batch size [%d / %d]\n", m_batch_size, m_algo_batch_size);
    pad_inputs_to_batch_size();
}

void MobileNetV2_050_GlobalRunner::create_net() {
    m_net = std::make_unique<MobileNetV2_050_Global>(m_device, m_batch_size, m_algo_batch_size);
    m_iface = std::make_unique<util::NetWrap<MobileNetV2_050_Global>>(m_net.get());
    set_iface(m_iface.get());
}

void MobileNetV2_050_GlobalRunner::init_net(const std::string &data_dir) {
    m_net->init(data_dir);
}

void MobileNetV2_050_GlobalRunner::sync_run() {
    core::Queue queue(m_device, 0);
    queue.finish();
}

void MobileNetV2_050_GlobalRunner::print_outputs() {
    std::vector<float> output;
    m_net->get_output(0, output);
    util::print_imagenet_classes(output, m_input_batch_size, 5);
}

void MobileNetV2_050_GlobalRunner::pad_inputs_to_batch_size() {
    int pad = m_batch_size - m_input_batch_size;
    if (pad == 0) {
        return;
    }
    // just replicate last frame
    std::vector<float> &input = m_inputs[0];
    input.resize(m_batch_size * FRAME_SIZE);
    float *dst = input.data() + m_input_batch_size * FRAME_SIZE;
    const float *src = dst - FRAME_SIZE;
    for (int i = 0; i < pad; i++) {
        copy(dst, src, FRAME_SIZE);
        dst += FRAME_SIZE;
    }
}

int get_algo_batch_size(const util::NetCmdArgs &args) {
    if (args.batch > 8) {
        int result = std::min(args.batch, 64);
        result = std::max(result, 16);
        result = pow2_up(result);
        return result;
    } else if (args.batch > 0) {
        return args.batch;
    } else {
        // default 16 is compatible with all supported algorithms
        return 16;
    }
}

} // namespace

void run_global(const util::NetCmdArgs &args) {
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    int algo_batch_size = get_algo_batch_size(args);
    MobileNetV2_050_GlobalRunner runner(device, algo_batch_size);
    runner.run(args);
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
}

