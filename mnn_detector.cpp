// MNN backend. Regions (SAHI slices, the optional full image) are letterboxed
// on the CPU (OpenMP, one region per thread), copied to the backend in batches,
// run, and decoded on the CPU; only boxes above the threshold are kept.

#include "mnn_detector.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

const char* forward_name(int t) {
    switch (t) {
        case MNN_FORWARD_CPU:
        case MNN_FORWARD_CPU_EXTENSION: return "CPU";
        case MNN_FORWARD_CUDA: return "CUDA";
        case MNN_FORWARD_VULKAN: return "Vulkan";
        default: return "other";
    }
}

}  // namespace

std::vector<std::string> mnn_available_backends() {
    std::vector<std::string> b;
#if RTMDET_WITH_CUDA
    b.push_back("cuda");
#endif
#if RTMDET_WITH_VULKAN
    b.push_back("vulkan");
#endif
    b.push_back("cpu");
    return b;
}

struct MnnDetector::Impl {
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    MNN::Tensor* output = nullptr;
    std::unique_ptr<MNN::Tensor> host_in, host_out;
    MnnConfig cfg;
    MnnInfo info;
    bool feature_rows = false;

    ~Impl() {
        if (net && session) net->releaseSession(session);
    }

    // Shape the session for `batch` regions; returns false if the model can't take it.
    bool resize(int batch, std::string& error) {
        net->resizeTensor(input, {batch, 3, cfg.size, cfg.size});
        net->resizeSession(session);
        output = cfg.out_name.empty() ? net->getSessionOutput(session, nullptr)
                                      : net->getSessionOutput(session, cfg.out_name.c_str());
        if (!output) {
            error = "output '" + cfg.out_name + "' not found";
            return false;
        }
        const std::vector<int> s = output->shape();
        if (s.size() != 3 || s[0] != batch) return false;
        host_in.reset(new MNN::Tensor(input, MNN::Tensor::CAFFE));
        host_out.reset(new MNN::Tensor(output, MNN::Tensor::CAFFE));
        return true;
    }

    bool run(std::string& error) {
        if (!input->copyFromHostTensor(host_in.get())) {
            error = "copy to the backend failed";
            return false;
        }
        const MNN::ErrorCode rc = net->runSession(session);
        if (rc != MNN::NO_ERROR) {
            error = "runSession failed (MNN error " + std::to_string(static_cast<int>(rc)) + ")";
            return false;
        }
        if (!output->copyToHostTensor(host_out.get())) {
            error = "copy from the backend failed";
            return false;
        }
        return true;
    }
};

MnnDetector::~MnnDetector() = default;

std::unique_ptr<MnnDetector> MnnDetector::create(const MnnConfig& cfg, MnnInfo& info, std::string& error) {
    std::unique_ptr<MnnDetector> d(new MnnDetector);
    d->impl_.reset(new Impl);
    Impl& m = *d->impl_;
    m.cfg = cfg;

    m.net.reset(MNN::Interpreter::createFromFile(cfg.model_path.c_str()), MNN::Interpreter::destroy);
    if (!m.net) {
        error = "MNN cannot load " + cfg.model_path;
        return nullptr;
    }
    m.net->setSessionMode(MNN::Interpreter::Session_Release);

    // Backend: the requested one, or the first compiled GPU backend for "auto"
    std::vector<std::string> order;
    if (cfg.backend == "auto") order = mnn_available_backends();
    else order = {cfg.backend};
    for (const std::string& b : order) {
        MNN::ScheduleConfig sc;
        sc.type = b == "cuda" ? MNN_FORWARD_CUDA : b == "vulkan" ? MNN_FORWARD_VULKAN : MNN_FORWARD_CPU;
        sc.backupType = MNN_FORWARD_CPU;  // ops a GPU backend lacks run on the CPU
        if (sc.type == MNN_FORWARD_CPU) sc.numThread = std::max(1, cfg.threads);
        MNN::BackendConfig bc;
        bc.precision = cfg.precision == "high" ? MNN::BackendConfig::Precision_High
                     : cfg.precision == "low" ? MNN::BackendConfig::Precision_Low
                                              : MNN::BackendConfig::Precision_Normal;
        sc.backendConfig = &bc;
        MNN::Session* s = m.net->createSession(sc);
        if (!s) continue;
        int used[4] = {MNN_FORWARD_CPU, MNN_FORWARD_CPU, MNN_FORWARD_CPU, MNN_FORWARD_CPU};
        m.net->getSessionInfo(s, MNN::Interpreter::BACKENDS, used);
        if (used[0] == MNN_FORWARD_CPU_EXTENSION) used[0] = MNN_FORWARD_CPU;  // x86/ARM CPU backend
        if (used[0] != sc.type) {
            // MNN fell back to the CPU: the backend isn't available on this machine
            if (cfg.backend != "auto") {
                std::cerr << "Warning: " << b << " backend not available here; MNN uses "
                          << forward_name(used[0]) << "\n";
            } else if (sc.type != MNN_FORWARD_CPU) {
                m.net->releaseSession(s);
                continue;
            }
        }
        m.session = s;
        info.backend = forward_name(used[0]);
        if (used[0] == MNN_FORWARD_CPU) info.backend += " (" + std::to_string(std::max(1, cfg.threads)) + " threads)";
        break;
    }
    if (!m.session) {
        error = "MNN could not create a session";
        return nullptr;
    }

    // One input, one decoded output (B, anchors, 4 + classes) expected
    const auto& outs = m.net->getSessionOutputAll(m.session);
    if (cfg.out_name.empty() && outs.size() != 1) {
        error = "the model has " + std::to_string(outs.size()) + " outputs; expected one decoded output "
                "(batch, anchors, 4 + classes). Export the default ONNX (not a raw-heads layout), or "
                "pass the output name with --out.";
        return nullptr;
    }
    m.input = cfg.in_name.empty() ? m.net->getSessionInput(m.session, nullptr)
                                  : m.net->getSessionInput(m.session, cfg.in_name.c_str());
    if (!m.input) {
        error = "input '" + cfg.in_name + "' not found";
        return nullptr;
    }
    info.in_name = cfg.in_name.empty() ? m.net->getSessionInputAll(m.session).begin()->first : cfg.in_name;
    info.out_name = cfg.out_name.empty() ? outs.begin()->first : cfg.out_name;

    // Batch: try the requested size; models exported without a dynamic batch stay at 1
    std::string err;
    info.batch = std::max(1, cfg.batch);
    if (!m.resize(info.batch, err)) {
        if (!err.empty()) { error = err; return nullptr; }
        info.batch = 1;
        if (!m.resize(1, err)) {
            const std::vector<int> s = m.output ? m.output->shape() : std::vector<int>{};
            std::string sh;
            for (int v : s) sh += (sh.empty() ? "" : ",") + std::to_string(v);
            error = err.empty() ? "output has shape [" + sh + "], expected [batch, anchors, 4 + classes]" : err;
            return nullptr;
        }
    }

    // Probe on a gray input: checks the run and learns the output layout
    std::fill(m.host_in->host<float>(), m.host_in->host<float>() + m.host_in->elementSize(), 0.f);
    if (!m.run(err)) {
        error = err;
        return nullptr;
    }
    const std::vector<int> s = m.output->shape();
    m.feature_rows = s[1] <= s[2];  // far more anchors than features
    info.n_feat = m.feature_rows ? s[1] : s[2];
    info.n_anchor = m.feature_rows ? s[2] : s[1];
    if (info.n_feat < 5) {
        error = "output has " + std::to_string(info.n_feat) + " values per anchor; expected 4 box values + class scores";
        return nullptr;
    }
    if (cfg.box != "auto") {
        info.xyxy = cfg.box == "xyxy";
        info.box_source = "--box";
    } else if (cfg.box_from_metadata) {
        info.xyxy = true;
        info.box_source = "metadata";
    } else {
        // Corner boxes always have x2 >= x1 and y2 >= y1; centre boxes fail that
        // somewhere among the anchors (centre further out than the width).
        const float* p = m.host_out->host<float>();
        bool xyxy = true;
        for (int a = 0; a < info.n_anchor && xyxy; ++a) {
            auto at = [&](int f) {
                return m.feature_rows ? p[static_cast<size_t>(f) * info.n_anchor + a] : p[static_cast<size_t>(a) * info.n_feat + f];
            };
            if (at(2) < at(0) || at(3) < at(1)) xyxy = false;
        }
        info.xyxy = xyxy;
        info.box_source = "detected";
    }
    m.info = info;
    return d;
}

int MnnDetector::detect(const cv::Mat& bgr, const std::vector<Region>& regions, float conf, float nms_iou,
                        std::vector<std::vector<Detection>>& per_region) {
    Impl& m = *impl_;
    per_region.assign(regions.size(), {});
    const int S = m.cfg.size, B = m.info.batch, F = m.info.n_feat, A = m.info.n_anchor;
    const size_t in_per = static_cast<size_t>(3) * S * S, out_per = static_cast<size_t>(F) * A;
    int failed = 0, runs = 0;
    std::vector<RegionXform> xf(B);
    for (size_t start = 0; start < regions.size(); start += B) {
        const int n = static_cast<int>(std::min<size_t>(B, regions.size() - start));
        ++runs;
        float* in = m.host_in->host<float>();

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            const Region& r = regions[start + i];
            xf[i] = make_xform(r.x, r.y, r.width, r.height, S, m.cfg.topleft);
            fill_region_input(bgr.data, bgr.step[0], xf[i], S, m.cfg.norm, in + i * in_per);
        }

        std::string err;
        if (!m.run(err)) {
            std::cerr << "Inference failed on regions " << start << ".." << (start + n - 1) << ": " << err << "\n";
            failed += n;
            continue;
        }

        const float* out = m.host_out->host<float>();
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            const RegionXform& t = xf[i];
            std::vector<Candidate> cands;
            decode_head(out + i * out_per, F, A, m.feature_rows, m.info.xyxy, S, conf, cands);
            std::vector<Detection> dets;
            dets.reserve(cands.size());
            for (const Candidate& c : cands) {
                // Undo the letterbox, clamp to the region, then shift to image coordinates
                const float x1 = std::clamp((c.x1 - t.pad_l) / t.scale, 0.0f, static_cast<float>(t.rw));
                const float y1 = std::clamp((c.y1 - t.pad_t) / t.scale, 0.0f, static_cast<float>(t.rh));
                const float x2 = std::clamp((c.x2 - t.pad_l) / t.scale, 0.0f, static_cast<float>(t.rw));
                const float y2 = std::clamp((c.y2 - t.pad_t) / t.scale, 0.0f, static_cast<float>(t.rh));
                if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;
                dets.push_back({cv::Rect2f(x1 + t.rx, y1 + t.ry, x2 - x1, y2 - y1), c.score, c.cls});
            }
            // The head predicts many overlapping boxes per object
            per_region[start + i] = apply_class_aware_nms(dets, conf, nms_iou);
        }
    }
    last_ = "batch " + std::to_string(B) + ", " + std::to_string(runs) + " run(s)";
    return failed;
}
