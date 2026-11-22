// ONNX Runtime engine: MIGraphX execution provider (AMD GPUs with ROCm) or CPU.
//
// MIGraphX compiles the model for the exact input shape on first use, so every
// run uses the same batch size (unused slots of the last batch are left as they
// are) and the shape never changes.

#include "runner.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rtmdet_sahi");
    return env;
}

// MIGraphX through the EP built into libonnxruntime (an ORT build with
// --use_migraphx, or AMD's onnxruntime-migraphx package).
bool append_builtin_migraphx(Ort::SessionOptions& so, const EngineConfig& cfg, std::string& why) {
    OrtMIGraphXProviderOptions o{};
    o.device_id = 0;
    o.migraphx_fp16_enable = cfg.precision == "low" ? 1 : 0;
    o.migraphx_mem_limit = SIZE_MAX;
    try {
        so.AppendExecutionProvider_MIGraphX(o);
        return true;
    } catch (const Ort::Exception& e) {
        why = e.what();
        return false;
    }
}

// MIGraphX as a plugin EP library (AMD's onnxruntime-ep-migraphx package):
// registered with the environment, then selected by device.
// Plugin EP library: --ep-lib, else $RTMDET_MIGRAPHX_EP, else the one CMake found
// in the onnxruntime-ep-migraphx Python package (if it still exists).
std::string plugin_library(const EngineConfig& cfg) {
    if (!cfg.ep_lib.empty()) return cfg.ep_lib;
    if (const char* e = std::getenv("RTMDET_MIGRAPHX_EP")) {
        if (*e) return e;
    }
#ifdef RTMDET_DEFAULT_MIGRAPHX_EP
    std::error_code ec;
    if (*RTMDET_DEFAULT_MIGRAPHX_EP && std::filesystem::is_regular_file(RTMDET_DEFAULT_MIGRAPHX_EP, ec))
        return RTMDET_DEFAULT_MIGRAPHX_EP;
#endif
    return "";
}

bool append_plugin_migraphx(Ort::SessionOptions& so, const EngineConfig& cfg, const std::string& lib,
                            std::string& why) {
    try {
        static bool registered = false;
        if (!registered) {
            ort_env().RegisterExecutionProviderLibrary("MIGraphX",
                                                       std::basic_string<ORTCHAR_T>(lib.begin(), lib.end()));
            registered = true;
        }
        std::vector<Ort::ConstEpDevice> devices;
        for (const Ort::ConstEpDevice& d : ort_env().GetEpDevices()) {
            if (lower(d.EpName()).find("migraphx") != std::string::npos) devices.push_back(d);
        }
        if (devices.empty()) {
            why = "the plugin EP in " + lib + " reports no MIGraphX device";
            return false;
        }
        devices.resize(1);  // first GPU
        std::unordered_map<std::string, std::string> opts;
        if (cfg.precision == "low") opts["migraphx_fp16_enable"] = "1";
        for (const auto& kv : cfg.ep_options) opts[kv.first] = kv.second;
        so.AppendExecutionProvider_V2(ort_env(), devices, opts);
        return true;
    } catch (const Ort::Exception& e) {
        why = e.what();
        return false;
    }
}

class OrtRunner : public Runner {
public:
    std::unique_ptr<Ort::Session> session;
    std::string in_name, out_name;
    std::vector<int64_t> in_shape, out_shape;
    std::vector<float> in_buf;
    std::vector<Ort::Value> outputs;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    float* input() override { return in_buf.data(); }

    bool run(std::string& error) override {
        try {
            Ort::Value x = Ort::Value::CreateTensor<float>(mem, in_buf.data(), in_buf.size(), in_shape.data(),
                                                           in_shape.size());
            const char* in_names[] = {in_name.c_str()};
            const char* out_names[] = {out_name.c_str()};
            outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names, 1);
            out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            return true;
        } catch (const Ort::Exception& e) {
            error = e.what();
            return false;
        }
    }

    std::vector<int64_t> output_shape() const override { return out_shape; }
    const float* output() const override { return outputs[0].GetTensorData<float>(); }
};

}  // namespace

const char* engine_name() { return "ONNX Runtime"; }

std::vector<std::string> available_backends() { return {"rocm", "cpu"}; }

std::unique_ptr<Runner> create_runner(const EngineConfig& cfg, RunnerInfo& info, std::string& error) {
    ort_env();  // the environment must exist before execution providers are added
    std::unique_ptr<OrtRunner> r(new OrtRunner);
    // want_gpu: try MIGraphX first. In "auto" mode a MIGraphX failure (missing EP,
    // or the EP can't take the model) falls back to the CPU.
    bool want_gpu = cfg.backend == "auto" || cfg.backend == "rocm";
    for (;;) {
        std::string stage = "load";
        try {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            so.SetIntraOpNumThreads(std::max(1, cfg.threads));
            info.backend = "CPU (" + std::to_string(std::max(1, cfg.threads)) + " threads)";
            if (want_gpu) {
                // The plugin EP if there is one, else the EP built into ONNX Runtime
                std::string why;
                bool ok = false;
                const std::string lib = plugin_library(cfg);
                if (!lib.empty()) {
                    ok = append_plugin_migraphx(so, cfg, lib, why);
                    if (ok) info.backend = "MIGraphX (ROCm, plugin EP)";
                    else why = "plugin EP " + lib + ": " + why;
                }
                if (!ok && cfg.ep_lib.empty()) {
                    std::string why_builtin;
                    ok = append_builtin_migraphx(so, cfg, why_builtin);
                    if (ok) info.backend = "MIGraphX (ROCm)";
                    else why += (why.empty() ? "" : "; built-in EP: ") + why_builtin;
                }
                if (!ok) {
                    if (cfg.backend == "rocm") {
                        error = "MIGraphX execution provider not available: " + why +
                                "\n  Use an ONNX Runtime built with MIGraphX, or AMD's plugin EP with --ep-lib <path>";
                        return nullptr;
                    }
                    std::cerr << "Warning: MIGraphX not available (" << why << "); running on the CPU\n";
                    want_gpu = false;
                }
            }
            if (cfg.precision == "low" && !want_gpu) std::cerr << "Note: --precision low has no effect on the CPU\n";
            if (want_gpu) std::cout << "Compiling the model with MIGraphX (can take a while) ..." << std::endl;
            stage = "session";
            r->session.reset(new Ort::Session(ort_env(), cfg.model_path.c_str(), so));
            break;
        } catch (const Ort::Exception& e) {
            if (want_gpu && cfg.backend == "auto" && stage == "session") {
                std::cerr << "Warning: MIGraphX could not load the model (" << e.what() << "); running on the CPU\n";
                want_gpu = false;
                continue;
            }
            error = std::string("ONNX Runtime cannot load ") + cfg.model_path + ": " + e.what();
            return nullptr;
        }
    }

    Ort::Session& s = *r->session;
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t n_in = s.GetInputCount(), n_out = s.GetOutputCount();
    if (n_in < 1) {
        error = "the model has no inputs";
        return nullptr;
    }
    if (cfg.out_name.empty() && n_out != 1) {
        error = "the model has " + std::to_string(n_out) + " outputs; expected one decoded output "
                "(batch, anchors, 4 + classes). Export the default ONNX (not a raw-heads layout), or "
                "pass the output name with --out.";
        return nullptr;
    }
    size_t in_idx = 0;
    if (!cfg.in_name.empty()) {
        in_idx = n_in;
        for (size_t i = 0; i < n_in; ++i) {
            if (s.GetInputNameAllocated(i, alloc).get() == cfg.in_name) in_idx = i;
        }
        if (in_idx == n_in) {
            error = "input '" + cfg.in_name + "' not found";
            return nullptr;
        }
    }
    r->in_name = s.GetInputNameAllocated(in_idx, alloc).get();
    if (cfg.out_name.empty()) {
        r->out_name = s.GetOutputNameAllocated(0, alloc).get();
    } else {
        bool found = false;
        for (size_t i = 0; i < n_out; ++i) found |= s.GetOutputNameAllocated(i, alloc).get() == cfg.out_name;
        if (!found) {
            error = "output '" + cfg.out_name + "' not found";
            return nullptr;
        }
        r->out_name = cfg.out_name;
    }
    info.in_name = r->in_name;
    info.out_name = r->out_name;

    // Input shape: (batch, 3, size, size); a fixed batch in the model wins
    const std::vector<int64_t> dims = s.GetInputTypeInfo(in_idx).GetTensorTypeAndShapeInfo().GetShape();
    if (dims.size() != 4) {
        error = "input has " + std::to_string(dims.size()) + " dimensions; expected (batch, 3, height, width)";
        return nullptr;
    }
    for (int k : {2, 3}) {
        if (dims[k] > 0 && dims[k] != cfg.size) {
            error = "the model's input is " + std::to_string(dims[2]) + "x" + std::to_string(dims[3]) +
                    " but the input size is " + std::to_string(cfg.size) + " (use --size " +
                    std::to_string(dims[2]) + ")";
            return nullptr;
        }
    }
    info.batch = dims[0] > 0 ? static_cast<int>(dims[0]) : std::max(1, cfg.batch);
    r->in_shape = {info.batch, 3, cfg.size, cfg.size};
    r->in_buf.assign(static_cast<size_t>(info.batch) * 3 * cfg.size * cfg.size, 0.0f);
    return r;
}
