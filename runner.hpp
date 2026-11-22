// Inference engines behind the detector: a Runner owns one model session with a
// fixed batch size and a host input/output buffer. Exactly one engine is built:
//   MNN (CUDA, Vulkan or CPU)            -> mnn_runner.cpp
//   ONNX Runtime (MIGraphX/ROCm or CPU)  -> ort_runner.cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "region_ops.hpp"

struct EngineConfig {
    std::string model_path;            // .mnn for MNN, .onnx for ONNX Runtime
    std::string backend = "auto";      // auto or one of available_backends()
    std::string precision = "normal";  // normal, high, low (fp16 where the backend supports it)
    int threads = 4;                   // CPU threads
    int batch = 8;                     // regions per run (if the model accepts a batch dimension)
    int size = 640;                    // network input size
    InputNorm norm;
    bool topleft = true;               // letterbox placement
    std::string box = "auto";          // auto, xyxy, cxcywh
    bool box_from_metadata = false;    // metadata says corner boxes (LibreYOLO RTMDet / YOLOv9)
    std::string in_name, out_name;
    // ONNX Runtime only
    std::string ep_lib;                // MIGraphX plugin EP library (onnxruntime-ep-migraphx)
    std::vector<std::pair<std::string, std::string>> ep_options;  // extra MIGraphX EP options
};

struct RunnerInfo {
    std::string backend;  // what actually runs, e.g. "CUDA", "MIGraphX (ROCm)", "CPU (8 threads)"
    std::string in_name, out_name;
    int batch = 1;
};

class Runner {
public:
    virtual ~Runner() = default;
    // Host buffer for batch * 3 * size * size floats (NCHW)
    virtual float* input() = 0;
    virtual bool run(std::string& error) = 0;
    // Valid after a successful run(): (batch, d1, d2) and the host data
    virtual std::vector<int64_t> output_shape() const = 0;
    virtual const float* output() const = 0;
};

// Opens the model on the requested backend (falling back as "auto" allows) with
// cfg.batch regions per run, or 1 if the model has a fixed batch of 1.
std::unique_ptr<Runner> create_runner(const EngineConfig& cfg, RunnerInfo& info, std::string& error);

// Backends of the engine in this build, GPU first; always ends with "cpu".
std::vector<std::string> available_backends();

// Name of the inference engine in this build ("MNN" or "ONNX Runtime").
const char* engine_name();
