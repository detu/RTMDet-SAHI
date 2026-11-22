// MNN inference backend: CUDA, Vulkan or CPU, chosen at run time.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "detector.hpp"
#include "region_ops.hpp"

struct MnnConfig {
    std::string model_path;      // .mnn
    std::string backend = "auto";  // auto (cuda, then vulkan, then cpu), cuda, vulkan, cpu
    std::string precision = "normal";  // normal, high, low (fp16 where the backend supports it)
    int threads = 4;             // CPU backend threads
    int batch = 8;               // regions per run (if the model accepts a batch dimension)
    int size = 640;              // network input size
    InputNorm norm;
    bool topleft = true;         // letterbox placement
    std::string box = "auto";    // auto, xyxy, cxcywh
    bool box_from_metadata = false;  // metadata says corner boxes (LibreYOLO RTMDet / YOLOv9)
    std::string in_name, out_name;
};

struct MnnInfo {
    std::string backend;     // what MNN actually runs on, e.g. "CUDA"
    std::string in_name, out_name;
    int batch = 1;
    bool xyxy = true;
    std::string box_source;  // "--box", "metadata", "detected"
    int n_feat = 0, n_anchor = 0;
};

class MnnDetector {
public:
    // Loads the model, creates the session and checks it on a blank input.
    // Returns null and sets error on failure.
    static std::unique_ptr<MnnDetector> create(const MnnConfig& cfg, MnnInfo& info, std::string& error);
    ~MnnDetector();

    // Run the model on every region (letterboxed to the input size). per_region[i]
    // gets region i's detections in image coordinates, after class-aware NMS
    // within the region. Returns the number of regions that failed.
    int detect(const cv::Mat& bgr, const std::vector<Region>& regions, float conf, float nms_iou,
               std::vector<std::vector<Detection>>& per_region);

    std::string last_run_desc() const { return last_; }

private:
    MnnDetector() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_;
};

// Names of the backends compiled into this build (always includes "cpu").
std::vector<std::string> mnn_available_backends();
