# RTMDet-SAHI

C++ inference for [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) object detection on
large images with **SAHI** (Slicing Aided Hyper Inference), running on
[MNN](https://github.com/alibaba/MNN) with a **CUDA** or **Vulkan** backend (chosen at build time), or on the CPU.

You only need the model's **.onnx** file (e.g. exported with [LibreYOLO](https://github.com/LibreYOLO/libreyolo)).
On first use it is converted to MNN's format (`model.mnn`, next to the `.onnx`) by MNN's own converter, which is
built together with the program.

## How it works

1. **Slicing** – the image is cut into overlapping slices of the model's input size (default 640×640, 20 % overlap;
   the last slice in each row/column is moved back so it ends at the image border). With `--full`, the whole image is
   added as one more region so large objects are found too.
2. **Preprocessing** – each region is letterboxed exactly like LibreYOLO's RTMDet: Pillow-style bilinear resize
   (antialiased when downscaling), image at the top-left, gray (114) padding at the bottom/right, BGR channel order,
   mmdet mean/std normalisation.
3. **Inference** – regions are run in batches (`--batch`, default 8) on the selected MNN backend.
4. **Per-region NMS** – class-aware NMS inside each region (`--iou`).
5. **Merging across slices** – as in the SAHI library: greedy non-maximum *merging* (NMM) with the IOS metric
   (intersection over the smaller box), so an object cut by a slice edge becomes one box again. `--merge nms` and
   `--match iou` give plain NMS instead.

Preprocessing, decoding and un-letterboxing are parallelised with OpenMP; the next image is decoded and the previous
result is drawn and saved in the background.

## Requirements

- CMake ≥ 3.18, a C++17 compiler, OpenCV (core, imgproc, imgcodecs, dnn), OpenMP
- **CUDA backend:** the CUDA toolkit (`nvcc`). cuDNN is **not** needed.
- **Vulkan backend:** nothing at build time (MNN ships its own Vulkan headers); a Vulkan driver at run time.

```bash
sudo apt install build-essential cmake libopencv-dev
```

## Build

```bash
git clone --recursive https://github.com/detu/RTMDet-SAHI.git
cd RTMDet-SAHI
# (already cloned without --recursive?  git submodule update --init)

# CUDA (+ CPU), the default
cmake -B build
cmake --build build -j 2

# or Vulkan (+ CPU)
cmake -B build-vulkan -DRTMDET_CUDA=OFF -DRTMDET_VULKAN=ON
cmake --build build-vulkan -j 2

# or CPU only
cmake -B build-cpu -DRTMDET_CUDA=OFF
```

Only **one** GPU backend can be built: `RTMDET_CUDA` and `RTMDET_VULKAN` cannot both be `ON` (CMake stops with an
error). The CPU backend is always included, so `--backend cpu` works in every build. Use a separate build directory
per backend. Builds are Release by default.

The MNN submodule (pinned at 3.6.2) is configured and built automatically, including `MNNConvert`. The first build
takes a while; on machines with little RAM keep `-j` low (`-j 2`) — an out-of-memory kill shows up as
`Error 137` / `Killed`.

Options:

| CMake option | Default | Meaning |
|---|---|---|
| `RTMDET_CUDA` | `ON` | Build MNN's CUDA backend |
| `RTMDET_VULKAN` | `OFF` | Build MNN's Vulkan backend |
| `RTMDET_CUDA_ARCH` | *(nvidia-smi)* | GPU compute capability to compile for, e.g. `8.9` (RTX 40xx / RTX Ada laptop GPUs), `8.6` (RTX 30xx), `12.0` (RTX 50xx) |
| `RTMDET_CUDA_NATIVE_ARCH` | `ON` | Compile the CUDA kernels only for that one GPU (much faster build) |
| `RTMDET_NVCC` | *(search)* | Path to `nvcc` |

`nvcc` is searched on `PATH`, in `$CUDA_HOME`, `$CUDA_PATH`, `/usr/local/cuda` and `/usr/local/cuda-*` (newest
first), so it does not need to be on `PATH` (common on WSL2).

Example for a laptop RTX 1000 Ada under WSL2:

```bash
cmake -B build -DRTMDET_CUDA_ARCH=8.9
cmake --build build -j 2
```

## Usage

```bash
./build/rtmdet_sahi <model.onnx|model.mnn> <image|dir> [<image|dir> ...] [options]
```

```bash
# One image -> output.jpg
./build/rtmdet_sahi rtmdet_s.onnx big_image.jpg

# A folder (recursively) -> out/<name>_det.jpg, plus all detections as CSV
./build/rtmdet_sahi rtmdet_s.onnx images/ -r -o out --csv detections.csv

# Vulkan build, whole image as an extra region, own class names
./build-vulkan/rtmdet_sahi rtmdet_s.onnx images/ --full -n classes.txt
```

The model is loaded once and every image is processed with it. Output boxes are drawn with a label (class name and
score) on a green background.

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | `output.jpg` / `output/` | Output file (one image) or directory (several images) |
| `--csv <file>` | | Write all detections (`image,class_id,class_name,score,x1,y1,x2,y2`) |
| `-n, --names <file>` | model metadata | Class names, one per line: line 1 = class 0, line 2 = class 1, … |
| `-r, --recursive` | | Scan directories recursively |
| `-d, --backend` | `auto` | `cuda` or `vulkan` (whichever was built), or `cpu`; `auto` = the GPU backend if it works here, else CPU |
| `--precision` | `normal` | `normal`, `high`, or `low` (fp16 where the backend supports it) |
| `-b, --batch` | `8` | Regions per inference run (models exported with a fixed batch fall back to 1) |
| `-t, --threads` | all cores | CPU backend threads and pre/post-processing threads |
| `--fp16` | | Convert the `.onnx` with fp16 weights (`model.fp16.mnn`) |
| `--reconvert` | | Convert again even if the `.mnn` is up to date |
| `-s, --size` | model | Model input size |
| `--slice` | input size | Slice size |
| `--overlap` | `0.2` | Slice overlap ratio |
| `--full` | | Also detect on the whole image |
| `--conf` | `0.30` | Confidence threshold |
| `--iou` | `0.60` | NMS IoU threshold inside a region |
| `--merge` | `nmm` | Cross-slice merging: `nmm` (merge boxes) or `nms` (keep the best) |
| `--match` | `ios` | Overlap metric for merging: `ios` or `iou` |
| `--match-thresh` | `0.5` | Overlap above which boxes are merged |
| `--preprocess` | metadata / `rtmdet` | `rtmdet` (BGR, mmdet mean/std) or `yolo` (RGB, 0–1) |
| `--pad` | metadata / `topleft` | Letterbox placement: `topleft` or `center` |
| `--box` | metadata / detected | Box format of the output: `xyxy` or `cxcywh` |
| `--in`, `--out` | | Input / output tensor names |

### The model

The program expects the usual exported RTMDet ONNX: one input `(B, 3, S, S)` and one decoded output
`(B, anchors, 4 + classes)` (or transposed) with `x1, y1, x2, y2` boxes in input pixels and per-class scores.
LibreYOLO's metadata (class names, input size, model family, letterbox placement) is read from the `.onnx`
automatically; it is also used when you pass a `.mnn` and the `.onnx` of the same name is next to it.

- Export with a **dynamic batch** so several slices run at once, e.g. LibreYOLO `--dynamic`. A model exported with a
  fixed batch of 1 still works, one slice per run.
- A "raw heads" export (several outputs) is rejected — export the default decoded output.
- Converting fails? The converter's output is kept in `model.mnn.convert.log`.

## Troubleshooting

- **`nvcc not found`** – install the CUDA toolkit or pass `-DRTMDET_NVCC=/usr/local/cuda/bin/nvcc`; or build the
  Vulkan backend instead: `-DRTMDET_CUDA=OFF -DRTMDET_VULKAN=ON`.
- **`Error generating file ... MNN_Cuda_Main_generated_*.cu.o`** – nvcc failed on one of MNN's CUDA files; the real
  message is a few lines above it (rebuild with `cmake --build build -j 1 2>&1 | tee build.log` to see it clearly).
  - `Unsupported gpu architecture 'compute_60'` (CUDA 13): the GPU architecture was not known while configuring.
    It is now read from `nvidia-smi`; otherwise pass it, e.g. `-DRTMDET_CUDA_ARCH=8.9`, in a **new** build directory.
  - `cutlass/matrix.h: ... has no member named 'set_slice3x3'` (GCC 14 and newer): a typo in MNN's bundled CUTLASS
    that newer GCC rejects. Handled automatically (`-Wno-template-body` is passed to the host compiler).
  - `Killed` / `Error 137` / `cicc died`: out of memory; MNN's CUTLASS kernels are heavy to compile. Use `-j 1`.
- **`Warning: cuda backend not available here; MNN uses CPU`** – the CUDA driver/GPU is not visible (check
  `nvidia-smi`). Under WSL2, `/usr/lib/wsl/lib` must be on the library path.
- **Vulkan under WSL2** uses the Dozen (D3D12) driver, which works but is slower than CUDA.
- **Slower after a while on a laptop** – usually thermal/power throttling of the GPU; check with
  `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv -l 1`.

## License

MNN (the `MNN/` submodule) is licensed under Apache-2.0.
