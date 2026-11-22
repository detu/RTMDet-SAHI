# RTMDet-SAHI

C++ inference for [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) object detection on
large images with **SAHI** (Slicing Aided Hyper Inference). One GPU backend is chosen at build time:

| Backend | Engine | GPUs |
|---|---|---|
| **CUDA** (default) | [MNN](https://github.com/alibaba/MNN) | NVIDIA |
| **Vulkan** | MNN | any GPU with a Vulkan driver (NVIDIA, AMD, Intel) |
| **ROCm** | [ONNX Runtime](https://onnxruntime.ai) + [MIGraphX](https://github.com/ROCm/AMDMIGraphX) | AMD GPUs supported by ROCm (e.g. Radeon RX 7000/9000, Ryzen AI Max) |

Every build can also run on the CPU.

You only need the model's **.onnx** file (e.g. exported with [LibreYOLO](https://github.com/LibreYOLO/libreyolo)).
MNN builds convert it on first use to MNN's format (`model.mnn`, next to the `.onnx`) with MNN's own converter,
which is built together with the program; the ROCm build runs the `.onnx` directly.

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
- **ROCm backend:** ROCm, and ONNX Runtime with the MIGraphX execution provider — see [ROCm build](#rocm-build-amd-gpus).

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

# or ROCm (+ CPU), AMD GPUs -- see "ROCm build" below
cmake -B build-rocm -DRTMDET_ROCM=ON -DONNXRUNTIME_INCLUDE_DIR=$HOME/onnxruntime-linux-x64-1.29.0/include
cmake --build build-rocm -j 4

# or CPU only
cmake -B build-cpu -DRTMDET_CUDA=OFF
```

Only **one** GPU backend can be built: at most one of `RTMDET_CUDA`, `RTMDET_VULKAN` and `RTMDET_ROCM` can be `ON`
(CMake stops with an error otherwise; turning Vulkan or ROCm on switches the CUDA default off). The CPU backend is
always included, so `--backend cpu` works in every build. Use a separate build directory per backend. Builds are
Release by default.

The MNN submodule (pinned at 3.6.2) is configured and built automatically, including `MNNConvert`. The first build
takes a while; on machines with little RAM keep `-j` low (`-j 2`) — an out-of-memory kill shows up as
`Error 137` / `Killed`.

Options:

| CMake option | Default | Meaning |
|---|---|---|
| `RTMDET_CUDA` | `ON` | Build MNN's CUDA backend |
| `RTMDET_VULKAN` | `OFF` | Build MNN's Vulkan backend |
| `RTMDET_ROCM` | `OFF` | ONNX Runtime + MIGraphX instead of MNN (no MNN submodule needed) |
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

### ROCm build (AMD GPUs)

Uses ONNX Runtime with AMD's MIGraphX execution provider, on native Linux (ROCm on WSL2 is limited). Written for
Ubuntu 24.04 with a recent HWE kernel, e.g. a Ryzen AI Max+ 395 (Radeon 8060S, gfx1151); not yet tested on AMD
hardware.

1. **ROCm** – install it as AMD describes for your GPU
   ([Ryzen APUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/),
   [Radeon / Instinct](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)), add yourself to the
   `render` and `video` groups, and check that `rocminfo` lists the GPU (`gfx1151` for a Ryzen AI Max).
2. **ONNX Runtime + MIGraphX** – follow AMD's
   [Install ONNX Runtime for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html)
   (a Python virtual environment with `onnxruntime` and the `onnxruntime-ep-migraphx` plugin, using the package
   version AMD lists for your ROCm release), and run its check that the MIGraphX provider is available.
3. **C++ headers** – the Python package has the library but not the headers. Download the official release archive
   of the **same** version (`python -c "import onnxruntime; print(onnxruntime.__version__)"`):

   ```bash
   V=1.29.0   # your onnxruntime version
   wget https://github.com/microsoft/onnxruntime/releases/download/v$V/onnxruntime-linux-x64-$V.tgz
   tar xf onnxruntime-linux-x64-$V.tgz -C $HOME
   ```
4. **Build** with the virtual environment active (CMake asks its Python where the packages are):

   ```bash
   source ~/venv-ort/bin/activate          # your virtual environment
   cmake -B build-rocm -DRTMDET_ROCM=ON -DONNXRUNTIME_INCLUDE_DIR=$HOME/onnxruntime-linux-x64-$V/include
   cmake --build build-rocm -j 4
   ./build-rocm/rtmdet_sahi rtmdet_s.onnx images/ -o out --csv dets.csv
   ```

   Configure prints the ONNX Runtime library and the MIGraphX plugin it found. Their folders (and pip's
   `migraphx_libs`) are stored in the program, so no `LD_LIBRARY_PATH` is needed afterwards.

Instead of the Python packages you can use an ONNX Runtime built from source with `--use_migraphx`:
`-DONNXRUNTIME_ROOT=<install folder with include/ and lib/>`. Other options: `-DRTMDET_PYTHON=<python>`,
`-DRTMDET_MIGRAPHX_EP=<plugin library>`.

At start-up MIGraphX compiles the model for the GPU, which can take a while; every run then uses the same input
shape (a full batch), so it is compiled only once per start. `auto` falls back to the CPU (with a warning) if
MIGraphX is not available or cannot load the model; `--backend rocm` makes that an error instead.
`--precision low` enables MIGraphX's fp16 mode.

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
| `-d, --backend` | `auto` | `cuda`, `vulkan` or `rocm` (whichever was built), or `cpu`; `auto` = the GPU backend if it works here, else CPU |
| `--precision` | `normal` | `normal`, `high`, or `low` (fp16 where the backend supports it) |
| `-b, --batch` | `8` | Regions per inference run (models exported with a fixed batch fall back to 1) |
| `-t, --threads` | all cores | CPU backend threads and pre/post-processing threads |
| `--fp16` | | MNN: convert the `.onnx` with fp16 weights (`model.fp16.mnn`) |
| `--reconvert` | | MNN: convert again even if the `.mnn` is up to date |
| `--ep-lib <path>` | from the build | ROCm: MIGraphX plugin EP library to use (also `$RTMDET_MIGRAPHX_EP`) |
| `--ep-option k=v` | | ROCm: extra option for the MIGraphX plugin EP (repeatable) |
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
- **ROCm: `MIGraphX not available (...)`** – the message says what is missing:
  - `libonnxruntime_providers_migraphx.so ... cannot open`: this ONNX Runtime has no built-in MIGraphX and no
    plugin was found. Install `onnxruntime-ep-migraphx` (step 2) and configure again with that virtual environment
    active, or pass `--ep-lib <path to the plugin library>`.
  - a missing `libmigraphx*` / `libamdhip64` library: run once with
    `LD_LIBRARY_PATH=<site-packages>/onnxruntime/capi:<site-packages>/migraphx_libs:/opt/rocm/lib` to confirm, then
    reconfigure so the folders are stored in the program.
  - `rocminfo` does not list the GPU: a ROCm / kernel driver problem (Ryzen AI Max needs a recent kernel), not
    this program.
- **ROCm: CMake says the ONNX Runtime headers are newer than the library** – download the headers of the library's
  version (CMake prints it).
- **Slower after a while on a laptop** – usually thermal/power throttling of the GPU; check with
  `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv -l 1`.

## License

MNN (the `MNN/` submodule) is licensed under Apache-2.0; ONNX Runtime under MIT.
