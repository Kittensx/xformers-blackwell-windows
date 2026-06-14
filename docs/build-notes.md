# Build Notes

This repository documents an experimental Windows xFormers build for NVIDIA Blackwell `SM120`.

## Source

```text
facebookresearch/xformers
Commit: af84367eea46dba5da5b7b1ce55efc27904b4e79
git describe: v0.0.35-9-gaf84367e
```

## Target Environment

```text
Windows x64
Python 3.10.6
PyTorch 2.11.0+cu128
CUDA Toolkit 12.8.61
Visual Studio 2022 Build Tools x64
MSVC 19.44
```

## Important Build Settings

```bat
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=12.0
set XFORMERS_BUILD_TYPE=Release
set NVCC_FLAGS=-allow-unsupported-compiler
set MAX_JOBS=1
```

`TORCH_CUDA_ARCH_LIST=12.0` is required so the wheel contains `SM120` code for the validated Blackwell GPU.

`DISTUTILS_USE_SDK=1` is required when building inside an already initialized Visual Studio x64 developer environment.

`MAX_JOBS=1` was used to keep the Windows CUDA build stable.

## Release Asset

The wheel should be uploaded to GitHub Releases, not committed to the repository:

```text
xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl
```
