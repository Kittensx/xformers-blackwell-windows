# xformers-blackwell-windows

Experimental xFormers + MSLK builds validated on NVIDIA Blackwell (SM120) GPUs, CUDA 12.8, PyTorch 2.11, and RTX 5070 Laptop hardware.

## Project Overview

xFormers is a library of optimized Transformer building blocks, including memory-efficient attention operators used by many Stable Diffusion and PyTorch workflows. For image-generation users, xFormers matters because efficient attention can reduce VRAM pressure and improve iteration speed in applications such as AUTOMATIC1111, ComfyUI, Forge, and custom diffusion pipelines.

This repository exists because current xFormers distributions did not provide a validated Windows solution for the tested Blackwell environment. The goal is to provide a validated reference build for a specific hardware/software configuration, not to replace upstream xFormers.

Blackwell support is important because RTX 50-series GPUs report compute capability `12.0` / `SM120`, and some existing attention paths either do not build for this architecture or reject it as too new. This repository documents a working xFormers build and validation process for one concrete Windows Blackwell setup.

This xFormers build depends on MSLK. In the validated xFormers source line, `memory_efficient_attention` is exposed through MSLK, so the companion MSLK package must be installed before the xFormers wheel can provide the expected attention API.

## Experimental Warning

This is an experimental release.

- Validated only on NVIDIA GeForce RTX 5070 Laptop GPU.
- Other RTX 50-series Blackwell GPUs may work but are not yet validated.
- No guarantee is made for training workloads.
- No guarantee is made for SDXL, Flux, video generation, or other non-validated workflows.
- Windows x64 is the only validated operating system.
- Users should keep backups of working environments before installing experimental wheels.

Use this package only if you are comfortable testing a targeted compatibility build.

## Tested Environment

```text
Windows x64
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: 12.0 / SM120
Python: 3.10.6
PyTorch: 2.11.0+cu128
CUDA Runtime: 12.8
CUDA Toolkit: 12.8.61
xFormers: 0.0.35+af84367e
MSLK: 1.1.0.post1+sm120a1111
Primary validation app: AUTOMATIC1111 Stable Diffusion WebUI
```

## Why This Build Exists

The original Windows Stable Diffusion environment did not have a working xFormers path for the tested Blackwell GPU.

Before installing xFormers, the WebUI reported:

```text
No module 'xformers'. Proceeding without it.
```

After installing an available xFormers wheel, attention dispatch still failed with messages of this form:

```text
No operator found for memory_efficient_attention_forward
```

The failure occurred on:

```text
RTX 5070 Laptop GPU
Compute Capability 12.0
```

The objective was to create a functioning and validated xFormers environment for this hardware and software stack. The final working path required both:

1. A Windows xFormers wheel built for `SM120`.
2. A compatible MSLK Python package with a Triton Split-K adjustment for `head_dim=512`.

## Companion MSLK Dependency

Required companion repository:

```text
mslk-blackwell-windows
```

MSLK Repository Link:

```text
https://github.com/Kittensx/mslk-blackwell-windows
```

Install MSLK first. This xFormers build depends on the companion MSLK package:

```text
mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

The MSLK repository contains the Windows compatibility notes and documents the required Triton Split-K change:

```text
mslk/attention/fmha/triton_splitk.py
BLOCK_N: 64 -> 32
```

## Release Artifacts

Upload this xFormers wheel as a GitHub Release asset:

```text
xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl
```

Do not commit wheel binaries directly to the repository. They should be attached to the GitHub Release.

Naming convention:

- `0.0.35+af84367e`: xFormers version lineage and source commit identifier.
- `d20260613`: local build date marker.
- `py39-none-win_amd64`: generated wheel compatibility tag.
- `win_amd64`: Windows x64 platform target.

Source provenance:

```text
xFormers source: facebookresearch/xformers
Commit: af84367eea46dba5da5b7b1ce55efc27904b4e79
git describe: v0.0.35-9-gaf84367e
```

Build target:

```text
TORCH_CUDA_ARCH_LIST=12.0
CUDA Toolkit 12.8.61
PyTorch 2.11.0+cu128
Python 3.10.6
Windows x64
```

## SHA256 Verification

Known checksums:

```text
11cf4cc2b62cba04471c0976c5c4770a2dbb8c6556dd1d1d1bb6fe916a20239d  xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl
682cd2c4b85c6ef192ff9f1009c3b2af079bdcd006c54e4348c33da0c9763e58  mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

Verify on Windows:

```bat
certutil -hashfile xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl SHA256
certutil -hashfile mslk-1.1.0.post1+sm120a1111-py3-none-any.whl SHA256
```

Checksum verification matters because these are experimental binary artifacts. Verifying the hash confirms that the wheel you installed is the same artifact that was validated.

## Installation

Install inside the same Python environment that will run xFormers, A1111, ComfyUI, Forge, or your PyTorch application.

For AUTOMATIC1111:

```bat
cd /d C:\A1111\stable-diffusion-webui
venv\Scripts\activate.bat

python -m pip uninstall -y xformers mslk
python -m pip install --no-deps mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
python -m pip install --no-deps xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl
```

The `--no-deps` flag is intentional. This release was validated against a specific PyTorch/CUDA/MSLK/xFormers environment, and dependency auto-resolution may replace working packages with incompatible versions.

## Validation

Validation occurred at three levels:

1. Import validation.
2. Standalone attention validation.
3. Real-world AUTOMATIC1111 validation.

## Validation Scripts

### Import Validation

The following imports succeeded:

```python
import xformers
import xformers.ops
from xformers.ops import memory_efficient_attention
```

Result:

```text
PASS
```

### Validation Script 1

Script:

```text
tests/test_xformers_a1111_shape.py
```

Purpose:

Baseline validation using the original failing A1111 attention shape:

```text
query/key/value shape: (1, 9600, 1, 512)
```

Expected result:

```text
float16 PASS
float32 FAIL
```

### Validation Script 2

Script:

```text
tests/test_xformers_a1111_tuned.py
```

Purpose:

Validation using the MSLK tuning configuration that was later packaged into the companion MSLK wheel.

Tuning configuration:

```python
BLOCK_N = 32
NUM_WARPS = 2
NUM_STAGES = 1
AUTOTUNE = False
```

Expected result:

```text
float16 PASS
float32 FAIL
```

## Actual Validation Results

```text
Torch: 2.11.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: (12, 0)

Testing torch.float16
PASS torch.Size([1, 9600, 1, 512]) torch.float16 finite=True

Testing torch.float32
FAIL

Summary:
float16: PASS
float32: FAIL
```

Interpretation:

- `float16: PASS` is the required success condition.
- `float32: FAIL` is expected.
- A1111 must run attention in half precision.
- The validated path is intended for `float16` / `bfloat16`-style attention, not full precision `float32` attention.

## A1111 Runtime Validation

AUTOMATIC1111 Stable Diffusion WebUI was the primary real-world validation environment.

A1111 was launched with:

```bat
set COMMANDLINE_ARGS=--xformers
```

Validation outcome:

```text
AUTOMATIC1111 was launched with --xformers.
WebUI launched successfully.
Model loaded successfully.
Generation completed successfully.
No xFormers runtime errors were observed.
Fast iteration testing completed.
```

This is stronger validation than a synthetic benchmark alone because it verifies the package in the real application path where model loading, attention selection, image generation, and repeated iteration all occur together.

A1111 is not the only intended usage environment. ComfyUI, Forge, and custom PyTorch/xFormers workflows may also benefit, but they must be validated independently.

## xformers.info Validation

The build was inspected with:

```bat
python -m xformers.info
```

Relevant confirmed values:

```text
pytorch.version: 2.11.0+cu128
pytorch.cuda: available
gpu.compute_capability: 12.0
gpu.name: NVIDIA GeForce RTX 5070 Laptop GPU
build.cuda_version: 1208
```

Full output should be stored in:

```text
docs/xformers-info-release.txt
```

## Recommended A1111 Configuration

Use:

```bat
set COMMANDLINE_ARGS=--xformers
```

Avoid:

```text
--no-half
--precision full
--force-enable-xformers
--opt-sdp-attention
--opt-sdp-no-mem-attention
```

Do not combine `--xformers` with SDP attention flags while testing. Mixed attention flags make it unclear which attention path actually ran, and can hide whether xFormers is working.

The validated path expects half-precision attention tensors. `--no-half` and `--precision full` can force float32 paths that are expected to fail for this workload.

## Limitations

- Float32 attention is unsupported for the validated shape.
- RTX 5070 Laptop GPU is the only validated GPU.
- Windows x64 is the only validated operating system.
- No Linux testing was performed.
- No multi-GPU testing was performed.
- No training validation was performed.
- No SDXL-specific benchmarking was performed.
- No Flux benchmarking was performed.
- No video-generation benchmarking was performed.
- No ComfyUI-specific validation was performed.
- No Forge-specific validation was performed.
- Performance and VRAM usage may differ across models, drivers, and RTX 50-series GPUs.

## Future Work

- RTX 5060 validation.
- RTX 5070 desktop validation.
- RTX 5080 validation.
- RTX 5090 validation.
- Additional CUDA Toolkit versions.
- Additional PyTorch versions.
- Additional NVIDIA driver versions.
- Upstream Blackwell support tracking.
- Performance benchmarking.
- VRAM benchmarking.
- ComfyUI validation.
- Forge validation.
- SDXL validation.
- Flux validation.
- Investigation of upstream-compatible MSLK and xFormers fixes.

## Credits

This work depends on the upstream projects and communities that made the stack possible:

- Facebook Research xFormers
- Meta PyTorch MSLK
- Triton
- PyTorch
- AUTOMATIC1111 Stable Diffusion WebUI

## License Notes

This repository distributes documentation, validation scripts, and release metadata for experimental xFormers builds. It does not relicense upstream projects.

The xFormers project is developed upstream by Facebook Research. Refer to the upstream xFormers repository for its license.

The companion MSLK dependency is developed upstream by Meta PyTorch. Refer to the upstream MSLK repository for its license.
