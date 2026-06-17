# v0.1.0-blackwell-sm120

Initial experimental release for Windows Blackwell `SM120` xFormers validation.

## Release Assets

Upload these files as GitHub Release assets:

```text
xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl
mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

The MSLK wheel is a required companion dependency from the `mslk-blackwell-windows` project.

## SHA256

```text
11cf4cc2b62cba04471c0976c5c4770a2dbb8c6556dd1d1d1bb6fe916a20239d  xformers-0.0.35+af84367e.d20260613-py39-none-win_amd64.whl
682cd2c4b85c6ef192ff9f1009c3b2af079bdcd006c54e4348c33da0c9763e58  mslk-1.1.0.post1+sm120a1111-py3-none-any.whl
```

## Validation Summary

```text
Torch: 2.11.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: (12, 0)

float16: PASS
float32: FAIL
```

## A1111 Runtime Validation

```text
AUTOMATIC1111 was launched with --xformers.
WebUI launched successfully.
Model loaded successfully.
Generation completed successfully.
No xFormers runtime errors were observed.
Fast iteration testing completed.
```
