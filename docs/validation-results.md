# Validation Results

Validation was performed on the following environment:

```text
Torch: 2.11.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: (12, 0)
```

Standalone attention validation used:

```text
query/key/value shape: (1, 9600, 1, 512)
```

Result summary:

```text
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

```text
AUTOMATIC1111 was launched with --xformers.
WebUI launched successfully.
Model loaded successfully.
Generation completed successfully.
No xFormers runtime errors were observed.
Fast iteration testing completed.
```
