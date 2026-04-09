# Skip-Softmax Sparse Attention for Diffusion Models

Skip-softmax sparse attention (BLASST, <https://arxiv.org/pdf/2512.12087>) skips KV
tiles whose attention scores are negligible during the FlashAttention computation,
reducing FLOPs without retraining.

Two modes are supported:
- **Fixed raw threshold** — pass a log2-space threshold directly to the Triton
  kernel. No calibration needed. Good for quick testing and sweeps.
- **Calibrated threshold** — an exponential model
  (`scale_factor = a * exp(b * target_sparsity)`) is calibrated once via the
  Triton calibration kernel, then the target sparsity can be adjusted at runtime
  without recalibration.

## Supported Models

| Model | Script | Notes |
|-------|--------|-------|
| WAN 2.2 5B | `wan22_skip_softmax.py` | Single transformer, self-attention only |
| WAN 2.2 14B | `wan22_skip_softmax.py` | Dual transformer (auto-detected) |
| LTX-2 | (coming soon) | Via `ltx_triton_attention.py` backend |

## Quick Start

```bash
# Fixed raw threshold (no calibration, fast)
python wan22_skip_softmax.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --raw-threshold -0.7 \
    --prompt "A cat playing piano" --output out.mp4

# With calibration
python wan22_skip_softmax.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --calibrate --target-sparsity 0.5 \
    --prompt "A cat playing piano" --output out.mp4

# Dense baseline (no sparsity, for comparison)
python wan22_skip_softmax.py \
    --model-path /path/to/Wan2.2-T2V-A14B-Diffusers \
    --baseline \
    --prompt "A cat playing piano" --output baseline.mp4

# Report runtime sparsity (per-layer tile skip ratios)
python wan22_skip_softmax.py \
    --raw-threshold -0.7 --report-avg-sparsity \
    --prompt "A cat playing piano" --output out.mp4
```

## Architecture

### Inference Path (Triton kernel with tile skipping)

```text
SparseAttentionModule.forward()
  └─ triton_skip_softmax._triton_inference_context()
       ├─ Priority: raw_threshold > scale_factor (calibrated) > static threshold
       ├─ _set_triton_backends(raw_threshold=X)  or  (scale_factor=X)
       ├─ attention_backend("modelopt_triton")
       └─ _diffusers_triton_attention() → attention()
            └─ _attn_fwd kernel: skip tiles where tile_row_max < row_max + threshold
```

### Calibration Path (Triton calibration kernel)

```text
mtsa.sparsify(transformer, config, forward_loop)
  ├─ apply_mode() → replace attention with SparseAttentionModule
  └─ calibrate()
       ├─ DynamicThresholdCalibrator._set_thresholds()
       │    └─ method._threshold_trials = [1e-6, ..., 9.9e-1]
       ├─ forward_loop(model)
       │    └─ SparseAttentionModule.forward()
       │         └─ triton_skip_softmax._triton_calibration_context()
       │              ├─ set_triton_skip_softmax_config(calibration_mode=True)
       │              ├─ attention_backend("modelopt_triton")
       │              └─ _diffusers_triton_attention() → attention_calibrate()
       │                   └─ _attn_fwd_calibrate kernel:
       │                        - Full attention (no skipping) for correct output
       │                        - Vectorized multi-threshold sparsity measurement
       │                        - Per-program output buffers (no atomic contention)
       │                        - Python-side reduction: sum across programs
       ├─ Fit: scale_factor = a * exp(b * sparsity)
       └─ Apply a, b to all modules
            └─ Inference: threshold = scale_factor / seq_k
```

## Core Files

### Triton Kernels (`modelopt/torch/kernels/`)

| File | Role |
|------|------|
| `triton_fa.py` | `_attn_fwd`: forward kernel with optional tile skipping + sparsity measurement. `_attn_fwd_calibrate`: calibration kernel with vectorized multi-threshold testing and per-program buffers (zero atomic contention). `attention()` and `attention_calibrate()` Python APIs. |

### Sparse Attention Methods (`modelopt/torch/sparsity/attention_sparsity/methods/`)

| File | Role |
|------|------|
| `triton_skip_softmax.py` | Primary method for diffusion models. Calibration context → Triton calibration kernel. Inference context → Triton forward kernel. Supports `scale_factor` (calibrated), `raw_threshold` (direct), and static `skip_softmax_threshold`. |
| `flash_skip_softmax.py` | PyTorch-based method for HF LLMs (not used by diffusers/LTX). |
| `registry.py` | Base class `SparseAttentionMethod` with `calibration_params`, `target_sparse_ratio`, `set_calibration_mode()`. |

### Kernel Backends (`modelopt/torch/sparsity/attention_sparsity/kernels/`)

| File | Role |
|------|------|
| `diffusers_triton_attention.py` | Registers `modelopt_triton` backend in diffusers. Handles calibration mode (→ `attention_calibrate`) and inference mode (→ `attention` with `scale_factor/seq_k` or `raw_threshold`). Runtime sparsity counter accumulation. |
| `ltx_triton_attention.py` | Patches `ltx_core.Attention` modules for Triton dispatch. Same calibration/inference modes. |
| `hf_triton_attention.py` | HuggingFace `attn_implementation="modelopt_triton"` backend for LLMs. |

### Calibration (`modelopt/torch/sparsity/attention_sparsity/calibration/`)

| File | Role |
|------|------|
| `calibrate.py` | Orchestrates calibration. Skips RULER dataset when user provides `forward_loop` (diffusion models). Applies fitted (a, b) to all modules. |
| `calibrator.py` | `DynamicThresholdCalibrator`: collects (scale_factor, sparsity) pairs via Triton calibration kernel, fits exponential model `scale_factor = a * exp(b * sparsity)`. |

### Config & Conversion

| File | Role |
|------|------|
| `config.py` | `SparseAttentionAttributeConfig` with `skip_softmax_threshold`, `skip_softmax_raw_threshold`, calibration settings. |
| `conversion.py` | `_register_diffusers_backends_if_needed()` auto-registers Triton backends on `sparsify()`. |
| `sparse_attention.py` | `SparseAttentionModule` wrapper — delegates to method's `get_sparse_context()`. |

## Threshold Modes

| Mode | How threshold reaches the kernel | Use case |
|------|----------------------------------|----------|
| **Raw threshold** (`--raw-threshold -0.7`) | Passed directly as `skip_threshold_log2` — no conversion | Quick testing, sweeps |
| **Calibrated** (`--calibrate --target-sparsity 0.5`) | `scale_factor = a * exp(b * target)`, then backend computes `threshold = scale_factor / seq_k`, then kernel converts `log2(threshold) * sm_scale` | Production use with automatic seqlen adaptation |
| **Static lambda** (default `skip_softmax_threshold=0.1`) | `log2(lambda) * sm_scale` | Fallback when neither raw nor calibrated |

## Known Issues

- **Calibration sparsity ratio**: The calibrated threshold goes through `log2(threshold) * sm_scale` conversion, producing `skip_threshold_log2` values in a different scale than raw thresholds. Needs investigation to ensure the fitted (a, b) parameters produce expected sparsity levels.
- **14B dual transformer calibration**: Transformers are calibrated sequentially — transformer_2's calibration runs while transformer_1 is already sparsified, introducing asymmetric calibration conditions.
