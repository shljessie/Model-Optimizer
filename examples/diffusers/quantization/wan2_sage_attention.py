# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan2.2 Text-to-Video inference with SageAttention.

SageAttention (https://arxiv.org/pdf/2410.02367) replaces
``F.scaled_dot_product_attention`` with a faster INT8 kernel:

- Q and K are quantized to INT8 (K is smoothed via mean-subtraction first)
- V stays in FP16; the P·V product uses an FP16 accumulator
- ~2× faster attention than FlashAttention2 with negligible accuracy loss

This script patches ``F.scaled_dot_product_attention`` globally before
loading the pipeline so every attention call in the Wan2.2 transformer
goes through SageAttention automatically.

Requirements::

    pip install sageattention diffusers transformers accelerate

Usage::

    # SageAttention (default)
    python wan2_sage_attention.py --prompt "Two cats boxing on a stage"

    # Baseline — standard SDPA, no SageAttention
    python wan2_sage_attention.py --prompt "Two cats boxing on a stage" --baseline

    # Smaller 5B model (fits on a single 24 GB GPU)
    python wan2_sage_attention.py \\
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --prompt "Two cats boxing on a stage"

    # Benchmark both modes back-to-back
    python wan2_sage_attention.py --prompt "Two cats boxing on a stage" --benchmark
"""

import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

# Model IDs available on HuggingFace Hub
MODEL_T2V_14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

DEFAULT_MODEL = MODEL_T2V_14B
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text, cropped, overexposed"


# ---------------------------------------------------------------------------
# SageAttention patching
# ---------------------------------------------------------------------------

# Keep a reference to the original SDPA so we can restore it and fall back
# when SageAttention can't handle a particular call (e.g. non-None attn_mask).
_orig_sdpa = F.scaled_dot_product_attention


def _sageattn_sdpa_compat(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for ``F.scaled_dot_product_attention`` using SageAttention.

    Falls back to standard SDPA when SageAttention cannot be applied:
    - ``attn_mask`` is not None (SageAttention does not support arbitrary masks)
    - ``dropout_p > 0`` (training-time dropout)
    - Input dtype is not FP16 or BF16 (e.g. FP32 during VAE encoding/decoding)
    """
    if (
        attn_mask is not None
        or dropout_p > 0.0
        or query.dtype not in (torch.float16, torch.bfloat16)
    ):
        return _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    # sageattn uses sm_scale instead of scale; tensors are (B, H, N, D) = "HND"
    from sageattention import sageattn

    return sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)


def enable_sage_attention() -> None:
    """Patch ``F.scaled_dot_product_attention`` globally with SageAttention."""
    try:
        import sageattention  # noqa: F401
    except ImportError as e:
        raise ImportError("SageAttention is not installed. Run: pip install sageattention") from e

    F.scaled_dot_product_attention = _sageattn_sdpa_compat
    # Also patch the reference inside torch.nn.functional module object so
    # any code that imported SDPA directly still gets the patched version.
    import torch.nn.functional as _F

    _F.scaled_dot_product_attention = _sageattn_sdpa_compat
    print("[SageAttention] Patched F.scaled_dot_product_attention → sageattn")
    print("  Q/K: INT8 (K mean-smoothed), V: FP16, accumulator: FP16")


def disable_sage_attention() -> None:
    """Restore the original ``F.scaled_dot_product_attention``."""
    F.scaled_dot_product_attention = _orig_sdpa
    import torch.nn.functional as _F

    _F.scaled_dot_product_attention = _orig_sdpa


@contextmanager
def sage_attention_ctx():
    """Context manager that enables SageAttention for the duration of the block."""
    enable_sage_attention()
    try:
        yield
    finally:
        disable_sage_attention()


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def load_pipeline(model_id: str):
    """Load the Wan2.2 pipeline.

    The VAE must be loaded in FP32 to avoid numerical issues; the transformer
    and text encoder use BF16.
    """
    from diffusers import AutoencoderKLWan, WanPipeline

    print(f"[Pipeline] Loading VAE (fp32) from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

    print("[Pipeline] Loading transformer + text encoder (bf16)...")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def run_inference(pipe, args, label: str = "") -> float:
    """Run one generation pass and return wall-clock time in seconds."""
    generator = torch.Generator("cuda").manual_seed(args.seed)

    if label:
        print(f"\n[{label}] Generating {args.num_frames} frames @ {args.height}x{args.width}...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    output = pipe(
        prompt=args.prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=generator,
    ).frames[0]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    from diffusers.utils import export_to_video

    out_path = args.output if not label else args.output.replace(".mp4", f"_{label}.mp4")
    export_to_video(output, out_path, fps=16)
    print(f"[{label or 'result'}] Saved to {out_path}  ({elapsed:.1f}s)")
    return elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.2 T2V inference with SageAttention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=[MODEL_T2V_14B, MODEL_TI2V_5B],
        help="HuggingFace model ID",
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--num-steps", type=int, default=40, help="Denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run with standard SDPA instead of SageAttention",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run both baseline and SageAttention back-to-back and report speedup",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipe = load_pipeline(args.model)

    if args.benchmark:
        # Baseline pass (standard SDPA)
        t_base = run_inference(pipe, args, label="baseline")

        # SageAttention pass
        enable_sage_attention()
        t_sage = run_inference(pipe, args, label="sage")
        disable_sage_attention()

        print(f"\n{'=' * 50}")
        print(f"  Baseline:        {t_base:.1f}s")
        print(f"  SageAttention:   {t_sage:.1f}s")
        print(f"  Speedup:         {t_base / t_sage:.2f}x")
        print(f"{'=' * 50}")

    elif args.baseline:
        run_inference(pipe, args, label="baseline")

    else:
        enable_sage_attention()
        run_inference(pipe, args, label="sage")
        disable_sage_attention()


if __name__ == "__main__":
    main()
