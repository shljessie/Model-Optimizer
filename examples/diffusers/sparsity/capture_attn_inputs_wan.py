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

"""Capture attention Q/K/V inputs from Wan 2.2 5B model.

Saves per-step .pt files to experiments/attn_input_wan22/.
Each file: list of dicts with {q, k, v, heads, layer_idx} in bf16.

Wan 2.2 5B has 40 transformer blocks, each with attn1 (self-attention)
and attn2 (cross-attention). Only self-attention (attn1) is captured.

Q/K/V are captured after QKNorm and RoPE, with shape
[batch, seq_len, heads, head_dim] where heads=40, head_dim=128 for 5B.

Usage::
    CUDA_VISIBLE_DEVICES=0 python capture_attn_inputs_wan.py
    CUDA_VISIBLE_DEVICES=0 python capture_attn_inputs_wan.py --save-steps 0 19 39 49
    CUDA_VISIBLE_DEVICES=0 python capture_attn_inputs_wan.py --save-layers 0 10 20 30 39
"""

import argparse
import os
import threading

import torch

# ---- Capture state ----
_capture_lock = threading.Lock()
_capture_state = {
    "enabled": False,
    "step": -1,
    "layer_counter": 0,
    "data": [],
}

SAVE_DIR = os.path.join(
    os.path.dirname(__file__), "experiments", "attn_input_wan22"
)  # overridden by --save-dir

# Controlled by CLI args
_save_steps = None  # set of step indices to save (None = all)
_save_layers = None  # set of layer indices to save (None = all)


def _enable_capture(step: int):
    with _capture_lock:
        _capture_state["enabled"] = _save_steps is None or step in _save_steps
        _capture_state["step"] = step
        _capture_state["layer_counter"] = 0
        _capture_state["data"] = []


def _disable_capture_and_save():
    with _capture_lock:
        if not _capture_state["enabled"]:
            return
        step = _capture_state["step"]
        data = _capture_state["data"]
        _capture_state["enabled"] = False
        _capture_state["data"] = []

    if not data:
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_data = []
    for entry in data:
        save_data.append(
            {
                "q": entry["q"].to(torch.bfloat16).cpu(),
                "k": entry["k"].to(torch.bfloat16).cpu(),
                "v": entry["v"].to(torch.bfloat16).cpu(),
                "heads": entry["heads"],
                "layer_idx": entry["layer_idx"],
            }
        )
    path = os.path.join(SAVE_DIR, f"step_{step:03d}.pt")
    torch.save(save_data, path)
    print(f"  Saved {len(save_data)} attention inputs -> {path}")


def _capture_hook(q, k, v, heads, layer_idx):
    with _capture_lock:
        if not _capture_state["enabled"]:
            return
        if _save_layers is not None and layer_idx not in _save_layers:
            return
        _capture_state["data"].append(
            {
                "q": q.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "heads": heads,
                "layer_idx": layer_idx,
            }
        )


def _patch_wan_attention_for_capture():
    """Patch WanAttnProcessor to capture Q/K/V from self-attention layers.

    Wraps the original __call__. For self-attention (encoder_hidden_states=None),
    re-computes Q/K/V (cheap linear + norm + RoPE) to capture them, then delegates
    to the original processor for the actual attention computation.
    """
    from diffusers.models.transformers.transformer_wan import (
        WanAttnProcessor,
        _get_qkv_projections,
    )

    original_call = WanAttnProcessor.__call__

    def patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, rotary_emb=None):
        # Only capture self-attention (encoder_hidden_states is None for attn1)
        is_self_attn = encoder_hidden_states is None

        if is_self_attn:
            with _capture_lock:
                enabled = _capture_state["enabled"]
                if enabled:
                    layer_idx = _capture_state["layer_counter"]
                    _capture_state["layer_counter"] += 1

            if enabled:
                # Re-derive Q/K/V after QKNorm + RoPE (same as original processor)
                # This is cheap: just linear projections + norm + rotary, no attention
                with torch.no_grad():
                    query, key, value = _get_qkv_projections(attn, hidden_states, None)
                    query = attn.norm_q(query)
                    key = attn.norm_k(key)
                    query = query.unflatten(2, (attn.heads, -1))
                    key = key.unflatten(2, (attn.heads, -1))
                    value = value.unflatten(2, (attn.heads, -1))

                    if rotary_emb is not None:
                        def apply_rotary_emb(x, freqs_cos, freqs_sin):
                            x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                            cos = freqs_cos[..., 0::2]
                            sin = freqs_sin[..., 1::2]
                            out = torch.empty_like(x)
                            out[..., 0::2] = x1 * cos - x2 * sin
                            out[..., 1::2] = x1 * sin + x2 * cos
                            return out.type_as(x)

                        query = apply_rotary_emb(query, *rotary_emb)
                        key = apply_rotary_emb(key, *rotary_emb)

                    _capture_hook(query, key, value, attn.heads, layer_idx)

        # Delegate to original processor for the actual computation
        return original_call(self, attn, hidden_states, encoder_hidden_states,
                             attention_mask, rotary_emb)

    WanAttnProcessor.__call__ = patched_call
    print("Patched WanAttnProcessor.__call__ for Q/K/V capture on self-attention layers.")


def main():
    global _save_steps, _save_layers, SAVE_DIR

    parser = argparse.ArgumentParser(description="Capture attention inputs from Wan 2.2 5B")
    parser.add_argument("--num-steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--num-frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--height", type=int, default=704, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely "
        "on a spotlighted stage.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=(
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        ),
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=[0, 9, 19, 29, 39, 49],
        help="Which denoising steps to save (default: 0 9 19 29 39 49)",
    )
    parser.add_argument(
        "--save-layers",
        type=int,
        nargs="+",
        default=None,
        help="Which layer indices to save (default: all). Wan 2.2 5B has 40 self-attn layers.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override save directory (default: experiments/attn_input_wan22)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="HuggingFace model ID for Wan 2.2",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    args = parser.parse_args()

    _save_steps = set(args.save_steps)
    _save_layers = set(args.save_layers) if args.save_layers else None
    if args.save_dir is not None:
        SAVE_DIR = args.save_dir

    print(f"Loading Wan 2.2 pipeline from {args.model_id}...")

    from diffusers import AutoencoderKLWan, WanPipeline

    dtype = torch.bfloat16
    device = "cuda"

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model_id, vae=vae, torch_dtype=dtype)
    pipe.to(device)

    transformer = pipe.transformer
    num_blocks = len(transformer.blocks)
    print(f"Wan 2.2 transformer has {num_blocks} blocks (each with attn1 self-attention)")

    # Patch the attention processor to capture Q/K/V
    _patch_wan_attention_for_capture()

    # Track denoising steps via transformer.forward
    original_forward = transformer.forward
    step_counter = {"value": 0}
    num_steps = args.num_steps

    def tracked_forward(*fwd_args, **fwd_kwargs):
        step = step_counter["value"]
        print(f"Step {step}/{num_steps}...")
        _enable_capture(step)
        result = original_forward(*fwd_args, **fwd_kwargs)
        _disable_capture_and_save()
        step_counter["value"] += 1
        return result

    transformer.forward = tracked_forward

    steps_str = ", ".join(str(s) for s in sorted(_save_steps))
    layers_str = "all" if _save_layers is None else ", ".join(str(l) for l in sorted(_save_layers))
    print(f"Capturing: steps=[{steps_str}], layers=[{layers_str}]")
    print(f"Save dir: {SAVE_DIR}")
    print(f"Resolution: {args.height}x{args.width}, frames: {args.num_frames}")
    print(f"Guidance scale: {args.guidance_scale}")

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
    ).frames[0]

    print(f"\nDone! Captured {step_counter['value']} steps.")
    print(f"Saved steps: {sorted(_save_steps)}")
    print(f"Output dir: {SAVE_DIR}")

    # Optionally save the generated video
    try:
        from diffusers.utils import export_to_video

        video_path = os.path.join(SAVE_DIR, "generated_video.mp4")
        os.makedirs(SAVE_DIR, exist_ok=True)
        export_to_video(output, video_path, fps=16)
        print(f"Saved generated video -> {video_path}")
    except Exception as e:
        print(f"Could not save video: {e}")


if __name__ == "__main__":
    main()
