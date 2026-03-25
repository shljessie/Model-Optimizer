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

"""Capture attention Q/K/V inputs from LTX-2 baseline run.

Saves per-step .pt files to experiments/attn_input/.
Each file: list of dicts with {q, k, v, heads, layer_idx} in bf16.

Disk estimate (768×1280, 121 frames, 44 attn1 layers):
  Per step: 44 layers × 3 tensors × [4608, 32, 128] bf16 ≈ 4.7 GB
  Use --save-steps to control which steps are saved.
  Default: save steps 0,9,19,29,39 (5 steps ≈ 23 GB)

Usage::
    CUDA_VISIBLE_DEVICES=0 python capture_attn_inputs.py
    CUDA_VISIBLE_DEVICES=0 python capture_attn_inputs.py --save-steps 0 19 39
    CUDA_VISIBLE_DEVICES=0 python capture_attn_inputs.py --save-layers 5 10 23 40
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

SAVE_DIR = os.path.join(os.path.dirname(__file__), "experiments", "attn_input")

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
    print(f"  Saved {len(save_data)} attention inputs → {path}")


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


def _patch_ltx_attention_for_capture():
    from modelopt.torch.sparsity.attention_sparsity.kernels.ltx_triton_attention import (
        _TritonLTXAttentionWrapper,
    )

    original_call = _TritonLTXAttentionWrapper.__call__

    def patched_call(self, q, k, v, heads, mask=None):
        with _capture_lock:
            enabled = _capture_state["enabled"]
            if enabled:
                layer_idx = _capture_state["layer_counter"]
                _capture_state["layer_counter"] += 1

        if enabled:
            _capture_hook(q, k, v, heads, layer_idx)

        return original_call(self, q, k, v, heads, mask)

    _TritonLTXAttentionWrapper.__call__ = patched_call


def main():
    global _save_steps, _save_layers

    parser = argparse.ArgumentParser(description="Capture attention inputs from LTX-2 baseline")
    parser.add_argument("--num-steps", type=int, default=40, help="Denoising steps")
    parser.add_argument("--num-frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--prompt", type=str, default="A cat playing piano")
    parser.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=[0, 9, 19, 29, 39],
        help="Which denoising steps to save (default: 0 9 19 29 39)",
    )
    parser.add_argument(
        "--save-layers",
        type=int,
        nargs="+",
        default=None,
        help="Which layer indices to save (default: all). LTX-2 has 44 enabled attn1 layers.",
    )
    args = parser.parse_args()

    _save_steps = set(args.save_steps)
    _save_layers = set(args.save_layers) if args.save_layers else None

    from run_experiments import DEFAULT_HEIGHT, DEFAULT_WIDTH, build_pipeline
    from ltx_core.model.video_vae import TilingConfig
    from ltx_pipelines.utils.constants import (
        DEFAULT_FRAME_RATE,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_SEED,
        DEFAULT_VIDEO_GUIDER_PARAMS,
        DEFAULT_AUDIO_GUIDER_PARAMS,
    )

    import modelopt.torch.sparsity.attention_sparsity as mtsa

    print("Building LTX-2 pipeline...")
    pipeline = build_pipeline()

    transformer = pipeline.stage_1_model_ledger.transformer()
    pipeline.stage_1_model_ledger.transformer = lambda: transformer

    # Triton baseline (no sparsity) — just to get the Triton wrapper in place
    sparse_cfg = {
        "*.attn1": {
            "method": "triton_skip_softmax_diffusion",
            "br": 128,
            "bc": 128,
            "backend": "triton",
            "is_causal": False,
            "collect_stats": False,
            "enable": True,
        },
        "*.attn2": {"enable": False},
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }
    mtsa.sparsify(transformer, {"sparse_cfg": sparse_cfg})

    _patch_ltx_attention_for_capture()

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

    tiling_config = TilingConfig.default()
    pipeline(
        prompt=args.prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=DEFAULT_SEED,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_frames=args.num_frames,
        frame_rate=DEFAULT_FRAME_RATE,
        num_inference_steps=args.num_steps,
        video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
        audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
        images=[],
        tiling_config=tiling_config,
    )

    print(f"\nDone! Captured {step_counter['value']} steps to {SAVE_DIR}")


if __name__ == "__main__":
    main()
