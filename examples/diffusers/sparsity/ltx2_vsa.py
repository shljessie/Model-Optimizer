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

"""LTX-2 inference with Video Sparse Attention (VSA).

Usage::

    # 75% sparsity (top_k_ratio=0.25 means keep 25% of blocks)
    python ltx2_vsa.py --prompt "A cat playing piano" --output vsa_75pct.mp4 \
        --top-k-ratio 0.25

    # 50% sparsity
    python ltx2_vsa.py --prompt "A cat playing piano" --output vsa_50pct.mp4 \
        --top-k-ratio 0.5

    # With first/last 2 layers excluded
    python ltx2_vsa.py --prompt "A cat playing piano" --output vsa_75pct.mp4 \
        --top-k-ratio 0.25 --skip-first-last 2
"""

import argparse
import functools

import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_AUDIO_GUIDER_PARAMS,
    DEFAULT_FRAME_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_VIDEO_GUIDER_PARAMS,
)

# Match run_experiments.py inference resolution exactly
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1280
from ltx_pipelines.utils.media_io import encode_video

import modelopt.torch.sparsity.attention_sparsity as mtsa

# ---- Model paths ----
CHECKPOINT_PATH = "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-dev.safetensors"
DISTILLED_LORA_PATH = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-19b-distilled-lora-384.safetensors"
)
SPATIAL_UPSAMPLER_PATH = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
)
GEMMA_ROOT = (
    "/home/scratch.omniml_data_2/jingyux/models/LTX-2/gemma-3-12b-it-qat-q4_0-unquantized"
)

DEFAULT_NUM_FRAMES = 121


def parse_args():
    parser = argparse.ArgumentParser(description="LTX-2 video generation with VSA")
    parser.add_argument("--prompt", type=str, default="A cat playing piano", help="Text prompt")
    parser.add_argument("--output", type=str, default="output_vsa.mp4", help="Output video path")
    parser.add_argument(
        "--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of frames"
    )
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Video height")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Video width")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.25,
        help="Ratio of blocks to keep (0.25 = 75%% sparsity)",
    )
    parser.add_argument(
        "--skip-first-last",
        type=int,
        default=2,
        help="Number of first/last transformer layers to exclude from sparsity",
    )
    return parser.parse_args()


def _patch_vae_requires_grad(pipeline):
    for ledger_attr in ("stage_1_model_ledger", "stage_2_model_ledger"):
        ledger = getattr(pipeline, ledger_attr, None)
        if ledger is None:
            continue
        for loader_name in ("video_decoder", "audio_decoder"):
            orig_loader = getattr(ledger, loader_name, None)
            if orig_loader is None:
                continue

            def _make_patched(fn):
                @functools.wraps(fn)
                def patched():
                    model = fn()
                    model.requires_grad_(False)
                    return model

                return patched

            setattr(ledger, loader_name, _make_patched(orig_loader))


def build_pipeline():
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora=[
            LoraPathStrengthAndSDOps(DISTILLED_LORA_PATH, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)
        ],
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        loras=[],
    )
    _patch_vae_requires_grad(pipeline)
    return pipeline


def build_vsa_config(args):
    """Build VSA config matching the reference pattern."""
    sparse_cfg = {
        "*.attn1": {
            "method": "vsa",
            "top_k_ratio": args.top_k_ratio,
            "block_size_3d": [4, 4, 4],
            "enable": True,
        },
        "*.attn2": {"enable": False},
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }

    # Exclude first/last N layers
    for i in range(args.skip_first_last):
        sparse_cfg[f"*.{i}.attn1"] = {"enable": False}
        sparse_cfg[f"*.{47 - i}.attn1"] = {"enable": False}

    return {"sparse_cfg": sparse_cfg}


def main():
    args = parse_args()

    print("Building LTX-2 pipeline...")
    pipeline = build_pipeline()

    # Get the stage-1 transformer
    transformer = pipeline.stage_1_model_ledger.transformer()
    pipeline.stage_1_model_ledger.transformer = lambda: transformer

    # Apply VSA
    sparsity_pct = int((1 - args.top_k_ratio) * 100)
    print(f"Applying VSA with {sparsity_pct}% sparsity (top_k_ratio={args.top_k_ratio})...")
    print(f"  skip_first_last={args.skip_first_last}")

    vsa_config = build_vsa_config(args)
    mtsa.sparsify(transformer.velocity_model, vsa_config)

    # Generate video
    tiling_config = TilingConfig.default()
    print(f"Generating: {args.prompt[:80]}...")

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=DEFAULT_FRAME_RATE,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
        audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
        images=[],
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=DEFAULT_FRAME_RATE,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output,
        video_chunks_number=get_video_chunks_number(args.num_frames, tiling_config),
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
