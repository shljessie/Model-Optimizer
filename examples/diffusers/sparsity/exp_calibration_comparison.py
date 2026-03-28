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

r"""Compare LLM exponential vs diffusion percentile calibration on LTX-2.

Uses the standard mtsa.sparsify() API with two different methods:
  1. flash_skip_softmax  → DynamicThresholdCalibrator (exponential: a*exp(b*S))
  2. triton_skip_softmax_diffusion → PercentileThresholdCalibrator (gap percentile)

Both run the same calibration forward loop on the same LTX-2 model.

Usage::
    CUDA_VISIBLE_DEVICES=0 python exp_calibration_comparison.py 2>&1 | tee calib_comparison.log
    CUDA_VISIBLE_DEVICES=0 python exp_calibration_comparison.py --target-sparsity 0.5
"""

import argparse
import functools
import time

import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import (
    DEFAULT_2_STAGE_HEIGHT,
    DEFAULT_2_STAGE_WIDTH,
    DEFAULT_AUDIO_GUIDER_PARAMS,
    DEFAULT_FRAME_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_SEED,
    DEFAULT_VIDEO_GUIDER_PARAMS,
)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LLM exponential vs diffusion percentile calibration on LTX-2"
    )
    parser.add_argument(
        "--target-sparsity", type=float, default=0.2, help="Target sparsity (0.2 = 20%%)"
    )
    parser.add_argument("--calib-steps", type=int, default=35, help="Calibration denoising steps")
    parser.add_argument("--calib-frames", type=int, default=81, help="Calibration frame count")
    parser.add_argument(
        "--skip-first-last", type=int, default=2, help="First/last layers to exclude"
    )
    return parser.parse_args()


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

    # Patch VAE to disable grad
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

    return pipeline


def load_calib_prompts(calib_size=1):
    from datasets import load_dataset

    dataset = load_dataset("nkp37/OpenVid-1M")
    return list(dataset["train"]["caption"][:calib_size])


def build_calibration_forward_loop(pipeline, num_steps=10, num_frames=81):
    """Build calibration forward loop at DEFAULT_2_STAGE resolution."""
    calib_prompts = load_calib_prompts(1)
    tiling_config = TilingConfig.default()

    def forward_loop(model):
        for i, prompt in enumerate(calib_prompts):
            print(f"  Calibration [{i + 1}/{len(calib_prompts)}]: {prompt[:60]}...")
            pipeline(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                seed=DEFAULT_SEED,
                height=DEFAULT_2_STAGE_HEIGHT,
                width=DEFAULT_2_STAGE_WIDTH,
                num_frames=num_frames,
                frame_rate=DEFAULT_FRAME_RATE,
                num_inference_steps=num_steps,
                video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
                audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
                images=[],
                tiling_config=tiling_config,
            )

    return forward_loop


def make_sparse_cfg(method, target_sparsity, skip_first_last=2):
    """Build sparse config with calibration target."""
    cfg = {
        "calibration": {
            "target_sparse_ratio": {"prefill": target_sparsity},
            "threshold_trials": [
                1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,
                1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1,
                8e-1, 9e-1, 9.5e-1, 9.9e-1,
                # Extra thresholds close to 1.0 for diffusion (push sparsity higher)
                0.995, 0.999, 0.9995, 0.9999, 0.99999,
            ],
        },
        "*.attn1": {
            "method": method,
            "br": 128,
            "bc": 128,
            "backend": "triton" if "triton" in method else "pytorch",
            "is_causal": False,
            "collect_stats": True,
            "enable": True,
        },
        "*.attn2": {"enable": False},
        "*audio_attn1*": {"enable": False},
        "*audio_attn2*": {"enable": False},
        "*audio_to_video_attn*": {"enable": False},
        "*video_to_audio_attn*": {"enable": False},
        "default": {"enable": False},
    }

    for i in range(skip_first_last):
        cfg[f"*transformer_blocks.{i}.attn*"] = {"enable": False}
        cfg[f"*transformer_blocks.{47 - i}.attn*"] = {"enable": False}

    return cfg


def main():
    args = parse_args()

    print("Building LTX-2 pipeline...")
    pipeline = build_pipeline()
    transformer = pipeline.stage_1_model_ledger.transformer()
    pipeline.stage_1_model_ledger.transformer = lambda: transformer

    forward_loop = build_calibration_forward_loop(
        pipeline, num_steps=args.calib_steps, num_frames=args.calib_frames
    )

    # ---- Experiment 1: LLM Exponential (flash_skip_softmax) ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: LLM Exponential Calibration (flash_skip_softmax)")
    print("  Uses DynamicThresholdCalibrator -> scale_factor = a * exp(b * S)")
    print("=" * 70)

    sparse_cfg_llm = make_sparse_cfg(
        "flash_skip_softmax", args.target_sparsity, args.skip_first_last
    )

    t0 = time.time()
    mtsa.sparsify(transformer, {"sparse_cfg": sparse_cfg_llm}, forward_loop=forward_loop)
    t_exp = time.time() - t0
    print(f"\nExponential calibration took {t_exp:.1f}s")

    # # Un-sparsify to reset for next experiment
    # mtsa.unsparsify(transformer)

    # # ---- Experiment 2: Diffusion Percentile (triton_skip_softmax_diffusion) ----
    # print("\n" + "=" * 70)
    # print("EXPERIMENT 2: Diffusion Percentile Calibration (triton_skip_softmax_diffusion)")
    # print("  Uses PercentileThresholdCalibrator -> threshold = percentile(gaps, p)")
    # print("=" * 70)

    # sparse_cfg_diff = make_sparse_cfg(
    #     "triton_skip_softmax_diffusion", args.target_sparsity, args.skip_first_last
    # )

    # t0 = time.time()
    # mtsa.sparsify(transformer, {"sparse_cfg": sparse_cfg_diff}, forward_loop=forward_loop)
    # t_pct = time.time() - t0
    # print(f"\nPercentile calibration took {t_pct:.1f}s")

    # print("\n" + "=" * 70)
    # print("DONE -- compare the R-squared, thresholds, and calibration summaries above")
    # print("=" * 70)


if __name__ == "__main__":
    main()
