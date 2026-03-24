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

"""Utility functions for bypass distillation."""

from pathlib import Path

from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist


def set_experiment_id(cfg: DictConfig) -> None:
    """Set the experiment ID based on the model config overrides."""
    if cfg.bypass.experiment_id is None:
        overrides = cfg.bypass.model.model_config_overrides
        if "ffn" in overrides:
            ffn_override = overrides.ffn[0]
            if "intermediate_size" in ffn_override:
                # Dense FFN model: identify by FFN size and attention heads
                cfg.bypass.experiment_id = "bypass_ffn_{}_heads_{}".format(
                    ffn_override["intermediate_size"],
                    overrides.attention[0]["num_key_value_heads"],
                )
            else:
                # MoE model: identify by number of experts per layer
                cfg.bypass.experiment_id = "bypass_experts_{}".format(
                    ffn_override["moe"]["num_local_experts"]
                )
        elif "attention" in overrides:
            # Attention-only bypass: identify by number of KV heads
            cfg.bypass.experiment_id = "bypass_heads_{}".format(
                overrides.attention[0]["num_key_value_heads"]
            )


def set_experiment_dir(cfg: DictConfig) -> None:
    """Set the experiment directory for the bypass run."""
    cfg.bypass.experiment_dir = Path(cfg.puzzle_dir) / "bypass" / "bypass_runs" / cfg.bypass.experiment_id
    if dist.is_master():
        cfg.bypass.experiment_dir.mkdir(parents=True, exist_ok=True)


def get_distributed_modules_ownership(module_count: int, world_size: int) -> list[int]:
    """Map module (block) indices to GPU ranks for pipeline-parallel distribution."""
    modules_process_ownership: list[int] = []

    for i in range(world_size):
        num_modules_for_process = module_count // world_size
        if i < module_count % world_size:
            num_modules_for_process += 1

        modules_process_ownership.extend([i] * num_modules_for_process)

    return modules_process_ownership
