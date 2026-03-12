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

"""Predefined LoRA configurations for common Megatron-Core model architectures.

These configs are designed to be passed directly to
:func:`modelopt.torch.peft.update_model` or used as the ``--lora-config``
argument in the PTQ script.

Config name strings map to entries in :data:`LORA_CFG_CHOICES`.
"""

import torch.nn.init as init

__all__ = ["DENSE_LORA_CFG", "LORA_CFG_CHOICES", "MOE_LORA_CFG"]

# ---------------------------------------------------------------------------
# Dense (non-MoE) model config
# ---------------------------------------------------------------------------
# Targets the four linear projections that are standard in every transformer
# decoder layer:
#   - linear_qkv  : fused Q/K/V projection  (ColumnParallelLinear)
#   - linear_proj : attention output projection (RowParallelLinear)
#   - linear_fc1  : MLP gate/up projection  (ColumnParallelLinear)
#   - linear_fc2  : MLP down projection     (RowParallelLinear)
#
# All other modules are excluded via the wildcard ``"*": {"enable": False}``
# fallback (later patterns override earlier ones).
DENSE_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_cfg": {
        "*": {"enable": False},
        "*linear_qkv*": {"rank": 64, "enable": True},
        "*linear_proj*": {"rank": 64, "enable": True},
        "*linear_fc1*": {"rank": 64, "enable": True},
        "*linear_fc2*": {"rank": 64, "enable": True},
    },
}

# ---------------------------------------------------------------------------
# MoE model config
# ---------------------------------------------------------------------------
# Targets attention projections, dense MLP layers, and expert blocks.
#
# Expert FFN blocks in Megatron-Core are implemented as ``SequentialMLP``
# modules (matched by ``*experts*``).  The ``_LoRAMegatronSequentialMLP``
# plugin handles the shared-down / per-expert-up adapter layout internally,
# so the adapter is applied to the container, not the layers inside it.
#
# Dense MLP layers that co-exist with expert layers (e.g. the first/last
# layers in some architectures) use ``*mlp.linear_fc1*`` / ``*mlp.linear_fc2*``
# rather than ``*linear_fc1*`` / ``*linear_fc2*``.  This is intentional:
# the expert inner paths look like ``...mlp.experts.local_experts.0.linear_fc1``
# which does NOT contain ``mlp.linear_fc1`` as a substring, so these more
# specific patterns only match the dense MLP layers and not the individual
# linear layers inside expert blocks.
MOE_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_cfg": {
        "*": {"enable": False},
        "*mlp.experts*": {"rank": 64, "enable": True},
        "*linear_fc1*": {"enable": False},
        "*linear_fc2*": {"enable": False},
    },
}

MOE_LORA_RANDOM_INIT_CFG = {
    "adapter_type": "lora",
    "adapter_cfg": {
        "*": {"enable": False},
        "*mlp.experts*": {
            "rank": 64,
            "enable": True,
            "scale": 1,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
        },
        "*linear_fc1*": {"enable": False},
        "*linear_fc2*": {"enable": False},
    },
}
# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
LORA_CFG_CHOICES: dict[str, dict] = {
    "dense": DENSE_LORA_CFG,
    "moe": MOE_LORA_CFG,
}
