# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist

# ---------------------------------------------------------------------------
# Experiment ID generation
# ---------------------------------------------------------------------------

# Priority-ordered specs: (section, dot-path into first entry, tag prefix).
# For each section the *first* matching non-None field contributes one
# component; later fields in the same section are skipped.
# Add entries here to support new block types or dimension fields.
_OVERRIDE_COMPONENT_SPECS: list[tuple[str, str, str]] = [
    ("ffn", "intermediate_size", "ffn"),
    ("ffn", "moe.num_local_experts", "experts"),
    ("attention", "num_key_value_heads", "kv"),
    ("attention", "mamba.state_dim", "mambastate"),
]

# Fallback type tag when no structural change exists in model_config_overrides.
_KEYS_TO_LEARN_FALLBACK: dict[str, str] = {
    "subblock_ffn": "ffn",
    "subblock_attention": "attn",
    "subblock_mamba": "mamba",
    "entire_block": "block",
}


def _get_nested(obj: Any, dotpath: str) -> Any:
    """Return a nested value from a dict/DictConfig via dot-separated path.

    Returns ``None`` for any missing key or traversal failure so callers can
    safely treat absent and ``None``-valued fields identically.
    """
    for key in dotpath.split("."):
        if obj is None:
            return None
        try:
            obj = obj[key]
        except Exception:
            return None
    return obj


def _build_experiment_id_components(overrides: Any) -> list[str]:
    """Return ID components derived from non-None values in *overrides*.

    Each section (``ffn``, ``attention``, …) contributes at most one
    component, chosen by the first matching entry in
    ``_OVERRIDE_COMPONENT_SPECS``.  When per-layer entries hold multiple
    distinct non-None values they are listed ascending with ``-`` as
    separator (e.g. ``ffn256-3072``).
    """
    seen_sections: set[str] = set()
    components: list[str] = []

    for section, field_path, tag_prefix in _OVERRIDE_COMPONENT_SPECS:
        if section in seen_sections:
            continue
        if section not in overrides or not overrides[section]:
            continue

        values = [
            v for entry in overrides[section] if (v := _get_nested(entry, field_path)) is not None
        ]
        if not values:
            continue

        unique_vals = sorted(set(values))
        components.append(tag_prefix + "-".join(str(v) for v in unique_vals))
        seen_sections.add(section)

    return components


def set_experiment_id(cfg: DictConfig) -> None:
    """Set the experiment ID derived from model config overrides and keys_to_learn.

    The ID has the form ``bypass_{component1}_{component2}...`` where each
    component encodes one structural change:

    * ``ffn{size}``         — FFN ``intermediate_size``    (e.g. ``ffn256``)
    * ``experts{n}``        — MoE ``num_local_experts``    (e.g. ``experts4``)
    * ``kv{n}``             — Attention ``num_key_value_heads`` (e.g. ``kv4``)
    * ``mambastate{dim}``   — Mamba ``state_dim`` change

    Multiple distinct per-layer values are joined with ``-``
    (e.g. ``ffn256-3072``).  When no structural change is present (pure
    training bypass) the ``keys_to_learn`` type is used as a fallback
    (e.g. ``bypass_mamba``, ``bypass_block``).
    """
    if cfg.bypass.experiment_id is not None:
        return

    overrides = cfg.bypass.model.model_config_overrides
    components = _build_experiment_id_components(overrides)

    if not components:
        keys_to_learn = cfg.bypass.model_factory.get("keys_to_learn", "entire_block")
        fallback = (
            _KEYS_TO_LEARN_FALLBACK.get(keys_to_learn, keys_to_learn)
            if isinstance(keys_to_learn, str)
            else "block"
        )
        components = [fallback]

    cfg.bypass.experiment_id = "bypass_" + "_".join(components)


def set_experiment_dir(cfg: DictConfig) -> None:
    """Set the experiment directory for the bypass run."""
    experiment_dir = Path(cfg.puzzle_dir) / "bypass" / "bypass_runs" / cfg.bypass.experiment_id
    cfg.bypass.experiment_dir = str(experiment_dir)
    if dist.is_master():
        experiment_dir.mkdir(parents=True, exist_ok=True)


def get_distributed_modules_ownership(module_count: int, world_size: int) -> list[int]:
    """Map module (block) indices to GPU ranks for pipeline-parallel distribution."""
    modules_process_ownership: list[int] = []

    for i in range(world_size):
        num_modules_for_process = module_count // world_size
        if i < module_count % world_size:
            num_modules_for_process += 1

        modules_process_ownership.extend([i] * num_modules_for_process)

    return modules_process_ownership
