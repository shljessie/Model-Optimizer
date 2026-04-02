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

"""Unit tests for bypass_utils helpers and _set_keys_to_learn."""

import types

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.bypass_distillation.bypass_utils import (
    get_distributed_modules_ownership,
    set_experiment_id,
)
from modelopt.torch.puzzletron.bypass_distillation.stitched_model_factory import _set_keys_to_learn


def test_single_gpu_all_to_rank_0():
    """With world_size=1, all 4 modules should be assigned to rank 0."""
    ownership = get_distributed_modules_ownership(module_count=4, world_size=1)
    assert ownership == [0, 0, 0, 0]


def test_even_distribution():
    """With world_size=2 and 4 modules, each rank should own exactly 2 modules."""
    ownership = get_distributed_modules_ownership(module_count=4, world_size=2)
    assert ownership.count(0) == 2
    assert ownership.count(1) == 2
    assert len(ownership) == 4


def test_uneven_distribution():
    """With world_size=2 and 3 modules, rank 0 should own 2 and rank 1 should own 1."""
    ownership = get_distributed_modules_ownership(module_count=3, world_size=2)
    assert ownership.count(0) == 2
    assert ownership.count(1) == 1
    assert len(ownership) == 3


@pytest.mark.parametrize(
    ("module_count", "world_size"),
    [
        (1, 1),
        (4, 1),
        (4, 2),
        (4, 4),
        (7, 3),
        (10, 4),
        (1, 2),
    ],
)
def test_total_equals_module_count(module_count, world_size):
    """The length of the ownership list must always equal module_count."""
    ownership = get_distributed_modules_ownership(module_count=module_count, world_size=world_size)
    assert len(ownership) == module_count


def test_consecutive_ownership():
    """Each rank should own a contiguous block of indices (no interleaving)."""
    ownership = get_distributed_modules_ownership(module_count=7, world_size=3)
    # Verify that once we see a new rank, we never see the previous rank again.
    seen_ranks = set()
    prev_rank = ownership[0]
    seen_ranks.add(prev_rank)
    for rank in ownership[1:]:
        if rank != prev_rank:
            assert rank not in seen_ranks, (
                f"Rank {rank} appears non-consecutively in ownership list: {ownership}"
            )
            seen_ranks.add(rank)
            prev_rank = rank


def test_single_module():
    """With world_size=2 and only 1 module, rank 0 should be the sole owner."""
    ownership = get_distributed_modules_ownership(module_count=1, world_size=2)
    assert ownership == [0]
    assert len(ownership) == 1


# ---------------------------------------------------------------------------
# Helpers for _set_keys_to_learn tests
# ---------------------------------------------------------------------------


def _make_flat_model(*param_names: str) -> nn.Module:
    """Return a flat module whose named_parameters() yields exactly the given names.

    Parameter names must not contain dots (use underscores instead).
    All parameters are float32 and start with requires_grad=False.
    """
    model = nn.Module()
    model.config = types.SimpleNamespace()
    for name in param_names:
        assert "." not in name, f"Use underscores, not dots, in flat model param names: {name}"
        model.register_parameter(name, nn.Parameter(torch.randn(4), requires_grad=False))
    return model


class _FakeLMConfig:
    def __init__(self, num_hidden_layers, block_configs=None):
        self.num_hidden_layers = num_hidden_layers
        self.block_configs = block_configs


class _FakeDescriptor:
    """Minimal descriptor stub for _set_keys_to_learn tests."""

    def __init__(self, lm_config, weight_groups):
        self._lm_config = lm_config
        self._weight_groups = weight_groups

    def get_language_model_config(self, model_config):
        return self._lm_config

    def get_weight_groups(self, state_dict_keys, num_hidden_layers):
        return self._weight_groups


# ---------------------------------------------------------------------------
# _set_keys_to_learn tests
# ---------------------------------------------------------------------------


def test_set_keys_to_learn_sequence():
    """Passing a list of parameter names enables grad only on those params."""
    model = _make_flat_model("weight_a", "weight_b", "weight_c")
    _set_keys_to_learn(model, descriptor=None, keys_to_learn=["weight_a", "weight_c"])

    assert model.get_parameter("weight_a").requires_grad is True
    assert model.get_parameter("weight_b").requires_grad is False
    assert model.get_parameter("weight_c").requires_grad is True


def test_set_keys_to_learn_regex():
    """A bare regex string selects parameters by re.search."""
    model = _make_flat_model("block_0_ffn_weight", "block_0_attn_weight", "block_1_ffn_weight")
    _set_keys_to_learn(model, descriptor=None, keys_to_learn=r"_ffn_")

    assert model.get_parameter("block_0_ffn_weight").requires_grad is True
    assert model.get_parameter("block_0_attn_weight").requires_grad is False
    assert model.get_parameter("block_1_ffn_weight").requires_grad is True


def test_set_keys_to_learn_no_match_is_noop():
    """A regex that matches nothing should not raise and leave all params unchanged."""
    model = _make_flat_model("weight_a", "weight_b")
    _set_keys_to_learn(model, descriptor=None, keys_to_learn=r"NONEXISTENT_PATTERN_XYZ")

    assert model.get_parameter("weight_a").requires_grad is False
    assert model.get_parameter("weight_b").requires_grad is False


def test_set_keys_to_learn_subblock_ffn():
    """'subblock_ffn' should enable only params in _ffn weight groups."""
    # Names use underscores throughout (no dots) so register_parameter accepts them.
    model = _make_flat_model("block_0_ffn_w1", "block_0_ffn_w2", "block_0_attn_q", "block_0_attn_k")
    weight_groups = {
        "block_0_ffn": ["block_0_ffn_w1", "block_0_ffn_w2"],
        "block_0_attention": ["block_0_attn_q", "block_0_attn_k"],
    }
    lm_config = _FakeLMConfig(num_hidden_layers=1)
    descriptor = _FakeDescriptor(lm_config, weight_groups)

    _set_keys_to_learn(model, descriptor=descriptor, keys_to_learn="subblock_ffn")

    assert model.get_parameter("block_0_ffn_w1").requires_grad is True
    assert model.get_parameter("block_0_ffn_w2").requires_grad is True
    assert model.get_parameter("block_0_attn_q").requires_grad is False
    assert model.get_parameter("block_0_attn_k").requires_grad is False


def test_set_keys_to_learn_subblock_attention():
    """'subblock_attention' should enable only params in _attention weight groups."""
    model = _make_flat_model("block_0_ffn_w1", "block_0_attn_q", "block_0_attn_k")
    weight_groups = {
        "block_0_ffn": ["block_0_ffn_w1"],
        "block_0_attention": ["block_0_attn_q", "block_0_attn_k"],
    }
    lm_config = _FakeLMConfig(num_hidden_layers=1)
    descriptor = _FakeDescriptor(lm_config, weight_groups)

    _set_keys_to_learn(model, descriptor=descriptor, keys_to_learn="subblock_attention")

    assert model.get_parameter("block_0_ffn_w1").requires_grad is False
    assert model.get_parameter("block_0_attn_q").requires_grad is True
    assert model.get_parameter("block_0_attn_k").requires_grad is True


def test_set_keys_to_learn_entire_block():
    """'entire_block' should enable all attention and ffn params."""
    model = _make_flat_model("block_0_ffn_w1", "block_0_attn_q")
    weight_groups = {
        "block_0_ffn": ["block_0_ffn_w1"],
        "block_0_attention": ["block_0_attn_q"],
    }
    lm_config = _FakeLMConfig(num_hidden_layers=1)
    descriptor = _FakeDescriptor(lm_config, weight_groups)

    _set_keys_to_learn(model, descriptor=descriptor, keys_to_learn="entire_block")

    assert model.get_parameter("block_0_ffn_w1").requires_grad is True
    assert model.get_parameter("block_0_attn_q").requires_grad is True


def test_set_keys_to_learn_hybrid_mamba_filtering():
    """For hybrid models, subblock_attention skips Mamba blocks and vice-versa."""
    model = _make_flat_model("block_0_attn_q", "block_1_attn_ssm")
    weight_groups = {
        "block_0_attention": ["block_0_attn_q"],
        "block_1_attention": ["block_1_attn_ssm"],
    }

    # block_configs: block_0 is GQA (mamba=None), block_1 is Mamba (mamba != None)
    block_cfg_0 = types.SimpleNamespace(attention=types.SimpleNamespace(mamba=None))
    block_cfg_1 = types.SimpleNamespace(attention=types.SimpleNamespace(mamba=object()))

    lm_config = _FakeLMConfig(num_hidden_layers=2, block_configs=[block_cfg_0, block_cfg_1])
    descriptor = _FakeDescriptor(lm_config, weight_groups)

    # subblock_attention: only GQA (block_0), not Mamba (block_1)
    _set_keys_to_learn(model, descriptor=descriptor, keys_to_learn="subblock_attention")
    assert model.get_parameter("block_0_attn_q").requires_grad is True
    assert model.get_parameter("block_1_attn_ssm").requires_grad is False

    # Reset and test subblock_mamba: only Mamba (block_1), not GQA (block_0)
    for p in model.parameters():
        p.requires_grad_(False)
    _set_keys_to_learn(model, descriptor=descriptor, keys_to_learn="subblock_mamba")
    assert model.get_parameter("block_0_attn_q").requires_grad is False
    assert model.get_parameter("block_1_attn_ssm").requires_grad is True


# ---------------------------------------------------------------------------
# set_experiment_id tests
# ---------------------------------------------------------------------------


def _make_exp_cfg(overrides: dict, keys_to_learn: str = "entire_block", experiment_id=None):
    """Build a minimal DictConfig that set_experiment_id can operate on."""
    return OmegaConf.create(
        {
            "bypass": {
                "experiment_id": experiment_id,
                "model": {"model_config_overrides": overrides},
                "model_factory": {"keys_to_learn": keys_to_learn},
            }
        }
    )


def test_exp_id_ffn_only():
    """FFN intermediate_size change → bypass_ffn{size}."""
    cfg = _make_exp_cfg({"ffn": [{"intermediate_size": 256}]})
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_ffn256"


def test_exp_id_attention_only():
    """KV-head change only (FFN None) → bypass_kv{n}."""
    cfg = _make_exp_cfg(
        {
            "ffn": [{"intermediate_size": None}],
            "attention": [{"num_key_value_heads": 4}],
        }
    )
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_kv4"


def test_exp_id_ffn_and_attention():
    """Combined FFN + attention change → bypass_ffn{size}_kv{n}."""
    cfg = _make_exp_cfg(
        {
            "ffn": [{"intermediate_size": 256}],
            "attention": [{"num_key_value_heads": 4}],
        }
    )
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_ffn256_kv4"


def test_exp_id_moe():
    """MoE expert-count change → bypass_experts{n}."""
    cfg = _make_exp_cfg(
        {
            "ffn": [{"moe": {"num_local_experts": 4}}],
        }
    )
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_experts4"


def test_exp_id_mamba_with_state_dim():
    """Mamba state_dim change → bypass_mambastate{dim}."""
    cfg = _make_exp_cfg(
        {
            "attention": [{"mamba": {"state_dim": 64}}],
        }
    )
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_mambastate64"


def test_exp_id_mamba_no_structural_change():
    """Mamba bypass with no structural override → fallback to keys_to_learn type."""
    cfg = _make_exp_cfg(
        overrides={"attention": [{"num_key_value_heads": None}]},
        keys_to_learn="subblock_mamba",
    )
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_mamba"


def test_exp_id_fallback_keys_to_learn_variants():
    """No non-None overrides → experiment_id from keys_to_learn."""
    cases = [
        ("subblock_ffn", "bypass_ffn"),
        ("subblock_attention", "bypass_attn"),
        ("subblock_mamba", "bypass_mamba"),
        ("entire_block", "bypass_block"),
    ]
    for keys_to_learn, expected in cases:
        cfg = _make_exp_cfg(overrides={}, keys_to_learn=keys_to_learn)
        set_experiment_id(cfg)
        assert cfg.bypass.experiment_id == expected, (
            f"keys_to_learn={keys_to_learn!r}: expected {expected!r}, "
            f"got {cfg.bypass.experiment_id!r}"
        )


def test_exp_id_per_layer_uniform():
    """All layers same size → single value (no dash separator)."""
    cfg = _make_exp_cfg({"ffn": [{"intermediate_size": 256}] * 4})
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_ffn256"


def test_exp_id_per_layer_mixed():
    """Different per-layer sizes → values joined with dash."""
    cfg = _make_exp_cfg(
        {
            "ffn": [
                {"intermediate_size": 256},
                {"intermediate_size": 3072},
                {"intermediate_size": 256},
            ]
        }
    )
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "bypass_ffn256-3072"


def test_exp_id_already_set_is_noop():
    """experiment_id already populated → set_experiment_id is a no-op."""
    cfg = _make_exp_cfg(
        overrides={"ffn": [{"intermediate_size": 256}]},
        experiment_id="my_custom_id",
    )
    set_experiment_id(cfg)
    assert cfg.bypass.experiment_id == "my_custom_id"


def test_exp_id_none_fields_not_included():
    """None-valued fields do not contribute to the experiment ID."""
    cfg = _make_exp_cfg(
        {
            "ffn": [{"intermediate_size": None, "moe": None}],
            "attention": [{"num_key_value_heads": 4}],
        }
    )
    set_experiment_id(cfg)
    # Only the kv component should appear
    assert cfg.bypass.experiment_id == "bypass_kv4"
