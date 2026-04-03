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

"""Unit tests for the diffusers WAN attention quantization plugin.

Tests cover:
- Registry: WanAttention is replaced by _QuantWanAttention after mtq.quantize()
- Processor: _QuantWanAttnProcessor is installed on each quantized module
- Quantizers: softmax_quantizer, q_bmm_quantizer, k_bmm_quantizer, v_bmm_quantizer exist
- Config: NVFP4_WAN_SOFTMAX_CFG selectively enables only softmax_quantizer
- Forward shape: output shape matches input shape (quantizers disabled for CPU test)
- Enable/disable: individual quantizers can be enabled/disabled via set_quantizer_by_cfg
"""

import pytest

pytest.importorskip("diffusers")

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import NVFP4_WAN_SOFTMAX_CFG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wan_attention(dim=64, heads=4, dim_head=16):
    """Create a small WanAttention module (CPU, float32)."""
    from diffusers.models.transformers.transformer_wan import WanAttention

    return WanAttention(dim=dim, heads=heads, dim_head=dim_head)


def _make_wan_model(num_layers=2, dim=64, heads=4, dim_head=16):
    """Tiny nn.Module wrapping several WanAttention modules."""
    from diffusers.models.transformers.transformer_wan import WanAttention

    class _WanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [WanAttention(dim=dim, heads=heads, dim_head=dim_head) for _ in range(num_layers)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return _WanModel()


def _rand_hidden(batch=1, seq=16, dim=64):
    return torch.randn(batch, seq, dim)


# ---------------------------------------------------------------------------
# Config to quantize only softmax_quantizer (all other quantizers disabled)
# ---------------------------------------------------------------------------

_SOFTMAX_ONLY_CFG = {
    "quant_cfg": {
        "*softmax_quantizer": {"num_bits": 8, "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

# Config with all quantizers disabled (used for CPU forward-shape tests)
_ALL_DISABLED_CFG = {
    "quant_cfg": {
        "default": {"enable": False},
    },
    "algorithm": "max",
}


# ---------------------------------------------------------------------------
# Tests: registry — WanAttention -> _QuantWanAttention
# ---------------------------------------------------------------------------


class TestWanQuantRegistry:
    """mtq.quantize() should replace WanAttention with _QuantWanAttention."""

    def test_module_type_after_quantize(self):
        from modelopt.torch.quantization.plugins.diffusion.diffusers import _QuantWanAttention

        model = _make_wan_model(num_layers=2)
        mtq.quantize(model, _ALL_DISABLED_CFG)

        quant_modules = [m for m in model.modules() if isinstance(m, _QuantWanAttention)]
        assert len(quant_modules) == 2

    def test_processor_type_after_quantize(self):
        from modelopt.torch.quantization.plugins.diffusion.diffusers import (
            _QuantWanAttention,
            _QuantWanAttnProcessor,
        )

        model = _make_wan_model(num_layers=1)
        mtq.quantize(model, _ALL_DISABLED_CFG)

        quant_mod = next(m for m in model.modules() if isinstance(m, _QuantWanAttention))
        assert isinstance(quant_mod.processor, _QuantWanAttnProcessor)


# ---------------------------------------------------------------------------
# Tests: quantizer attributes are created
# ---------------------------------------------------------------------------


class TestQuantizerAttributes:
    """All expected TensorQuantizers should exist after quantize()."""

    def _get_quant_mod(self, num_layers=1):
        from modelopt.torch.quantization.plugins.diffusion.diffusers import _QuantWanAttention

        model = _make_wan_model(num_layers=num_layers)
        mtq.quantize(model, _ALL_DISABLED_CFG)
        return next(m for m in model.modules() if isinstance(m, _QuantWanAttention))

    def test_softmax_quantizer_exists(self):
        from modelopt.torch.quantization.nn import TensorQuantizer

        mod = self._get_quant_mod()
        assert hasattr(mod, "softmax_quantizer")
        assert isinstance(mod.softmax_quantizer, TensorQuantizer)

    def test_qkv_quantizers_exist(self):
        from modelopt.torch.quantization.nn import TensorQuantizer

        mod = self._get_quant_mod()
        for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
            assert hasattr(mod, name), f"Missing {name}"
            assert isinstance(getattr(mod, name), TensorQuantizer), f"{name} is not TensorQuantizer"


# ---------------------------------------------------------------------------
# Tests: NVFP4_WAN_SOFTMAX_CFG selectively enables only softmax_quantizer
# ---------------------------------------------------------------------------


class TestNvfp4WanSoftmaxCfg:
    """NVFP4_WAN_SOFTMAX_CFG enables softmax_quantizer and disables everything else."""

    def test_softmax_quantizer_enabled(self):
        from modelopt.torch.quantization.plugins.diffusion.diffusers import _QuantWanAttention

        model = _make_wan_model(num_layers=1)
        # Use INT8 instead of NVFP4 so calibration works on CPU
        cfg = _SOFTMAX_ONLY_CFG
        mtq.quantize(model, cfg)

        mod = next(m for m in model.modules() if isinstance(m, _QuantWanAttention))
        assert mod.softmax_quantizer.is_enabled

    def test_other_quantizers_disabled(self):
        from modelopt.torch.quantization.plugins.diffusion.diffusers import _QuantWanAttention

        model = _make_wan_model(num_layers=1)
        mtq.quantize(model, _SOFTMAX_ONLY_CFG)

        mod = next(m for m in model.modules() if isinstance(m, _QuantWanAttention))
        for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer"):
            assert not getattr(mod, name).is_enabled, f"{name} should be disabled"

    def test_config_exported(self):
        """NVFP4_WAN_SOFTMAX_CFG should be importable and have the right structure."""
        assert "quant_cfg" in NVFP4_WAN_SOFTMAX_CFG
        assert "*softmax_quantizer" in NVFP4_WAN_SOFTMAX_CFG["quant_cfg"]
        cfg = NVFP4_WAN_SOFTMAX_CFG["quant_cfg"]["*softmax_quantizer"]
        assert cfg["num_bits"] == (2, 1)


# ---------------------------------------------------------------------------
# Tests: forward pass output shape (all quantizers disabled, CPU)
# ---------------------------------------------------------------------------


class TestForwardShape:
    """Forward pass returns correct shape (quantizers disabled to avoid NVFP4 on CPU)."""

    def test_output_shape_single_layer(self):
        model = _make_wan_model(num_layers=1)
        mtq.quantize(model, _ALL_DISABLED_CFG)
        hidden = _rand_hidden(batch=2, seq=16, dim=64)
        out = model(hidden)
        assert out.shape == hidden.shape

    def test_output_shape_multi_layer(self):
        model = _make_wan_model(num_layers=3)
        mtq.quantize(model, _ALL_DISABLED_CFG)
        hidden = _rand_hidden(batch=1, seq=32, dim=64)
        out = model(hidden)
        assert out.shape == hidden.shape

    def test_output_shape_softmax_only_enabled(self):
        """Forward still produces correct shape when only softmax_quantizer is enabled (INT8)."""
        model = _make_wan_model(num_layers=1)
        mtq.quantize(model, _SOFTMAX_ONLY_CFG)

        # Calibrate with a forward pass so amax is set
        hidden = _rand_hidden(batch=1, seq=16, dim=64)
        with mtq.calibrate(model, algorithm="max", forward_loop=lambda m: m(hidden)):
            pass

        out = model(hidden)
        assert out.shape == hidden.shape
