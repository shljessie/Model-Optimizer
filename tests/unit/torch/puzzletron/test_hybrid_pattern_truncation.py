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

"""Tests for hybrid_override_pattern handling via ModelDescriptor.truncate_pattern_for_subblock.

Covers the base descriptor method that selects the correct pattern
character when calculate_subblock_params builds a 1-layer model.
End-to-end validation with the real model is in
tests/gpu/puzzletron/test_nemotron_h_gpu_validation.py.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor

NEMOTRON_H_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"

descriptor = ModelDescriptor


def _make_config(pattern=NEMOTRON_H_PATTERN):
    return SimpleNamespace(hybrid_override_pattern=pattern)


class TestTruncatePatternForSubblock:
    """Test truncate_pattern_for_subblock with parent_layer_index lookups."""

    def test_index_selects_mamba(self):
        cfg = _make_config()
        descriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=0)
        assert cfg.hybrid_override_pattern == "M"

    def test_index_selects_ffn(self):
        cfg = _make_config()
        descriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=1)
        assert cfg.hybrid_override_pattern == "-"

    def test_index_selects_attention(self):
        cfg = _make_config()
        descriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=7)
        assert cfg.hybrid_override_pattern == "*"

    def test_pipe_separator_stripped(self):
        """Pipe-delimited patterns are normalised before index lookup."""
        cfg = _make_config("M|-|*")
        descriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=1)
        assert cfg.hybrid_override_pattern == "-"

    def test_pipe_index_after_stripping(self):
        """Index maps to the stripped pattern, not the raw string."""
        cfg = _make_config("M|-|*")
        descriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=2)
        assert cfg.hybrid_override_pattern == "*"

    def test_no_index_falls_back_to_first_char(self):
        """Without parent_layer_index the method falls back to pattern[0]."""
        cfg = _make_config()
        descriptor.truncate_pattern_for_subblock(cfg)
        assert cfg.hybrid_override_pattern == "M"

    def test_out_of_range_falls_back_to_first_char(self):
        cfg = _make_config("M-*")
        descriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=999)
        assert cfg.hybrid_override_pattern == "M"

    def test_no_pattern_is_noop(self):
        """Config without hybrid_override_pattern should be left unchanged."""
        cfg = SimpleNamespace()
        descriptor.truncate_pattern_for_subblock(cfg)
        assert not hasattr(cfg, "hybrid_override_pattern")

    def test_empty_pattern_is_noop(self):
        """Empty pattern string should be left unchanged."""
        cfg = _make_config("")
        descriptor.truncate_pattern_for_subblock(cfg)
        assert cfg.hybrid_override_pattern == ""
