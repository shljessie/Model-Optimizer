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

"""Test ModelOpt GPTQ with PSX LUTS VQ on a single expert linear layer."""

import copy

import pytest
import torch

RTN_CFG_NAME = (
    "PSX_LUTS_WEIGHT_VL_VS8_Entries65536_LFSR_max_sorted_bs16_ACTIVATION_NONE_CFG_routed_moes"
)
GPTQ_CFG_NAME = (
    "GPTQ_PSX_LUTS_WEIGHT_VL_VS8_Entries65536_LFSR_max_sorted_bs16_ACTIVATION_NONE_CFG_routed_moes"
)


def _configs_available():
    try:
        import modelopt.torch.quantization as mtq

        return getattr(mtq, RTN_CFG_NAME, None) is not None
    except Exception:
        return False


class _SingleExpertModel(torch.nn.Module):
    """Wraps a Linear so its path contains 'experts' to match the quant_cfg patterns."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.experts = torch.nn.ModuleList([torch.nn.Linear(in_features, out_features, bias=False)])

    def forward(self, x):
        return self.experts[0](x)


@pytest.mark.skipif(not _configs_available(), reason="PSX LUTS plugin configs not available")
def test_modelopt_gptq_vs_rtn():
    """GPTQ should produce lower output NMSE than RTN on a single expert layer."""
    import modelopt.torch.quantization as mtq

    rtn_cfg = copy.deepcopy(getattr(mtq, RTN_CFG_NAME))
    gptq_cfg = copy.deepcopy(getattr(mtq, GPTQ_CFG_NAME))
    # Single-layer model has no decoder layers for sequential calibration
    gptq_cfg["algorithm"]["use_sequential"] = False

    torch.manual_seed(42)
    out_features, in_features, n_samples = 64, 256, 128

    model = _SingleExpertModel(in_features, out_features).cuda().float()
    orig_weight = model.experts[0].weight.data.clone()
    calib_data = [torch.randn(1, in_features, device="cuda") for _ in range(n_samples)]

    def forward_loop(m):
        for x in calib_data:
            m(x)

    # RTN (fold weights so we get the actual QDQ'd values)
    rtn_model = mtq.quantize(copy.deepcopy(model), rtn_cfg, forward_loop=forward_loop)
    mtq.fold_weight(rtn_model)
    rtn_weight = rtn_model.experts[0].weight.data.float()

    # GPTQ
    gptq_model = mtq.quantize(copy.deepcopy(model), gptq_cfg, forward_loop=forward_loop)
    gptq_weight = gptq_model.experts[0].weight.data.float()

    # Output NMSE
    act = torch.cat(calib_data, dim=0).squeeze().T  # (in_features, n_samples)
    w = orig_weight.float()
    ref_norm_sq = (w @ act).norm() ** 2
    nmse_rtn = ((rtn_weight - w) @ act).norm() ** 2 / ref_norm_sq
    nmse_gptq = ((gptq_weight - w) @ act).norm() ** 2 / ref_norm_sq

    print(f"\nRTN  NMSE: {nmse_rtn:.8f}")
    print(f"GPTQ NMSE: {nmse_gptq:.8f}")
    print(f"GPTQ gain over RTN: {nmse_rtn / nmse_gptq:.4f}x")

    assert nmse_gptq < nmse_rtn, "GPTQ should beat RTN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
