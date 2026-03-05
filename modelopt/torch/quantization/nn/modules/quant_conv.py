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

"""Quantized convolution."""

import logging

import torch.nn as nn

from ... import tensor_quant
from .quant_module import QuantLinearConvBase, QuantModuleRegistry, _LegacyQuantLinearConvBaseMixin

logger = logging.getLogger(__name__)

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "QuantConv1d",
    "QuantConv2d",
    "QuantConv3d",
    "QuantConvTranspose1d",
    "QuantConvTranspose2d",
    "QuantConvTranspose3d",
]


@QuantModuleRegistry.register({nn.Conv1d: "nn.Conv1d"})
class _QuantConv1d(QuantLinearConvBase):
    """Quantized 1D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL


class QuantConv1d(_LegacyQuantLinearConvBaseMixin, nn.Conv1d):
    """Quantized 1D convolution."""

    default_quant_desc_weight = _QuantConv1d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.Conv2d: "nn.Conv2d"})
class _QuantConv2d(QuantLinearConvBase):
    """Quantized 2D convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL


class QuantConv2d(_LegacyQuantLinearConvBaseMixin, nn.Conv2d):
    """Quantized 2D convolution."""

    default_quant_desc_weight = _QuantConv2d.default_quant_desc_weight


def _is_nvfp4_quantizer(quantizer) -> bool:
    """Check if a TensorQuantizer is configured for NVFP4 dynamic block quantization."""
    return (
        hasattr(quantizer, "_num_bits")
        and quantizer._num_bits == (2, 1)
        and hasattr(quantizer, "_block_sizes")
        and quantizer._block_sizes is not None
        and quantizer._block_sizes.get("scale_bits") == (4, 3)
        and quantizer._block_sizes.get("type") == "dynamic"
    )


def _nvfp4_quantize_weight_along_k(weight, weight_quantizer):
    """Apply NVFP4 fake quantization to Conv3D weight along the GEMM K dimension.

    The implicit GEMM maps K = Cin * kD * kH * kW. The default quantizer would
    quantize along the last dim (kW), which is wrong. We reshape to [Cout, K]
    so blocks are along the contraction dimension.
    """
    cout = weight.shape[0]
    k = weight[0].numel()  # Cin * kD * kH * kW
    w_flat = weight.reshape(cout, k)
    w_q_flat = weight_quantizer(w_flat)
    return w_q_flat.reshape_as(weight)


@QuantModuleRegistry.register({nn.Conv3d: "nn.Conv3d"})
class _QuantConv3d(QuantLinearConvBase):
    """Quantized 3D convolution.

    When both input and weight quantizers are configured for NVFP4, the forward
    uses the fused implicit GEMM kernel from experimental/conv which performs
    activation FP4 quantization inside the kernel. The weight is FP4-quantized
    along the GEMM K dimension (Cin*kD*kH*kW) before being passed to the kernel.

    For all other quantization configs, the default cuDNN path is used.
    """

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL

    def _should_use_implicit_gemm(self):
        """Check if both quantizers are NVFP4 and the implicit GEMM kernel is available."""
        if not (
            hasattr(self, "input_quantizer")
            and hasattr(self, "weight_quantizer")
            and _is_nvfp4_quantizer(self.input_quantizer)
            and _is_nvfp4_quantizer(self.weight_quantizer)
        ):
            return False
        try:
            from experimental.conv.implicit_gemm_cuda import conv3d_implicit_gemm_cuda  # noqa: F401

            return True
        except ImportError:
            return False

    def forward(self, input, *args, **kwargs):
        """Forward with implicit GEMM for NVFP4, default path otherwise."""
        if not self._should_use_implicit_gemm():
            return super().forward(input, *args, **kwargs)

        # During calibration we only need to collect amax — use the faster
        # default cuDNN path since the conv output itself doesn't matter.
        if self.input_quantizer._if_calib and not self.input_quantizer._if_quant:
            return super().forward(input, *args, **kwargs)

        from experimental.conv.implicit_gemm_cuda import conv3d_implicit_gemm_cuda

        # --- Get activation amax for the kernel ---
        act_amax = self.input_quantizer._get_amax(input)

        # --- Quantize weight along K dimension ---
        weight = _nvfp4_quantize_weight_along_k(self.weight, self.weight_quantizer)

        # --- Get fp4_block_size from the input quantizer config ---
        fp4_block_size = self.input_quantizer._block_sizes.get(-1, 16)

        # --- Call implicit GEMM kernel ---
        output = conv3d_implicit_gemm_cuda(
            input,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            act_amax=act_amax,
            quant_act=not self.input_quantizer._disabled,
            fp4_block_size=fp4_block_size,
        )

        # --- Output quantizer (usually disabled for NVFP4) ---
        if hasattr(self, "output_quantizer"):
            output = self.output_quantizer(output)

        return output


class QuantConv3d(_LegacyQuantLinearConvBaseMixin, nn.Conv3d):
    """Quantized 3D convolution."""

    default_quant_desc_weight = _QuantConv3d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.ConvTranspose1d: "nn.ConvTranspose1d"})
class _QuantConvTranspose1d(QuantLinearConvBase):
    """Quantized 1D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL


class QuantConvTranspose1d(_LegacyQuantLinearConvBaseMixin, nn.ConvTranspose1d):
    """Quantized 1D transposed convolution."""

    default_quant_desc_weight = _QuantConvTranspose1d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.ConvTranspose2d: "nn.ConvTranspose2d"})
class _QuantConvTranspose2d(QuantLinearConvBase):
    """Quantized 2D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL


class QuantConvTranspose2d(_LegacyQuantLinearConvBaseMixin, nn.ConvTranspose2d):
    """Quantized 2D transposed convolution."""

    default_quant_desc_weight = _QuantConvTranspose2d.default_quant_desc_weight


@QuantModuleRegistry.register({nn.ConvTranspose3d: "nn.ConvTranspose3d"})
class _QuantConvTranspose3d(QuantLinearConvBase):
    """Quantized 3D transposed convolution."""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL


class QuantConvTranspose3d(_LegacyQuantLinearConvBaseMixin, nn.ConvTranspose3d):
    """Quantized 3D transposed convolution."""

    default_quant_desc_weight = _QuantConvTranspose3d.default_quant_desc_weight


# Define alias with Quant prefix
Conv1d = QuantConv1d
Conv2d = QuantConv2d
Conv3d = QuantConv3d
ConvTranspose1d = QuantConvTranspose1d
ConvTranspose2d = QuantConvTranspose2d
ConvTranspose3d = QuantConvTranspose3d
