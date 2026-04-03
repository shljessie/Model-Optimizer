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

"""Support quantization of diffusers layers."""

from collections.abc import Callable, Iterator
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING

import diffusers
import onnx
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from packaging.version import parse as parse_version

if parse_version(diffusers.__version__) >= parse_version("0.35.0"):
    from diffusers.models.attention import AttentionModuleMixin
    from diffusers.models.attention_dispatch import AttentionBackendName, attention_backend
    from diffusers.models.transformers.transformer_flux import FluxAttention
    from diffusers.models.transformers.transformer_ltx import LTXAttention
    from diffusers.models.transformers.transformer_wan import WanAttention

    try:
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2Attention,
            Flux2ParallelSelfAttention,
        )
    except ImportError:
        Flux2Attention = None
        Flux2ParallelSelfAttention = None
else:
    AttentionModuleMixin = type("_dummy_type_no_instance", (), {})  # pylint: disable=invalid-name
from torch.autograd import Function
from torch.nn import functional as F
from torch.onnx import symbolic_helper

if TYPE_CHECKING:
    if hasattr(torch.onnx._internal, "jit_utils"):
        from torch.onnx._internal.jit_utils import GraphContext
    else:  # torch >= 2.9
        from torch.onnx._internal.torchscript_exporter.jit_utils import GraphContext

from ...export_onnx import export_fp8_mha
from ...nn import (
    QuantConv2d,
    QuantInputBase,
    QuantLinear,
    QuantLinearConvBase,
    QuantModuleRegistry,
    TensorQuantizer,
)
from ..custom import _QuantFunctionalMixin

onnx_dtype_map = {
    "BFloat16": onnx.TensorProto.BFLOAT16,
    "Float": onnx.TensorProto.FLOAT,
    "Float8": onnx.TensorProto.FLOAT8E4M3FN,
    "Half": onnx.TensorProto.FLOAT16,
    "INT8": onnx.TensorProto.INT8,
    "UINT8": onnx.TensorProto.UINT8,
}
mha_valid_precisions = {"Half", "BFloat16"}


class _QuantLoRACompatibleLinearConvBase(QuantLinearConvBase):
    def _setup(self):
        assert self.lora_layer is None, (
            f"To quantize {self}, lora_layer should be None. Please fuse the LoRA layer before"
            " quantization."
        )
        return super()._setup()


@QuantModuleRegistry.register({LoRACompatibleConv: "LoRACompatibleConv"})
class _QuantLoRACompatibleConv(_QuantLoRACompatibleLinearConvBase):
    default_quant_desc_weight = QuantConv2d.default_quant_desc_weight


@QuantModuleRegistry.register({LoRACompatibleLinear: "LoRACompatibleLinear"})
class _QuantLoRACompatibleLinear(_QuantLoRACompatibleLinearConvBase):
    default_quant_desc_weight = QuantLinear.default_quant_desc_weight


def _quantized_bmm(self, input, mat2, *args, **kwargs):
    attn, v = input, mat2
    return self.bmm2_output_quantizer(
        torch._bmm(self.softmax_quantizer(attn), self.v_bmm_quantizer(v), *args, **kwargs)
    )


def _quantized_baddbmm(self, input, batch1, batch2, *args, **kwargs):
    q, k = batch1, batch2
    return torch._baddbmm(input, self.q_bmm_quantizer(q), self.k_bmm_quantizer(k), *args, **kwargs)


def _quantized_sdpa(self, *args, **kwargs):
    fp8_sdpa = FP8SDPA.apply
    parameters = [
        "query",
        "key",
        "value",
        "attn_mask",
        "dropout_p",
        "is_causal",
        "scale",
        "q_quantized_scale",
        "k_quantized_scale",
        "v_quantized_scale",
        "high_precision_flag",
    ]
    default_values = [None, None, None, None, 0.0, False, None, None, None, None, "Half"]
    param_dict = dict(zip(parameters, default_values))
    for i, arg in enumerate(args):
        param_dict[parameters[i]] = arg
    param_dict.update(kwargs)
    fp8_sdpa_args = [param_dict[param] for param in parameters]
    while fp8_sdpa_args and fp8_sdpa_args[-1] is None:
        fp8_sdpa_args.pop()
    query, key, value = fp8_sdpa_args[:3]

    if not torch.onnx.is_in_onnx_export():
        query = self.q_bmm_quantizer(query)
        key = self.k_bmm_quantizer(key)
        value = self.v_bmm_quantizer(value)

    q_quantized_scale = self.q_bmm_quantizer._get_amax(query)
    k_quantized_scale = self.k_bmm_quantizer._get_amax(key)
    v_quantized_scale = self.v_bmm_quantizer._get_amax(value)

    # We don't need to calibrate the output of softmax
    return self.bmm2_output_quantizer(
        fp8_sdpa(
            query,
            key,
            value,
            *fp8_sdpa_args[3:7],
            q_quantized_scale,
            k_quantized_scale,
            v_quantized_scale,
            self.q_bmm_quantizer.trt_high_precision_dtype
            if hasattr(self.q_bmm_quantizer, "trt_high_precision_dtype")
            else "Half",
            self._disable_fp8_mha if hasattr(self, "_disable_fp8_mha") else True,
        )
    )


class _QuantAttention(_QuantFunctionalMixin):
    """Quantized processor for performing attention-related computations."""

    _functionals_to_replace = [
        (torch, "bmm", _quantized_bmm),
        (torch, "baddbmm", _quantized_baddbmm),
        (F, "scaled_dot_product_attention", _quantized_sdpa),
    ]

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        for package, func_name, quantized_func in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(quantized_func, self)
            yield package, func_name, quantized_func

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.softmax_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.bmm2_output_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)


QuantModuleRegistry.register({Attention: "Attention"})(_QuantAttention)


if AttentionModuleMixin.__module__.startswith(diffusers.__name__):

    class _QuantAttentionModuleMixin(_QuantAttention):
        """Quantized AttentionModuleMixin for performing attention-related computations."""

        def forward(self, *args, **kwargs):
            with attention_backend(AttentionBackendName.NATIVE):
                return super().forward(*args, **kwargs)

    QuantModuleRegistry.register({FluxAttention: "FluxAttention"})(_QuantAttentionModuleMixin)
    QuantModuleRegistry.register({LTXAttention: "LTXAttention"})(_QuantAttentionModuleMixin)
    if Flux2Attention is not None:
        QuantModuleRegistry.register({Flux2Attention: "Flux2Attention"})(_QuantAttentionModuleMixin)
    if Flux2ParallelSelfAttention is not None:
        QuantModuleRegistry.register({Flux2ParallelSelfAttention: "Flux2ParallelSelfAttention"})(
            _QuantAttentionModuleMixin
        )

    def _apply_rotary_emb_wan(
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embeddings to WAN attention tensors."""
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden_states)

    class _QuantWanAttnProcessor:
        """WAN attention processor that applies TensorQuantizer to the P-matrix.

        Performs manual Q@K^T -> softmax -> softmax_quantizer -> @V so that the
        post-softmax attention weights (P-matrix) can be quantized with any
        TensorQuantizer configuration (e.g. NVFP4).

        Also honours q_bmm_quantizer, k_bmm_quantizer, and v_bmm_quantizer
        from the parent _QuantWanAttention module.
        """

        # Kept for diffusers compatibility checks.
        _attention_backend = None

        @staticmethod
        def _prepare_qkv(
            attn: "WanAttention",
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None,
            rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Project, norm, reshape, and apply rotary embeddings."""
            enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states

            if attn.fused_projections:
                if attn.cross_attention_dim_head is None:
                    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
                else:
                    query = attn.to_q(hidden_states)
                    key, value = attn.to_kv(enc).chunk(2, dim=-1)
            else:
                query = attn.to_q(hidden_states)
                key = attn.to_k(enc)
                value = attn.to_v(enc)

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            # Reshape to BSND: [B, S, heads, head_dim]
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            if rotary_emb is not None:
                query = _apply_rotary_emb_wan(query, *rotary_emb)
                key = _apply_rotary_emb_wan(key, *rotary_emb)

            return query, key, value

        # Row-chunk size to avoid materialising the full S×T float32 matrix OOM
        # on long video sequences (WAN2.2 can exceed 8k tokens).
        _CHUNK = 512

        def _attention(
            self,
            attn_module: "WanAttention",
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            """Quantized attention: Q@K^T -> softmax -> softmax_quantizer -> @V.

            Processes query rows in chunks of ``_CHUNK`` to avoid materialising the
            full S×T attention matrix in float32 (which OOMs for long video sequences).

            Args:
                attn_module: The (quantized) WanAttention module holding quantizers.
                query: ``[B, S, H, D]``
                key:   ``[B, T, H, D]``
                value: ``[B, T, H, D]``
                attention_mask: Optional additive mask broadcastable to ``[B, S, H, T]``.

            Returns:
                Output ``[B, S, H*D]``.
            """
            head_dim = query.shape[-1]
            scale = head_dim**-0.5

            # Apply Q/K/V quantizers once before the chunked loop.
            q = attn_module.q_bmm_quantizer(query)
            k = attn_module.k_bmm_quantizer(key)
            v = attn_module.v_bmm_quantizer(value)

            seq_q = q.shape[1]
            out_chunks = []

            for start in range(0, seq_q, self._CHUNK):
                end = min(start + self._CHUNK, seq_q)
                q_chunk = q[:, start:end]  # [B, chunk, H, D]

                # [B, chunk, H, D] x [B, T, H, D] -> [B, chunk, H, T]
                w_chunk = torch.einsum("bshd,bthd->bsht", q_chunk, k) * scale

                if attention_mask is not None:
                    # mask shape: [B, 1, S, T] or [B, H, S, T] — slice along S dim
                    w_chunk = w_chunk + attention_mask[..., start:end, :]

                w_chunk = F.softmax(w_chunk, dim=-1, dtype=torch.float32).to(query.dtype)

                # Quantize the P-matrix chunk (post-softmax attention weights).
                w_chunk = attn_module.softmax_quantizer(w_chunk)

                # [B, chunk, H, T] x [B, T, H, D] -> [B, chunk, H, D]
                out_chunks.append(torch.einsum("bsht,bthd->bshd", w_chunk, v))

            out = torch.cat(out_chunks, dim=1)  # [B, S, H, D]
            return out.flatten(2, 3).type_as(query)

        def __call__(
            self,
            attn: "WanAttention",
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> torch.Tensor:
            """Forward pass — replaces WanAttnProcessor.__call__."""
            encoder_hidden_states_img = None
            if attn.add_k_proj is not None:
                # I2V: split image and text context (512 is WAN's hardcoded text length).
                assert encoder_hidden_states is not None
                image_context_length = encoder_hidden_states.shape[1] - 512
                encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
                encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

            query, key, value = self._prepare_qkv(
                attn, hidden_states, encoder_hidden_states, rotary_emb
            )

            # I2V image cross-attention.
            hidden_states_img = None
            if encoder_hidden_states_img is not None:
                if attn.fused_projections:
                    key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(
                        2, dim=-1
                    )
                else:
                    key_img = attn.add_k_proj(encoder_hidden_states_img)
                    value_img = attn.add_v_proj(encoder_hidden_states_img)
                key_img = attn.norm_added_k(key_img)
                key_img = key_img.unflatten(2, (attn.heads, -1))
                value_img = value_img.unflatten(2, (attn.heads, -1))
                hidden_states_img = self._attention(attn, query, key_img, value_img, None)

            hidden_states = self._attention(attn, query, key, value, attention_mask)

            if hidden_states_img is not None:
                hidden_states = hidden_states + hidden_states_img

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    class _QuantWanAttention(_QuantAttentionModuleMixin):
        """Quantized WanAttention with P-matrix (post-softmax) quantization.

        Installs ``_QuantWanAttnProcessor`` which performs manual attention
        computation so that ``softmax_quantizer`` is applied to the attention
        weights between softmax and the value matmul.  The standard
        ``q_bmm_quantizer``, ``k_bmm_quantizer``, ``v_bmm_quantizer``, and
        ``softmax_quantizer`` are all created by the parent ``_QuantAttention._setup()``.
        """

        def _setup(self) -> None:
            """Set up quantizers and install the quantized attention processor."""
            super()._setup()
            self.set_processor(_QuantWanAttnProcessor())

        def forward(self, *args, **kwargs):
            """Forward without function-patching context (processor handles quantization)."""
            # Skip _QuantFunctionalMixin's torch.bmm/F.sdpa patching — the
            # _QuantWanAttnProcessor calls quantizers directly, so no context needed.
            with attention_backend(AttentionBackendName.NATIVE):
                # Jump past _QuantFunctionalMixin.forward() to WanAttention.forward()
                return super(_QuantAttentionModuleMixin, self).forward(*args, **kwargs)

    QuantModuleRegistry.register({WanAttention: "WanAttention"})(_QuantWanAttention)


original_scaled_dot_product_attention = F.scaled_dot_product_attention


class FP8SDPA(Function):
    """A customized FP8 SDPA op for the onnx export."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        q_quantized_scale=None,
        k_quantized_scale=None,
        v_quantized_scale=None,
        high_precision_flag=None,
        disable_fp8_mha=True,
    ):
        """Forward method."""
        ctx.save_for_backward(query, key, value, attn_mask)
        ctx.q_quantized_scale = q_quantized_scale
        ctx.k_quantized_scale = k_quantized_scale
        ctx.v_quantized_scale = v_quantized_scale
        # During runtime, ignore x or use it as needed
        return original_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v", "t", "t", "t", "s", "b")
    def symbolic(
        g: "GraphContext",
        query: "torch._C.Value",
        key: "torch._C.Value",
        value: "torch._C.Value",
        attn_mask: "torch._C.Value | None" = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: "torch._C.Value | None" = None,
        q_quantized_scale: float = 1.0,
        k_quantized_scale: float = 1.0,
        v_quantized_scale: float = 1.0,
        high_precision_flag: str = "Half",
        disable_fp8_mha: bool = True,
    ):
        """Symbolic method."""
        return export_fp8_mha(
            g,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            q_quantized_scale,
            k_quantized_scale,
            v_quantized_scale,
            high_precision_flag,
            disable_fp8_mha,
        )
