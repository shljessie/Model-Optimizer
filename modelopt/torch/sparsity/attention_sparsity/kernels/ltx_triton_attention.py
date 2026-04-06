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

"""Triton flash attention wrapper for LTX-2 (ltx_core) skip-softmax sparse attention.

Patches ``Attention`` modules from ``ltx_core`` so that when the Triton
skip-softmax flag is active, attention is computed via the Triton FA kernel
with fused tile skipping and gap/log(seq_k) normalization.

Used during **inference** — calibration uses the eager wrapper instead.
"""

import math
import threading

import torch
from ltx_core.model.transformer.attention import Attention

from modelopt.torch.kernels.triton_fa import attention

# Thread-local storage for skip-softmax configuration
_thread_local = threading.local()


def set_ltx_triton_context(
    active: bool,
    threshold: float | None = None,
    normalize_by_seqlen: bool = False,
    enable_v25: bool = False,
    enable_v32: bool = False,
    majority_pct: float = 0.0,
    lite_threshold: float | None = None,
    sparsity_counters: "torch.Tensor | None" = None,
) -> None:
    """Set thread-local Triton config for LTX-2 attention.

    Called per-module per-forward. Lite attention state (_lite_layer_masks,
    _lite_step, _lite_call_idx) is initialized only once and preserved
    across calls — step_boundary() manages the step counter.
    """
    _thread_local.active = active
    _thread_local.threshold = threshold
    _thread_local.normalize_by_seqlen = normalize_by_seqlen
    _thread_local.enable_v25 = enable_v25
    _thread_local.enable_v32 = enable_v32
    _thread_local.majority_pct = majority_pct
    _thread_local.lite_threshold = lite_threshold
    _thread_local.sparsity_counters = sparsity_counters
    if not enable_v25:
        _thread_local.v_mean_cache = None
    if not enable_v32:
        _thread_local.k_mean_cache = None
    # Initialize lite state only if not already set
    if lite_threshold is not None and not hasattr(_thread_local, "_lite_layer_masks"):
        _thread_local._lite_layer_masks = {}
        _thread_local._lite_call_idx = 0
        _thread_local._lite_step = -1


def clear_ltx_triton_context() -> None:
    """Clear thread-local Triton config.

    Note: lite attention state (_lite_layer_masks, _lite_step, _lite_call_idx)
    is NOT reset here because this is called after each module's forward.
    Lite state persists across modules and denoising steps by design.
    """
    _thread_local.active = False
    _thread_local.threshold = None
    _thread_local.normalize_by_seqlen = False
    _thread_local.enable_v25 = False
    _thread_local.enable_v32 = False
    _thread_local.majority_pct = 0.0
    _thread_local.sparsity_counters = None
    _thread_local.v_mean_cache = None
    _thread_local.k_mean_cache = None
    # lite_threshold is preserved — set/cleared by set_ltx_triton_context only
    # _lite_layer_masks, _lite_step, _lite_call_idx persist across calls


def _get_ltx_triton_context() -> tuple[bool, float | None, bool, bool]:
    """Return (active, threshold, normalize_by_seqlen, enable_v25)."""
    return (
        getattr(_thread_local, "active", False),
        getattr(_thread_local, "threshold", None),
        getattr(_thread_local, "normalize_by_seqlen", False),
        getattr(_thread_local, "enable_v25", False),
    )


# ---------------------------------------------------------------------------
# LiteAttention simulation config (LTX-2 path)
# ---------------------------------------------------------------------------
def lite_attention_set_num_layers(num_layers: int) -> None:
    """Set number of enabled attention layers per transformer forward.

    Call once after mtsa.sparsify() so the dispatch code can compute
    layer_idx = call_idx % num_layers (sharing masks across CFG passes).
    """
    _thread_local._lite_num_layers = num_layers


def lite_attention_step_boundary() -> None:
    """Notify a transformer forward is about to start.

    Increments the forward counter used for denoising step detection.
    """
    _thread_local._lite_fwd_count = getattr(_thread_local, "_lite_fwd_count", 0) + 1


def lite_attention_get_sparsity() -> float | None:
    """Return average sparsity (fraction of tiles skipped) across all layers."""
    masks = getattr(_thread_local, "_lite_layer_masks", {})
    if not masks:
        return None
    step = getattr(_thread_local, "_lite_step", -1)
    total_skip = 0
    total_tiles = 0
    for _layer_idx, (mask_a, mask_b) in masks.items():
        mask = mask_a if step % 2 == 0 else mask_b
        total_skip += (mask == 1).sum().item()
        total_tiles += mask.numel()
    return total_skip / total_tiles if total_tiles > 0 else 0.0


def _ltx_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    threshold: float | None = None,
    normalize_by_seqlen: bool = False,
) -> torch.Tensor:
    """Triton FA attention on LTX-2 layout ``[B, T, H*D]``.

    Converts from LTX-2's fused-head layout to the Triton kernel's varlen
    format, calls the kernel with skip-softmax, and converts back.
    """
    b, seq_q, dim_total = q.shape
    dim_head = dim_total // heads
    seq_k = k.shape[1]
    device = q.device

    # LTX-2 layout: [B, T, H*D] → reshape to [B, T, H, D] → flat [B*T, H, D]
    q_flat = q.view(b, seq_q, heads, dim_head).reshape(b * seq_q, heads, dim_head).contiguous()
    k_flat = k.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()
    v_flat = v.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()

    # Build varlen metadata
    b_start_loc_q = torch.arange(b, device=device, dtype=torch.int32) * seq_q
    b_seq_len_q = torch.full((b,), seq_q, device=device, dtype=torch.int32)

    scale = 1.0 / math.sqrt(dim_head)

    kw: dict = {
        "b_start_loc": b_start_loc_q,
        "b_seq_len": b_seq_len_q,
        "max_input_len": seq_q,
        "is_causal": False,  # Diffusion uses bidirectional attention
        "softmax_scale": scale,
    }

    # Handle different Q/KV sequence lengths
    if seq_q != seq_k:
        b_start_loc_k = torch.arange(b, device=device, dtype=torch.int32) * seq_k
        b_seq_len_k = torch.full((b,), seq_k, device=device, dtype=torch.int32)
        kw["b_start_loc_k"] = b_start_loc_k
        kw["b_seq_len_k"] = b_seq_len_k
        kw["max_input_len_k"] = seq_k

    # LiteAttention simulation (takes priority over skip-softmax)
    lite_threshold = getattr(_thread_local, "lite_threshold", None)
    if lite_threshold is not None:
        BLOCK_M, BLOCK_N = 128, 64
        n_qt = math.ceil(seq_q / BLOCK_M)
        n_kt = math.ceil(seq_k / BLOCK_N)
        mask_shape = (b, heads, n_qt, n_kt)

        # With CFG, transformer.forward() is called N times per denoising step.
        # Each forward runs L enabled layers. All CFG passes at the same layer
        # share the same mask: layer_idx = call_idx % L. The last CFG pass's
        # write overwrites earlier ones — this is fine for simulation.
        # Step advances every L calls (each CFG pass is treated as a step for
        # mask write, but reads from the previous step's write).
        call_idx = getattr(_thread_local, "_lite_call_idx", 0)
        step = getattr(_thread_local, "_lite_step", -1)
        num_layers = getattr(_thread_local, "_lite_num_layers", 0)

        if step == -1:
            step = 0
            call_idx = 0
            _thread_local._lite_step = 0

        # Layer index wraps within num_layers
        if num_layers > 0:
            layer_idx = call_idx % num_layers
            # Advance step when a full layer cycle completes
            if call_idx > 0 and layer_idx == 0:
                step += 1
                _thread_local._lite_step = step
        else:
            layer_idx = call_idx

        _thread_local._lite_call_idx = call_idx + 1

        layer_masks = getattr(_thread_local, "_lite_layer_masks", {})

        if layer_idx not in layer_masks:
            layer_masks[layer_idx] = (
                torch.zeros(mask_shape, device=device, dtype=torch.int8),
                torch.zeros(mask_shape, device=device, dtype=torch.int8),
            )
            _thread_local._lite_layer_masks = layer_masks

        mask_a, mask_b = layer_masks[layer_idx]

        kw["lite_attention_threshold"] = lite_threshold
        if step == 0:
            kw["lite_attention_skip_write"] = mask_a
        else:
            if step % 2 == 1:
                kw["lite_attention_skip_read"] = mask_a
                kw["lite_attention_skip_write"] = mask_b
            else:
                kw["lite_attention_skip_read"] = mask_b
                kw["lite_attention_skip_write"] = mask_a

    elif threshold is not None and threshold > 0.0:
        # Skip-softmax threshold
        kw["skip_softmax_threshold"] = threshold
        kw["skip_softmax_normalize_by_seqlen"] = normalize_by_seqlen

        # V2.5: lazy-allocate and pass caches
        if getattr(_thread_local, "enable_v25", False):
            import triton

            BLOCK_N = 64
            n_kt = math.ceil(seq_k / BLOCK_N)
            BLOCK_D = triton.next_power_of_2(dim_head)

            vm = getattr(_thread_local, "v_mean_cache", None)
            if vm is None or vm.shape != (b, heads, n_kt, BLOCK_D):
                _thread_local.v_mean_cache = torch.zeros(
                    b, heads, n_kt, BLOCK_D, device=device, dtype=torch.float32
                )

            kw["v_mean_cache"] = _thread_local.v_mean_cache

        # V3.2: lazy-allocate k_mean_cache alongside v_mean_cache
        if getattr(_thread_local, "enable_v32", False):
            import triton

            BLOCK_N = 64
            n_kt = math.ceil(seq_k / BLOCK_N)
            BLOCK_D = triton.next_power_of_2(dim_head)

            km = getattr(_thread_local, "k_mean_cache", None)
            if km is None or km.shape != (b, heads, n_kt, BLOCK_D):
                _thread_local.k_mean_cache = torch.zeros(
                    b, heads, n_kt, BLOCK_D, device=device, dtype=torch.float32
                )

            kw["k_mean_cache"] = _thread_local.k_mean_cache

        # V3: majority vote
        majority_pct = getattr(_thread_local, "majority_pct", 0.0)
        if majority_pct > 0:
            kw["majority_pct"] = majority_pct

        # Runtime sparsity measurement
        counters = getattr(_thread_local, "sparsity_counters", None)
        if counters is not None:
            kw["sparsity_counters"] = counters

    o = attention(q_flat, k_flat, v_flat, **kw)

    # Reshape back: [B*T, H, D] → [B, T, H*D]
    return o.view(b, seq_q, heads * dim_head)


class _TritonLTXAttentionWrapper:
    """Wraps an ``attention_function`` callable from ltx_core.

    When the thread-local Triton skip-softmax flag is active, routes to the
    Triton FA kernel.  Otherwise calls the original function.
    """

    def __init__(self, original_fn):
        self._original_fn = original_fn

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active, threshold, normalize_by_seqlen, _ = _get_ltx_triton_context()
        if active:
            return _ltx_triton_attention(q, k, v, heads, mask, threshold, normalize_by_seqlen)
        return self._original_fn(q, k, v, heads, mask)


def register_ltx_triton_attention(model: torch.nn.Module) -> None:
    """Walk *model* and patch all ``ltx_core.Attention`` modules for Triton dispatch.

    Safe to call multiple times — already-wrapped modules are skipped.
    """
    for module in model.modules():
        if isinstance(module, Attention):
            fn = module.attention_function
            if not isinstance(fn, _TritonLTXAttentionWrapper):
                module.attention_function = _TritonLTXAttentionWrapper(fn)
