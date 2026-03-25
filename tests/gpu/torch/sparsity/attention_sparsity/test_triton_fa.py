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

"""GPU tests for Triton flash attention kernel."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.kernels import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels import attention, register_triton_attention

    if register_triton_attention is not None:
        register_triton_attention()


def _sdpa_reference(q, k, v, b_start_loc, b_seq_len):
    """SDPA causal reference. Supports GQA. Returns [total_tokens, num_heads, dim]."""
    batch = b_seq_len.shape[0]
    num_q, num_kv = q.shape[1], k.shape[1]
    parts = []
    for b in range(batch):
        s, n = int(b_start_loc[b].item()), int(b_seq_len[b].item())
        qb = q[s : s + n].unsqueeze(0).permute(0, 2, 1, 3)
        kb = k[s : s + n].unsqueeze(0).permute(0, 2, 1, 3)
        vb = v[s : s + n].unsqueeze(0).permute(0, 2, 1, 3)
        if num_q != num_kv:
            r = num_q // num_kv
            kb = kb.repeat_interleave(r, dim=1)
            vb = vb.repeat_interleave(r, dim=1)
        ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
        parts.append(ob.permute(0, 2, 1, 3).squeeze(0))
    return torch.cat(parts, dim=0)


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    """Tiny Llama: 2 layers, 64 hidden, 4 q-heads, 2 kv-heads, head_dim=16."""
    from _test_utils.torch.transformers_models import create_tiny_llama_dir

    return create_tiny_llama_dir(
        tmp_path_factory.mktemp("tiny_llama"),
        with_tokenizer=True,
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestTritonFaVsSdpa:
    """Triton flash attention matches PyTorch SDPA for prefill and decode."""

    @pytest.mark.parametrize(
        ("dtype", "num_heads", "num_kv_heads", "head_dim"),
        [
            (torch.float32, 2, 2, 32),
            (torch.float16, 4, 2, 64),
            (torch.bfloat16, 4, 2, 128),
        ],
        ids=["fp32_mha", "fp16_gqa", "bf16_gqa_hdim128"],
    )
    def test_prefill_matches_sdpa(self, dtype, num_heads, num_kv_heads, head_dim):
        """Prefill matches SDPA."""
        seq_lens = [8, 12]
        total = sum(seq_lens)
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(123)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        locs = torch.tensor([0, seq_lens[0]], device="cuda", dtype=torch.int32)
        lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

        o = attention(
            q,
            k,
            v,
            b_start_loc=locs,
            b_seq_len=lens,
            max_input_len=max(seq_lens),
            is_causal=True,
            softmax_scale=scale,
        )
        torch.testing.assert_close(o, _sdpa_reference(q, k, v, locs, lens), rtol=1e-3, atol=1e-3)

    def test_decode_matches_sdpa(self):
        """Decode matches SDPA."""
        batch = 2
        seq_lens_k = [5, 9]  # KV lengths (context + current token)
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(103)
        # Q: one token per batch element -> flat [batch, num_heads, head_dim]
        q_flat = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float32)

        # K/V: variable-length, packed into flat tensors
        total_kv = sum(seq_lens_k)
        k_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)
        v_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float32)

        cumsum = [0]
        for sl in seq_lens_k:
            cumsum.append(cumsum[-1] + sl)
        b_start_loc_q = torch.arange(batch, device="cuda", dtype=torch.int32)
        b_seq_len_q = torch.ones(batch, device="cuda", dtype=torch.int32)
        b_start_loc_k = torch.tensor(cumsum[:-1], device="cuda", dtype=torch.int32)
        b_seq_len_k = torch.tensor(seq_lens_k, device="cuda", dtype=torch.int32)

        out = attention(
            q_flat,
            k_flat,
            v_flat,
            b_start_loc=b_start_loc_q,
            b_seq_len=b_seq_len_q,
            max_input_len=1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=max(seq_lens_k),
        )

        for i in range(batch):
            sl = seq_lens_k[i]
            s = cumsum[i]
            qb = q_flat[i : i + 1].unsqueeze(2)  # [1, heads, 1, dim]
            kb = k_flat[s : s + sl].unsqueeze(0).permute(0, 2, 1, 3)
            vb = v_flat[s : s + sl].unsqueeze(0).permute(0, 2, 1, 3)
            kb = kb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            vb = vb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=False).squeeze(2)
            torch.testing.assert_close(out[i : i + 1], ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSparseAttentionIntegration:
    """HF model + mtsa.sparsify integration."""

    def test_triton_matches_eager(self, tiny_llama_dir):
        """Triton attention produces same logits and generated tokens as eager."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")

        # Eager baseline
        model_eager = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_eager.eval()
        with torch.no_grad():
            logits_eager = model_eager(input_ids=ids).logits
            out_eager = model_eager.generate(
                ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        del model_eager

        # Triton
        model_triton = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_triton.eval()
        with torch.no_grad():
            logits_triton = model_triton(input_ids=ids).logits
            out_triton = model_triton.generate(
                ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

        # Logits should be close (bf16 tolerance)
        torch.testing.assert_close(logits_triton, logits_eager, rtol=2e-2, atol=2e-2)
        # Generated tokens must be identical (greedy decoding is deterministic)
        assert torch.equal(out_triton, out_eager), (
            f"Generated tokens differ:\n  eager:  {out_eager}\n  triton: {out_triton}"
        )

    def test_triton_padded_batch(self, tiny_llama_dir):
        """Padded batch (2D attention mask) produces valid logits for each sequence."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"

        inputs = tok(
            ["Hello world", "The capital of France is Paris and"],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        assert not torch.isnan(logits).any() and not torch.isinf(logits).any()


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestBackward:
    """Backward pass gradient correctness tests."""

    def _sdpa_backward_ref(self, q, k, v, scale, is_causal=True):
        """Run SDPA forward+backward, return output and gradients."""
        q_ref = q.clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        k_ref = k.clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        v_ref = v.clone().unsqueeze(0).permute(0, 2, 1, 3).requires_grad_(True)
        num_q, num_kv = q_ref.shape[1], k_ref.shape[1]
        if num_q != num_kv:
            r = num_q // num_kv
            k_exp = k_ref.repeat_interleave(r, dim=1)
            v_exp = v_ref.repeat_interleave(r, dim=1)
        else:
            k_exp, v_exp = k_ref, v_ref
        o_ref = F.scaled_dot_product_attention(
            q_ref, k_exp, v_exp, is_causal=is_causal, scale=scale
        )
        o_ref.sum().backward()
        dq = q_ref.grad.permute(0, 2, 1, 3).squeeze(0)
        dk = k_ref.grad.permute(0, 2, 1, 3).squeeze(0)
        dv = v_ref.grad.permute(0, 2, 1, 3).squeeze(0)
        return o_ref.permute(0, 2, 1, 3).squeeze(0).detach(), dq.detach(), dk.detach(), dv.detach()

    def test_backward_causal_matches_sdpa(self):
        """dQ, dK, dV match SDPA backward for causal self-attention."""
        from modelopt.torch.kernels import attention

        seq_len = 16
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(42)
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        _, dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(
            q.detach(), k.detach(), v.detach(), scale, is_causal=True
        )

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)

    def test_backward_gqa(self):
        """Backward with GQA (4 q-heads, 2 kv-heads), multi-tile (seq_len=256)."""
        from modelopt.torch.kernels import attention

        seq_len = 256
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(43)
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        _, dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(
            q.detach(), k.detach(), v.detach(), scale, is_causal=True
        )

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)

    def test_backward_multi_batch_variable_length(self):
        """Multi-batch variable-length causal backward matches per-sample SDPA."""
        from modelopt.torch.kernels import attention

        seq_lens = [8, 12]
        total = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(45)
        q = torch.randn(
            total, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        locs = torch.tensor([0, seq_lens[0]], device="cuda", dtype=torch.int32)
        lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

        o = attention(
            q,
            k,
            v,
            b_start_loc=locs,
            b_seq_len=lens,
            max_input_len=max(seq_lens),
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        # Per-sample SDPA reference
        dq_ref = torch.zeros_like(q)
        dk_ref = torch.zeros_like(k)
        dv_ref = torch.zeros_like(v)
        for b in range(len(seq_lens)):
            s, n = int(locs[b].item()), seq_lens[b]
            _, dq_b, dk_b, dv_b = self._sdpa_backward_ref(
                q.detach()[s : s + n],
                k.detach()[s : s + n],
                v.detach()[s : s + n],
                scale,
                is_causal=True,
            )
            dq_ref[s : s + n] = dq_b
            dk_ref[s : s + n] = dk_b
            dv_ref[s : s + n] = dv_b

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)

    def test_backward_longer_sequences(self):
        """Backward with seq_len=512, GQA, exercises multi-tile loops."""
        from modelopt.torch.kernels import attention

        seq_len = 512
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(49)
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )

        o = attention(
            q,
            k,
            v,
            b_start_loc=torch.tensor([0], device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor([seq_len], device="cuda", dtype=torch.int32),
            max_input_len=seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o.sum().backward()

        _, dq_ref, dk_ref, dv_ref = self._sdpa_backward_ref(
            q.detach(), k.detach(), v.detach(), scale, is_causal=True
        )

        torch.testing.assert_close(q.grad, dq_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(k.grad, dk_ref, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v.grad, dv_ref, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSkipSoftmax:
    """Skip-softmax tile-skipping approximation tests."""

    def _make_inputs(self, batch=2, seq_len=256, num_heads=4, num_kv_heads=2, head_dim=64):
        total = batch * seq_len
        torch.manual_seed(77)
        q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        b_seq_len = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
        b_start_loc = torch.zeros(batch, device="cuda", dtype=torch.int32)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
        return q, k, v, b_start_loc, b_seq_len

    def test_disabled_matches_dense(self):
        """skip_softmax_threshold=None/0.0 produces bit-identical output to dense."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        out_none = attention(q, k, v, locs, lens, 256, softmax_scale=scale)
        out_zero = attention(
            q, k, v, locs, lens, 256, softmax_scale=scale, skip_softmax_threshold=0.0
        )
        assert torch.equal(out_none, out_zero)

    def test_small_threshold_close_to_dense(self):
        """A small threshold (1e-3) should produce output very close to dense."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 256, softmax_scale=scale)
        out_skip = attention(
            q, k, v, locs, lens, 256, softmax_scale=scale, skip_softmax_threshold=1e-3
        )
        torch.testing.assert_close(out_skip, out_dense, rtol=5e-2, atol=5e-2)

    def test_large_threshold_close_to_dense(self):
        """V1 skip-softmax (no v_mean_cache) only skips at tile level.

        With random data, all_skip is rare (requires all 128 rows to agree),
        so V1 output is close to dense. The skip only drops entire tiles
        where ALL rows agree the tile is negligible.
        """
        q, k, v, locs, lens = self._make_inputs(seq_len=512)
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 512, softmax_scale=scale)
        out_skip = attention(
            q, k, v, locs, lens, 512, softmax_scale=scale, skip_softmax_threshold=0.5
        )
        torch.testing.assert_close(out_skip, out_dense, rtol=5e-2, atol=5e-2)

    def test_output_shape_unchanged(self):
        """Skip-softmax does not change output shape."""
        q, k, v, locs, lens = self._make_inputs()
        scale = 1.0 / (64**0.5)
        out = attention(q, k, v, locs, lens, 256, softmax_scale=scale, skip_softmax_threshold=1e-2)
        assert out.shape == q.shape

    def test_monotonic_approximation_error(self):
        """Larger threshold -> larger error vs dense (monotonic degradation)."""
        q, k, v, locs, lens = self._make_inputs(seq_len=512)
        scale = 1.0 / (64**0.5)
        out_dense = attention(q, k, v, locs, lens, 512, softmax_scale=scale)
        errors = []
        for threshold in [1e-4, 1e-2, 1e-1]:
            out_skip = attention(
                q, k, v, locs, lens, 512, softmax_scale=scale, skip_softmax_threshold=threshold
            )
            errors.append((out_skip - out_dense).abs().mean().item())
        assert errors[0] <= errors[1] <= errors[2], f"Errors not monotonic: {errors}"

    def test_decode_single_token(self):
        """Skip-softmax works for decode (single Q token per sequence)."""
        batch = 2
        seq_lens_k = [64, 128]
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(42)
        q_flat = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
        total_kv = sum(seq_lens_k)
        k_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        b_start_loc_q = torch.arange(batch, device="cuda", dtype=torch.int32)
        b_seq_len_q = torch.ones(batch, device="cuda", dtype=torch.int32)
        cumsum = [0]
        for sl in seq_lens_k:
            cumsum.append(cumsum[-1] + sl)
        b_start_loc_k = torch.tensor(cumsum[:-1], device="cuda", dtype=torch.int32)
        b_seq_len_k = torch.tensor(seq_lens_k, device="cuda", dtype=torch.int32)

        out_dense = attention(
            q_flat,
            k_flat,
            v_flat,
            b_start_loc_q,
            b_seq_len_q,
            1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=max(seq_lens_k),
        )
        out_skip = attention(
            q_flat,
            k_flat,
            v_flat,
            b_start_loc_q,
            b_seq_len_q,
            1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=max(seq_lens_k),
            skip_softmax_threshold=1e-3,
        )
        torch.testing.assert_close(out_skip, out_dense, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSkipSoftmaxHFIntegration:
    """HF integration for skip-softmax via mtsa.sparsify()."""

    def test_skip_softmax_via_sparsify(self, tiny_llama_dir):
        """mtsa.sparsify() with triton_skip_softmax produces finite logits."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import modelopt.torch.sparsity.attention_sparsity as mtsa

        tok = AutoTokenizer.from_pretrained(tiny_llama_dir)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        ids = torch.randint(1, tok.vocab_size, (1, 64), device="cuda")

        # Dense baseline (triton backend, no skip)
        model_dense = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="modelopt_triton",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_dense.eval()
        with torch.no_grad():
            logits_dense = model_dense(input_ids=ids).logits
        del model_dense

        # Skip-softmax via mtsa.sparsify()
        model_skip = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        mtsa.sparsify(model_skip, mtsa.SKIP_SOFTMAX_TRITON_DEFAULT)
        model_skip.eval()
        with torch.no_grad():
            logits_skip = model_skip(input_ids=ids).logits

        assert not torch.isnan(logits_skip).any(), "NaN in skip-softmax logits"
        assert not torch.isinf(logits_skip).any(), "Inf in skip-softmax logits"
        # On short sequences (64 tokens), no tiles are skipped — output should match dense
        torch.testing.assert_close(logits_skip, logits_dense, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestSkipSoftmaxV25:
    """V2.51 pool-K approximate weight + precomputed v_mean tests."""

    def _make_inputs(
        self,
        batch=1,
        seq_len=2048,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        dtype=torch.float32,
    ):
        """Create diffusion-like Q/K/V with moderate clustering.

        Q tokens cluster near a centroid so that most KV tiles are fully skipped
        by all query rows, exercising the V2.51 all_skip path. A few 'hot' K
        tiles are aligned with the centroid; the rest are random background.
        """
        total = batch * seq_len
        BLOCK_N = 64
        torch.manual_seed(42)
        centroid = torch.randn(1, 1, head_dim, device="cuda", dtype=dtype)
        q = centroid.expand(total, num_heads, head_dim) + torch.randn(
            total, num_heads, head_dim, device="cuda", dtype=dtype
        ) * 0.8
        k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype) * 0.5
        # Make 2 tiles aligned (but not perfectly) with Q
        for tile_i in [0, 5]:
            s = tile_i * BLOCK_N
            k[s : s + BLOCK_N] = centroid.expand(
                BLOCK_N, num_kv_heads, head_dim
            ) + torch.randn(BLOCK_N, num_kv_heads, head_dim, device="cuda", dtype=dtype) * 0.5
        v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=dtype)
        b_seq_len = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
        b_start_loc = torch.zeros(batch, device="cuda", dtype=torch.int32)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)
        return q, k, v, b_start_loc, b_seq_len

    def _run_v25(self, q, k, v, locs, lens, seq_len, threshold, num_kv_heads, head_dim):
        """Run V2.51 attention with v_mean_cache."""
        import math

        import triton

        BLOCK_N = 64
        batch = lens.shape[0]
        n_kv_tiles = math.ceil(seq_len / BLOCK_N)
        BLOCK_D = triton.next_power_of_2(head_dim)
        v_mean_cache = torch.zeros(
            batch, num_kv_heads, n_kv_tiles, BLOCK_D, device="cuda", dtype=torch.float32
        )
        return attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            is_causal=False,
            skip_softmax_threshold=threshold,
            skip_softmax_normalize_by_seqlen=True,
            v_mean_cache=v_mean_cache,
        )

    def test_v25_closer_to_dense_than_v1(self):
        """V2.51 approximation should be closer to dense than V1 (which zeros skipped tiles)."""
        seq_len, num_heads, num_kv_heads, head_dim = 2048, 4, 4, 64
        q, k, v, locs, lens = self._make_inputs(
            seq_len=seq_len, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        threshold = 0.15

        out_dense = attention(q, k, v, locs, lens, seq_len, is_causal=False)
        out_v1 = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            is_causal=False,
            skip_softmax_threshold=threshold,
            skip_softmax_normalize_by_seqlen=True,
        )
        out_v25 = self._run_v25(q, k, v, locs, lens, seq_len, threshold, num_kv_heads, head_dim)

        err_v1 = (out_v1 - out_dense).abs().mean().item()
        err_v25 = (out_v25 - out_dense).abs().mean().item()

        assert err_v25 < err_v1 * 0.5, (
            f"V2.51 should be significantly closer to dense than V1: "
            f"err_v25={err_v25:.6f}, err_v1={err_v1:.6f}, ratio={err_v25/err_v1:.3f}"
        )

    def test_v25_no_nan_inf(self):
        """V2.51 output should be finite."""
        seq_len, num_heads, num_kv_heads, head_dim = 2048, 4, 4, 64
        q, k, v, locs, lens = self._make_inputs(
            seq_len=seq_len, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        out = self._run_v25(q, k, v, locs, lens, seq_len, 0.15, num_kv_heads, head_dim)
        assert not torch.isnan(out).any(), "NaN in V2.51 output"
        assert not torch.isinf(out).any(), "Inf in V2.51 output"

    def test_v25_high_threshold_matches_dense(self):
        """V2.51 with very high threshold (no skipping) should closely match dense.

        threshold=100 → skip_threshold_log2 = -100*log2(seq) ≈ -1100, so no tile
        is ever skipped. The V2.5 code path still runs (precompute_vmean, fixed
        block config), so small numerical diffs from different accumulation order
        are expected.
        """
        seq_len, num_heads, num_kv_heads, head_dim = 2048, 4, 4, 64
        q, k, v, locs, lens = self._make_inputs(
            seq_len=seq_len, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        out_dense = attention(q, k, v, locs, lens, seq_len, is_causal=False)
        # threshold=100 → effectively no tiles skipped
        out_v25 = self._run_v25(q, k, v, locs, lens, seq_len, 100.0, num_kv_heads, head_dim)
        torch.testing.assert_close(out_v25, out_dense, rtol=5e-3, atol=5e-3)

    def test_v25_differs_from_v1(self):
        """V2.51 output should differ from V1 when tiles are fully skipped."""
        seq_len, num_heads, num_kv_heads, head_dim = 2048, 4, 4, 64
        q, k, v, locs, lens = self._make_inputs(
            seq_len=seq_len, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        threshold = 0.15

        out_v1 = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            is_causal=False,
            skip_softmax_threshold=threshold,
            skip_softmax_normalize_by_seqlen=True,
        )
        out_v25 = self._run_v25(q, k, v, locs, lens, seq_len, threshold, num_kv_heads, head_dim)
        assert not torch.allclose(out_v25, out_v1, atol=1e-4), (
            "V2.51 output should differ from V1 when tiles are skipped"
        )

    def test_v25_gqa(self):
        """V2.51 works with GQA (4 q-heads, 2 kv-heads)."""
        seq_len, num_heads, num_kv_heads, head_dim = 2048, 4, 2, 64
        q, k, v, locs, lens = self._make_inputs(
            seq_len=seq_len, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        out_dense = attention(q, k, v, locs, lens, seq_len, is_causal=False)
        out_v25 = self._run_v25(q, k, v, locs, lens, seq_len, 0.15, num_kv_heads, head_dim)
        out_v1 = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            is_causal=False,
            skip_softmax_threshold=0.15,
            skip_softmax_normalize_by_seqlen=True,
        )

        err_v1 = (out_v1 - out_dense).abs().mean().item()
        err_v25 = (out_v25 - out_dense).abs().mean().item()
        assert err_v25 < err_v1, f"V2.51 GQA: err_v25={err_v25:.6f} >= err_v1={err_v1:.6f}"

    def test_v25_partial_last_tile(self):
        """V2.51 handles partial last KV tile correctly (seq_len not multiple of BLOCK_N)."""
        # seq_len=2000, BLOCK_N=64 → last tile has 2000 - 31*64 = 2000 - 1984 = 16 valid tokens
        seq_len, num_heads, num_kv_heads, head_dim = 2000, 4, 4, 64
        q, k, v, locs, lens = self._make_inputs(
            seq_len=seq_len, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        out_dense = attention(q, k, v, locs, lens, seq_len, is_causal=False)
        out_v25 = self._run_v25(q, k, v, locs, lens, seq_len, 0.15, num_kv_heads, head_dim)

        assert not torch.isnan(out_v25).any(), "NaN with partial last tile"
        err = (out_v25 - out_dense).abs().mean().item()
        assert err < 0.1, f"V2.51 partial tile error too large: {err:.6f}"
