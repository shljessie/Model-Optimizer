# Skip-Softmax Studies & Captured Data

## Captured Attention Inputs

**Location:** `experiments/attn_input/`
**Script:** `capture_attn_inputs.py`
**Pipeline config:** 768x1280, 121 frames, 40 denoising steps, seed=42, prompt="A cat playing piano"
**Model:** LTX-2 19B, Triton baseline (no sparsity)

### Saved Files

| File | Size | Description |
|------|------|-------------|
| `step_000.pt` | 9.1 GB | Denoising step 0 (noisiest) |
| `step_009.pt` | 9.1 GB | Denoising step 9 |
| `step_019.pt` | 6.9 GB | Denoising step 19 |
| `step_029.pt` | 9.1 GB | Denoising step 29 |
| `step_039.pt` | 6.9 GB | Denoising step 39 (cleanest) |

Note: Size varies because the pipeline calls the transformer 160 times across 40 denoising steps
(4 transformer calls per step: 2 for CFG x 2 stages). Steps with different stage configs produce
different file sizes.

### Data Structure

```python
data = torch.load("experiments/attn_input/step_000.pt", map_location="cpu")
# data is a list of 288 dicts (one per attention call in the transformer forward)

entry = data[0]
# entry["q"]         — torch.bfloat16, shape [B, seq_q, H*D]
# entry["k"]         — torch.bfloat16, shape [B, seq_k, H*D]
# entry["v"]         — torch.bfloat16, shape [B, seq_k, H*D]
# entry["heads"]     — int (number of attention heads)
# entry["layer_idx"] — int (0-287, sequential call order within the step)
```

### LTX-2 Attention Layout (per transformer block)

Each of the 48 transformer blocks has 6 attention calls in this repeating pattern:

| Offset | Type | seq_q | seq_k | H*D | heads | D | Description |
|--------|------|-------|-------|-----|-------|---|-------------|
| +0 | attn1 (video self) | 3840 | 3840 | 4096 | 32 | 128 | Video self-attention |
| +1 | attn2 (video cross) | 3840 | 1024 | 4096 | 32 | 128 | Video cross-attention (text) |
| +2 | audio_attn1 | 126 | 126 | 2048 | 16 | 128 | Audio self-attention |
| +3 | audio_attn2 | 126 | 1024 | 2048 | 16 | 128 | Audio cross-attention (text) |
| +4 | video_to_audio | 3840 | 126 | 2048 | 16 | 128 | Video-to-audio cross-attn |
| +5 | audio_to_video | 126 | 3840 | 2048 | 16 | 128 | Audio-to-video cross-attn |

So layer_idx 0,6,12,...,282 are video self-attention (attn1) — these are the ones we apply sparsity to.
Layer_idx 1,7,13,...,283 are video cross-attention (attn2) — disabled for sparsity.

To get attn1 entries only:
```python
attn1_entries = [e for e in data if e["layer_idx"] % 6 == 0]
# len(attn1_entries) == 48 (one per transformer block)
# block_idx = entry["layer_idx"] // 6
```

### How to Load and Feed into Triton Kernel

```python
import torch
import math
from modelopt.torch.kernels.triton_fa import attention

data = torch.load("experiments/attn_input/step_019.pt", map_location="cpu")

# Pick a video self-attention layer (attn1): indices 0, 6, 12, ..., 282
entry = data[0]  # layer 0, block 0's attn1
q_ltx = entry["q"].cuda()  # [1, 3840, 4096] bf16
k_ltx = entry["k"].cuda()  # [1, 3840, 4096] bf16
v_ltx = entry["v"].cuda()  # [1, 3840, 4096] bf16
heads = entry["heads"]     # 32
dim_head = q_ltx.shape[-1] // heads  # 128

# Reshape to varlen format: [B*T, H, D]
b, seq_q = q_ltx.shape[0], q_ltx.shape[1]
seq_k = k_ltx.shape[1]
q = q_ltx.view(b, seq_q, heads, dim_head).reshape(b * seq_q, heads, dim_head).contiguous()
k = k_ltx.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()
v = v_ltx.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()

# Varlen metadata
b_start_loc = torch.arange(b, device="cuda", dtype=torch.int32) * seq_q
b_seq_len = torch.full((b,), seq_q, device="cuda", dtype=torch.int32)
b_start_loc_k = torch.arange(b, device="cuda", dtype=torch.int32) * seq_k
b_seq_len_k = torch.full((b,), seq_k, device="cuda", dtype=torch.int32)
scale = 1.0 / math.sqrt(dim_head)

# Dense baseline (no sparsity)
o_dense = attention(q, k, v, b_start_loc, b_seq_len, seq_q,
                    is_causal=False, softmax_scale=scale,
                    b_start_loc_k=b_start_loc_k, b_seq_len_k=b_seq_len_k,
                    max_input_len_k=seq_k)

# With skip-softmax (V1: pure skip)
o_sparse = attention(q, k, v, b_start_loc, b_seq_len, seq_q,
                     is_causal=False, softmax_scale=scale,
                     b_start_loc_k=b_start_loc_k, b_seq_len_k=b_seq_len_k,
                     max_input_len_k=seq_k,
                     skip_softmax_threshold=0.054199,
                     skip_softmax_normalize_by_seqlen=True)

# With V2.5 (pool-K + v_mean)
import triton
BLOCK_N = 64
n_kv_tiles = math.ceil(seq_k / BLOCK_N)
BLOCK_D = triton.next_power_of_2(dim_head)
v_mean_cache = torch.zeros(b, heads, n_kv_tiles, BLOCK_D, device="cuda", dtype=torch.float32)
o_v25 = attention(q, k, v, b_start_loc, b_seq_len, seq_q,
                  is_causal=False, softmax_scale=scale,
                  b_start_loc_k=b_start_loc_k, b_seq_len_k=b_seq_len_k,
                  max_input_len_k=seq_k,
                  skip_softmax_threshold=0.054199,
                  skip_softmax_normalize_by_seqlen=True,
                  v_mean_cache=v_mean_cache)

# Compare
diff_v1 = (o_dense - o_sparse).abs().max().item()
diff_v25 = (o_dense - o_v25).abs().max().item()
print(f"V1 max diff: {diff_v1:.6f}, V2.5 max diff: {diff_v25:.6f}")
```

## Experiment Results Summary

All experiments at 75% sparsity, fixed threshold=0.054199, 768x1280, 121 frames, 40 steps.

| Label | Skip Branch | Time | Video |
|-------|-------------|------|-------|
| dbg10 | Exact softmax + exact V (quality ceiling) | 138.0s | `sparse_v252dbg10_75pct.mp4` |
| dbg13 | Pure skip / nothing (V1) | 144.2s | `sparse_v252dbg13_75pct.mp4` |
| dbg9 | Exact softmax (8192 exp2) + v_mean | 147.9s | `sparse_v252dbg9_75pct.mp4` |
| dbg12 | Pool-K (128 exp2) + v_mean | 148.0s | `sparse_v252dbg12_75pct.mp4` |
| triton_baseline | No sparsity (Triton kernel) | ~165s | `triton_baseline.mp4` |

### Key Observations

1. **dbg10 is fastest** despite doing more work — likely because no branching in the kernel
   lets Triton optimize the full attention path better.
2. **dbg13 (pure skip)** saves V load + BMM2 but is slower than dbg10 — the branch
   overhead dominates at this sparsity level.
3. **dbg9 vs dbg12 identical speed** — the 8192 to 128 exp2 reduction doesn't matter because
   exp2 is not the bottleneck at this scale; memory bandwidth and branch overhead dominate.
4. **V_mean vs exact V quality gap** is the key question — compare videos to assess.
