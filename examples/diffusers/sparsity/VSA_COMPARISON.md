# Video Sparse Attention (VSA) vs Skip-Softmax: Algorithm Comparison

## VSA Overview

VSA (from FastVideo) is a **two-branch hybrid** sparse attention. It does NOT simply
drop tiles — it computes a cheap global approximation for ALL blocks, then adds
precise attention for the most important blocks only.

## VSA Algorithm Step by Step

### Input

Q, K, V shaped `[B, H, seq_len, d]`. The video sequence is divided into 3D tiles
of size 4x4x4 = 64 tokens each. With a sequence of 98,304 tokens, that gives
1,536 blocks.

Some blocks at the boundary of the video grid may have fewer than 64 valid
tokens. `variable_block_sizes[j]` stores how many valid tokens block j has.

### Step 1: Block Compression (mean pooling)

Each block's 64 token vectors are averaged into a single representative vector:

```python
# ops.py lines 114-123
q_c = q.view(B, H, num_blocks, 64, d)            # reshape into blocks
q_c = q_c.float().sum(dim=3) / block_sizes       # [B, H, num_blocks, d]
# same for k_c, v_c
```

This is just a mean over the token dimension within each block. If a block has
48 valid tokens (boundary block), it divides by 48, not 64.

After this step: `q_c, k_c, v_c` are each `[B, H, num_blocks, d]` — one vector
per block instead of 64.

### Step 2: Block-Level Attention (compression branch)

Standard scaled dot-product attention on the compressed representations:

```python
# ops.py lines 125-127
scores = q_c @ k_c^T / sqrt(d)       # [B, H, q_blocks, kv_blocks]
attn = softmax(scores, dim=-1)        # [B, H, q_blocks, kv_blocks]
out_c = attn @ v_c                    # [B, H, q_blocks, d]
```

This is a tiny attention computation — 1,536 x 1,536 instead of 98,304 x 98,304.
It captures which blocks globally attend to which other blocks, but at block
granularity (one vector per 64 tokens).

The per-block output is then broadcast back to every token in that block:

```python
# ops.py lines 129-131
out_c = out_c.unsqueeze(3).repeat(1, 1, 1, 64, 1)   # same output for all 64 tokens
out_c = out_c.view(B, H, seq_len, d)
```

**All 64 tokens in the same block get the identical output vector.** This is the
coarse approximation — it knows the global attention pattern but loses
within-block variation.

### Step 3: Top-K Block Selection

Reuse the block-level `scores` from step 2 to pick which KV blocks matter most
for each query block:

```python
# ops.py lines 134-136
topk = ceil((1 - sparsity) * num_kv_blocks)
topk_idx = topk(scores, k=topk, dim=-1).indices     # [B, H, q_blocks, topk]
mask = zeros_like(scores, bool).scatter_(-1, topk_idx, True)
```

The selection metric is the block-level attention score — "how strongly does
query-block i attend to key-block j, when both are represented by their mean
vector?"

With sparsity=0.9 and 1,536 blocks: topk = 154 blocks selected per query block.

### Step 4: Sparse Full-Precision Attention (sparse branch)

Run real flash attention, but only between each query block and its top-K
selected KV blocks:

```python
# ops.py lines 140-145
out_s = block_sparse_attn(q, k, v, mask, variable_block_sizes)
```

Inside the kernel (Triton or ThunderKittens), for each query block:
- Load the full 64 Q tokens
- Loop only over the selected KV blocks (154 instead of 1,536)
- For each selected KV block, load its 64 K/V tokens
- Compute exact QK^T, softmax, PV at full token resolution
- Use online softmax (Flash Attention style) across the selected blocks

This produces precise per-token outputs, but only from the top-K blocks.

### Step 5: Combine with Learned Gate

```python
# ops.py lines 147-148
output = out_c * gate_compress + out_s
```

Where `gate_compress` is a per-token weight from a learned linear layer:

```python
# ltx2.py lines 1286-1287, 1310-1311
self.to_gate_compress = nn.Linear(context_dim, inner_dim, bias=True)
gate_compress = self.to_gate_compress(context)   # [B, seq_len, H, d]
```

The gate learns how much to trust each branch per token. Initialized to produce
values that blend the two branches — the model learns during training which
tokens benefit more from the coarse global view vs the precise local view.

## Numerical Example

seq_len=98,304 tokens, 1,536 blocks, sparsity=0.9, d=128:

| | Compression branch | Sparse branch |
|---|---|---|
| **Attention matrix** | 1,536 x 1,536 = 2.4M entries | 1,536 x 154 x 64 x 64 = 617M entries |
| **Computes** | Global block-to-block attention | Local token-to-token attention for top-K |
| **Output** | Same vector for all 64 tokens in a block | Unique per-token output |
| **Cost** | Very cheap (~2.5% of full attention) | ~10% of full attention |

Full attention would be 98,304 x 98,304 = 9.7B entries. VSA computes ~12.5%.

## How VSA Differs from Skip-Softmax

| | Skip-Softmax (ModelOpt) | VSA (FastVideo) |
|---|---|---|
| **What happens to skipped blocks** | Zeroed out (contribute nothing) | Approximated via block-averaged attention |
| **Selection method** | Threshold on cummax gap | Top-K on block-level attention scores |
| **Selection signal** | QK scores from the actual attention | QK scores between block-mean vectors |
| **Sparsity guarantee** | Variable (depends on input) | Exact (always top-K) |
| **Learned parameters** | None | Gate weight (nn.Linear per layer) |
| **Output composition** | Single branch (sparse mask before softmax) | Two branches blended by learned gate |
| **Training support** | Grad flows but no compute saving | Full forward+backward kernels |
| **Calibration** | Required (percentile threshold) | Not required (top-K is parameter-free) |

### The Key Conceptual Difference

**Skip-Softmax** says: "this tile's contribution is negligible after softmax,
so drop it entirely."

**VSA** says: "we can't afford full attention on every tile, but every tile
matters somewhat. Approximate all tiles cheaply (compression branch), then
refine the most important ones with full attention (sparse branch). Learn to
blend the two."

VSA never fully drops any information — the compression branch always includes
all blocks, just at block granularity. Skip-softmax is a hard binary decision
per tile.

### Block Selection: Different Signals

Skip-softmax uses the **actual QK scores** during attention computation to
decide skipping. The cummax gap tells you "this tile's max score is far below
the running peak, so softmax will suppress it."

VSA uses **block-averaged QK scores** as a proxy. It cannot see within-block
variation — two blocks with the same mean but different distributions look
identical. But it's cheap enough to compute for all block pairs, giving a
global view of the attention pattern before committing to which blocks get
full attention.
