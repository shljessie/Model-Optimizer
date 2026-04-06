# Skip-Softmax Diffusion V3.2: Compression-Branch Skip Path

## Summary

V3.2 replaces the per-tile-independent pool-K + v_mean approximation (V2.5/V3)
with a **globally-normalized mini-softmax over precomputed k_mean/v_mean vectors**
for skipped tiles. This is inspired by VSA's compression branch architecture
and directly addresses the two dominant error sources in V3/V3.1.

The key insight: **VSA's compression branch produces properly normalized,
attention-weighted output even for "approximate" tiles, while skip-softmax's
pool-K treats each tile independently with no cross-tile normalization.**

## Prerequisite: V3 (Majority Vote)

V3.2 builds on V3. Read `skip_softmax_diffusion_v3.md` first. V3 provides:
- Majority vote: skip if >= X% of 128 rows agree
- row_max fix: always update row_max from actual tile_row_max
- Gray zone (V3.1): exact P + v_mean for borderline tiles

## The Problem: Why Pool-K + v_mean Has Poor Quality

### Error Source 1: No Cross-Tile Weight Normalization

Pool-K computes approximate attention weights per tile **independently**:

```
For each skipped tile j:
  k_mean_j = mean(K[tile_j])
  approx_score_j[i] = Q[i] @ k_mean_j * scale
  p_approx_j[i] = exp2(approx_score_j - m_new) * N
```

These weights are not normalized across tiles. The final normalization
happens implicitly through online softmax's `row_sum`, but pool-K
systematically underestimates weights (Jensen's inequality), so the
relative weighting between tiles is wrong.

### Error Source 2: Uniform v_mean Ignores Attention Distribution

Even with correct weights, `v_mean = mean(V[tile])` is a **uniform**
average. It ignores which V vectors the query actually attends to within
the tile. This is the "value error" from the V3.1 analysis (Term 2).

### How VSA's Compression Branch Solves This

VSA's compression branch does:

```
K_block = mean(K[block])   for ALL blocks
V_block = mean(V[block])   for ALL blocks
output = softmax(Q @ K_block^T) @ V_block
```

This is a **single softmax** over all block-averaged K vectors simultaneously,
producing correctly normalized weights. The output is an attention-weighted
combination of V_block vectors — different queries get different V_block
mixtures based on their actual attention pattern.

## V3.2 Algorithm

### Two-Pass Architecture

**Pass 1 (main tile loop)**: Same as V3 for KEEP and GRAY ZONE tiles.
For FULL SKIP tiles: instead of pool-K + v_mean, **defer to compression pass**.
Record the tile index in a skip_mask and update row_max only.

**Pass 2 (compression pass)**: After the main loop, compute a mini-softmax
over all deferred tiles' precomputed k_mean/v_mean vectors:

```
For each skipped tile t:
  score[i] = Q[i] @ k_mean[t] * qk_scale     # [BLOCK_M]
  # Online softmax accumulation across all skipped tiles
  → produces comp_acc, comp_row_max, comp_row_sum
```

**Merge**: Combine main loop (kept tiles) and compression pass (skipped tiles)
using the standard two-chunk online softmax merge formula:

```
m = max(row_max_kept, row_max_comp)
corr_kept = exp2(row_max_kept - m)
corr_comp = exp2(row_max_comp - m)
row_sum = row_sum_kept * corr_kept + row_sum_comp * corr_comp
acc = acc_kept * corr_kept + acc_comp * corr_comp
output = acc / row_sum
```

### Per-Tile Flow

```
Before tile loop:
  Precompute k_mean[j] and v_mean[j] for all KV tiles  ← NEW: also k_mean
  Initialize skip_mask[MAX_KV_TILES] = 0                ← NEW

Tile j:
  1. Load K tile, compute scores (same as V3)
  2. Skip decision: majority vote (same as V3)

  If NOT all_skip (KEEP):
    3-6. Full softmax + V load + BMM2 (unchanged)

  If all_skip AND gray zone (V3.1):
    3-6. Exact P + v_mean (unchanged)

  If all_skip AND full skip:
    3. skip_mask[j] = 1           ← NEW: mark as deferred
    4. row_max = max(row_max, tile_row_max)  ← prevent staleness

After tile loop:
  COMPRESSION PASS:                         ← NEW
    For each tile t where skip_mask[t] = 1:
      Load k_mean[t], v_mean[t] from cache
      score = Q @ k_mean[t] * qk_scale
      Online softmax update → comp_acc, comp_row_max, comp_row_sum

  MERGE:                                    ← NEW
    m = max(row_max, comp_row_max)
    acc = acc * exp2(row_max - m) + comp_acc * exp2(comp_row_max - m)
    row_sum = row_sum * exp2(row_max - m) + comp_row_sum * exp2(comp_row_max - m)

  output = acc / row_sum
```

## What Changed vs V3/V3.1

| | V3/V3.1 | V3.2 |
|---|---|---|
| **Full skip path** | Pool-K per tile (independent) | Mini-softmax over all skipped tiles (global) |
| **Weight normalization** | Per-tile (implicit via row_sum) | Global softmax (properly normalized) |
| **V approximation** | p_approx * v_mean (per tile) | softmax-weighted v_mean blend (cross-tile) |
| **Precompute** | v_mean only | k_mean + v_mean |
| **Extra memory** | 0 | k_mean_cache (~960 KB for LTX-2) |
| **KEEP path** | Unchanged | Unchanged |
| **Gray zone** | Exact P + v_mean (unchanged) | Unchanged |

## Cost Analysis

### Precompute Overhead

The new `_precompute_kv_mean` kernel computes both k_mean and v_mean in a
single pass (shared tile mask and position computation). One extra K load
per tile compared to `_precompute_vmean`.

For LTX-2: 60 tiles × 64 tokens × 128 dims = ~0.5M loads. Negligible.

### Compression Pass Cost Per Q-Tile

For LTX-2 with ~60 KV tiles and ~30 skipped:

```
Per skipped tile in compression pass:
  k_mean load: 128 floats from L2 (~512 bytes)
  v_mean load: 128 floats from L2 (~512 bytes)
  Q @ k_mean: 128 × 128 = 16K FMA
  exp2 + accumulate: 128 exp2 + 128 FMA
  v_mean accumulate: 128 × 128 = 16K FMA

Total for 30 skipped tiles:
  ~30 × 32K FMA = ~1M FMA
  ~30 × 1KB L2 = ~30 KB L2 loads

Compare: one KEPT tile costs ~2M FMA + 16KB HBM load
```

The compression pass costs roughly **half a kept tile**, regardless of how
many tiles were skipped. This is because the mini-softmax operates on
k_mean vectors (per-tile averages, not full K tiles), so each tile's
contribution is just one dot product instead of BLOCK_N=64 dot products.

### Memory Overhead

```
k_mean_cache: [batch, num_kv_heads, num_kv_tiles, BLOCK_D] float32
For LTX-2:    [1, 32, 60, 128] × 4 bytes = ~960 KB

skip_mask:    [MAX_KV_TILES] int32 per thread block (registers)
              128 × 4 = 512 bytes (negligible)
```

### Net Performance Impact

For a typical LTX-2 attention call with ~30 skipped tiles:
- **Removed**: 30 × pool-K path (~33K FMA each) = ~1M FMA
- **Added**: 1 compression pass = ~1M FMA
- **Net compute**: roughly neutral
- **Quality improvement**: significant (global vs per-tile normalization)

The win is in **quality, not speed**. The compute cost is similar to V3,
but the output is much closer to exact attention.

## Comparison Table

| | V2.51 | V3 | V3.2 |
|---|---|---|---|
| **Skip decision** | Unanimous | Majority vote | Majority vote |
| **Full skip approximation** | Pool-K + v_mean (per tile) | Pool-K + v_mean (per tile) | Mini-softmax(k_mean) @ v_mean (global) |
| **Weight normalization** | Per-tile | Per-tile | Global softmax |
| **Value approximation** | Uniform v_mean | Uniform v_mean | Attention-weighted v_mean blend |
| **Precompute** | v_mean | v_mean | k_mean + v_mean |
| **LTX-2 skip rate** | ~5% | ~30-50% | ~30-50% (same skip rate) |
| **Quality** | Good (few skips) | Moderate | Good (proper normalization) |
| **Extra memory** | 0 | 0 | ~960 KB (k_mean_cache) |

## Files Modified

| File | Changes |
|------|---------|
| `modelopt/torch/kernels/triton_fa.py` | New `_precompute_kv_mean` kernel, `APPLY_SKIP_V32` constexpr + skip_mask + compression pass + merge in `_attn_fwd_body`, new `_attention_v32_forward` wrapper, `k_mean_cache` param in `attention()` |
| `modelopt/torch/sparsity/attention_sparsity/config.py` | `enable_v32: bool` field |
| `modelopt/torch/sparsity/attention_sparsity/methods/triton_skip_softmax_diffusion.py` | Thread `enable_v32` to both backends |
| `modelopt/torch/sparsity/attention_sparsity/kernels/diffusers_triton_attention.py` | `enable_v32` in thread-local config, lazy k_mean_cache allocation |
| `modelopt/torch/sparsity/attention_sparsity/kernels/ltx_triton_attention.py` | Same as diffusers backend |
| `examples/diffusers/sparsity/ltx2_skip_softmax.py` | `--enable-v32` CLI flag |

## Usage

```bash
# V3.2 with calibration (recommended)
python ltx2_skip_softmax.py --prompt "A cat playing piano" --output out.mp4 \
    --calibrate --target-sparsity 0.2 --enable-v32

# V3.2 with custom majority percentage
python ltx2_skip_softmax.py --prompt "A cat playing piano" --output out.mp4 \
    --calibrate --target-sparsity 0.2 --enable-v32 --majority-pct 0.7
```

`--enable-v32` implies `--enable-v3` (majority vote) and `enable_v25` (V2.5 base).

## Open Questions

1. **Should the gray zone also use the compression pass?** Currently, gray
   zone tiles use exact P + v_mean (V3.1). They could instead be deferred
   to the compression pass. However, the gray zone already has exact softmax
   weights (just coarse values), which is arguably better than the compression
   pass's k_mean-based weights. Validate empirically.

2. **Does n_valid_kv scaling matter in the compression pass?** Currently the
   compression pass multiplies by `n_valid_kv` to account for partial tiles.
   This matches the pool-K path but may interact differently with the global
   softmax. May need tuning.

3. **Loop unrolling for the compression pass.** The current implementation
   loops over MAX_KV_TILES with dynamic skip_mask checks. For small tile
   counts (LTX-2: ~60), this is fast. For longer sequences, the loop overhead
   may become significant. Consider tiling the compression pass itself.
