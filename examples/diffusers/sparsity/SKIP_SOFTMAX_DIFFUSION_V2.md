# Skip-Softmax Diffusion V2: Cached Denominator Correction

## Motivation

Skip-softmax V1 hard-skips tiles — the skipped tile contributes zero to both
the softmax denominator (`l`) and the output accumulator (`acc`). This works
well at low sparsity (10-30%) where skipped tiles are truly negligible.

At higher sparsity (>50%), diffusion models degrade because unlike LLMs,
diffusion attention scores are relatively flat. The gap between cummax and
block_max is modest (e.g., 10 vs 1, not 100,000 vs 10). Skipped tiles still
carry meaningful weight, and dropping them entirely:

1. Makes `l` too small (missing denominator terms)
2. Inflates the kept tiles' weights (since `output = acc / l`, smaller `l`
   means larger weights for kept tiles)
3. Accumulates error across many skipped tiles

## Key Insight

The main source of error from hard-skipping is the **incorrect `l`**, not the
missing V contribution. When `l` is too small, every kept tile's weight gets
inflated proportionally. Fixing `l` alone — without computing BMM2 or loading
V — corrects the softmax normalization at near-zero cost.

## Algorithm

### What Changes vs V1

In V1 (and current TRT-LLM), when a tile is skipped:

```
l unchanged
m unchanged
acc unchanged
→ tile contributes nothing
```

In V2, when a tile is skipped:

```
l += cached_p    ← add cached softmax weight from previous denoising step
m unchanged
acc unchanged    ← still no BMM2, no V load
→ tile contributes to the denominator but not the numerator
```

### Per-Tile Flow

```
Tile j:
  1. BMM1: scores_j = Q @ K_j^T              (always computed)
  2. block_max_j = max(scores_j)              (always computed)
  3. gap = cummax - block_max_j

  If gap < threshold (KEEP):
    4. Compute softmax: p_j = exp(scores_j - m)
    5. Update l: l += sum(p_j)
    6. Compute BMM2: acc += p_j @ V_j
    7. Cache: store sum(p_j) for this tile     ← new in V2

  If gap >= threshold (SKIP):
    4. Retrieve cached_p from previous denoising step
    5. Update l: l += cached_p                  ← new in V2
    6. Skip softmax exp, BMM2, V load           (same as V1)
```

### Correction Factor Handling

The cached `p` was computed at the previous denoising step with a possibly
different `m`. Does it need correction?

**Usually no.** By the time we're skipping tile j, the cummax (`m`) is already
stable — the peak tile appeared earlier in the scan. A later tile changing `m`
would mean it has higher scores than everything before, so it wouldn't be
skipped itself.

**In the rare case** a later tile does change `m`, the standard Flash Attention
correction (`l *= e^(m_old - m_new)`) applies to the entire `l` including the
cached part. This is correct — the cached `p` is already folded into `l`, so
it gets corrected along with everything else. No special handling needed.

### What to Cache

Per tile, per head, per query row: one scalar — `sum(p_j)`, the total softmax
weight of the tile (sum of `e^(score_i - m)` for all elements in the tile).

Storage: `[num_tiles, num_heads, br]` floats per denoising step. For LTX-2
with seq_k=4224, br=128: `33 tiles × 32 heads × 128 rows × 4 bytes ≈ 0.5 MB`.
Negligible.

This cache is written during non-skipped tile processing (step 7 above) and
read during skipped tile processing at the next denoising step.

### First Denoising Step

At step 0, there is no cache. Two options:

1. Run step 0 with no skipping (full attention) to populate the cache
2. Run step 0 with V1 hard-skip (no cached_p correction)

Option 1 is safer and matches LiteAttention's approach (first step is always
full).

## Example

Scores: `[3, 1, 7, 2]`, skip tile 3 (score 2).

**V1 (hard skip):**
```
l = e^(3-7) + e^(1-7) + e^(7-7)
  = 0.018 + 0.002 + 1.0
  = 1.020

output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂) / 1.020

weights: [1.76%, 0.20%, 98.04%]     ← slightly inflated
```

**V2 (cached denominator correction):**
Previous step's cached p₃ = 0.006.

```
l = e^(3-7) + e^(1-7) + e^(7-7) + 0.006
  = 0.018 + 0.002 + 1.0 + 0.006
  = 1.026

output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂) / 1.026

weights: [1.75%, 0.19%, 97.47%]     ← closer to true [1.75%, 0.19%, 97.37%]
```

**True (no skip):**
```
l = 0.018 + 0.002 + 1.0 + 0.007 = 1.027

weights: [1.75%, 0.19%, 97.37%, 0.68%]
```

V2's kept-tile weights are much closer to the true values than V1's. The
error in `l` drops from 0.7% (V1) to 0.1% (V2).

## Cost Analysis

Per skipped tile, V2 adds:
- 1 memory read (cached_p, 4 bytes per query row)
- 1 addition to `l`

Per kept tile, V2 adds:
- 1 reduction (sum of p_j, already partially computed during softmax)
- 1 memory write (cache the sum)

No extra BMM, no extra V load, no extra exp computation. The overhead is
essentially zero compared to V1.

## What V2 Does NOT Fix

V2 fixes the softmax denominator but does NOT add the skipped tile's V
contribution to `acc`. The output is:

```
V1:  output = (sum of kept p*V) / (sum of kept p)           ← wrong l
V2:  output = (sum of kept p*V) / (sum of kept p + cached p) ← correct l
True: output = (sum of all p*V) / (sum of all p)
```

V2 makes the weights of kept tiles more accurate, but the skipped tile's V
contribution is still missing. This is acceptable because:

1. The skipped tiles have small weights (that's why they were skipped)
2. The denominator error in V1 affects ALL kept tiles proportionally
3. Fixing `l` provides most of the accuracy benefit with zero compute cost

## Implementation Path

1. **Python simulation** — Replay captured tile data from
   `examples/diffusers/sparsity/tile_data/` to measure accuracy improvement
   of V2 vs V1 at various sparsity levels. No GPU needed.

2. **Triton kernel** — Implement Flash Attention tile loop with V2 skip logic.
   Supports both forward and backward (backward uses full attention, no skip).

3. **TRT-LLM kernel modification** — Add `l += cached_p` in the skip path of
   `epilogue.h` (line 435-440). Inference only.

## Relation to Other Methods

| | V1 (current) | V2 (this proposal) | VSA (FastVideo) |
|---|---|---|---|
| Skipped tile → `l` | Not updated | Updated with cached p | N/A (top-K, no skip) |
| Skipped tile → `acc` | Not updated | Not updated | Approximated via compression branch |
| Extra storage | None | 1 scalar per tile per head | Compression branch output |
| Extra compute | None | ~0 (1 add per skip) | Block-level attention |
| Temporal cache | None | p from previous step | Skip list from previous step |
