# Skip-Softmax Diffusion V2.5: Cached Denominator + Approximate V

## Summary

V2 fixes the softmax denominator (`l`) for skipped tiles but leaves the
numerator (`acc`) missing their contribution. V2.5 adds an approximate V
contribution for skipped tiles using cached `v_mean` from when the tile was
last computed.

## Recap: What Each Version Does

```
True:  output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂ + 0.007*v₃) / 1.027
V1:    output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂)             / 1.020   ← wrong l, missing v₃
V2:    output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂)             / 1.026   ← better l, missing v₃
V2.5:  output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂ + 0.006*v₃') / 1.026   ← better l, approximate v₃
```

Where `v₃'` is the cached `v_mean` of tile 3 from the previous denoising step.

## Algorithm

### What We Cache

For **every tile** at every denoising step (when the tile is kept / not skipped):

1. `sum_p`: sum of softmax weights in the tile — one scalar per query row
2. `v_mean`: mean of the 128 V vectors in the tile — one d-dimensional vector

### Per-Tile Flow

```
Tile j:
  1. BMM1: scores_j = Q @ K_j^T                (always computed)
  2. block_max_j = max(scores_j)                (always computed)
  3. gap = cummax - block_max_j

  If gap < threshold (KEEP):
    4. Compute softmax: p_j = exp(scores_j - m)
    5. Update l: l += sum(p_j)
    6. Compute BMM2: acc += p_j @ V_j
    7. Cache sum_p: store sum(p_j)                     ← for future skipped steps
    8. Cache v_mean: store mean(V_j)                   ← V is already in SRAM, cheap reduction

  If gap >= threshold (SKIP):
    4. Load cached sum_p and v_mean from last kept step
    5. Update l: l += cached_sum_p                     ← fix denominator
    6. Update acc: acc += cached_sum_p * cached_v_mean ← approximate V contribution
    7. Skip: softmax exp (128 exponentials), BMM2 (128×d matmul), V load from HBM
```

### Why This Works for Diffusion

Diffusion models have **temporal coherence** — attention patterns and value
vectors change gradually between denoising steps. A tile's `sum(p)` and
`v_mean` at step t are close to their values at step t-1. This makes the
cached values good approximations.

Tiles that get skipped tend to stay unimportant across consecutive steps, so
the cached values remain relevant. Tiles that flip between kept and skipped
get their cache refreshed on kept steps.

### First Denoising Step

At step 0, there is no cache. Run full attention (no skipping) to populate
the cache for all tiles. This matches LiteAttention's approach.

### Stale Cache

If a tile is skipped for many consecutive steps, its cache gets stale (it
was last updated at the most recent kept step). This is acceptable because:

1. Tiles skipped for many steps have very small `sum_p` — their contribution
   is small regardless of how stale `v_mean` is
2. Diffusion V vectors evolve slowly — even a cache from 5 steps ago is a
   reasonable approximation
3. The alternative (V1) contributes nothing at all

## Concrete Example with Tile Loop

Scores: `[3, 1, 7, 2, 6]`, skip tile 3 (score 2).

Previous step's cache for tile 3: `sum_p = 0.006`, `v_mean = v₃'`

```
Tile 0 (score 3, KEEP):
  m = 3, p₀ = 1.0, l = 1.0
  acc = 1.0 * v₀
  cache: sum_p=1.0, v_mean=mean(V₀)

Tile 1 (score 1, KEEP):
  m = 3, p₁ = 0.135, l = 1.135
  acc = 1.0*v₀ + 0.135*v₁
  cache: sum_p=0.135, v_mean=mean(V₁)

Tile 2 (score 7, KEEP):
  m = 7, correction = 0.018
  l = 1.135 * 0.018 + 1.0 = 1.020
  acc = 0.018*v₀ + 0.002*v₁ + 1.0*v₂
  cache: sum_p=1.0, v_mean=mean(V₂)

Tile 3 (score 2, SKIP):
  BMM1 → block_max = 2, gap = 7 - 2 = 5 → skip!
  l = 1.020 + 0.006 = 1.026                           ← cached sum_p
  acc = 0.018*v₀ + 0.002*v₁ + 1.0*v₂ + 0.006*v₃'     ← cached sum_p * cached v_mean
  No V load, no softmax exp, no BMM2

Tile 4 (score 6, KEEP):
  m = 7, correction = 1.0 (m unchanged)
  p₄ = e^(6-7) = 0.368
  l = 1.026 + 0.368 = 1.394
  acc = 0.018*v₀ + 0.002*v₁ + 1.0*v₂ + 0.006*v₃' + 0.368*v₄
  cache: sum_p=0.368, v_mean=mean(V₄)

Output = acc / l
```

## Cost Analysis

### Cache Storage

Per tile, per head:
- `sum_p`: one scalar per query row = `br` floats = `128 × 4 = 512 bytes`
- `v_mean`: one d-dim vector = `d` floats = `128 × 4 = 512 bytes`
- Total per tile per head: `1 KB`

For LTX-2 (33 tiles, 32 heads): `33 × 32 × 1 KB ≈ 1 MB`. Negligible.

### Compute Overhead

Per **kept** tile (extra work on top of normal attention):
- `sum(p_j)`: reduction over 128 elements per query row. Cheap — p is
  already computed for BMM2.
- `mean(V_j)`: reduction over 128 vectors. Cheap — V is already in SRAM
  for BMM2.
- Two memory writes to cache. Negligible bandwidth.

Per **skipped** tile (instead of doing nothing):
- Two memory reads from cache (sum_p + v_mean). Small bandwidth.
- One addition to `l`: `l += cached_sum_p`. One scalar op.
- One scalar-vector multiply + addition to `acc`: `acc += cached_sum_p * cached_v_mean`.
  This is `d` multiply-adds (e.g., 128). Negligible compared to BMM2
  which is `128 × d` multiply-adds.

### Savings vs Full Attention (per skipped tile)

| Operation | Full | V2.5 |
|-----------|------|------|
| BMM1 (Q @ K^T) | 128 × d MADs | 128 × d MADs (same) |
| Softmax exp | 128 exponentials | 0 |
| BMM2 (p @ V) | 128 × d MADs | d MADs (scalar × vector) |
| V load from HBM | 128 × d × 2 bytes | d × 4 bytes (cached v_mean) |

BMM2 goes from `128 × d` to `d` — a 128x reduction. V load goes from
reading 128 vectors from HBM to reading 1 vector from cache.

## Comparison of All Versions

| | V1 | V2 | V2.5 |
|---|---|---|---|
| **Skipped tile → l** | Not updated | += cached sum_p | += cached sum_p |
| **Skipped tile → acc** | Not updated | Not updated | += sum_p * v_mean |
| **Cache per tile** | None | 1 scalar/row | 1 scalar/row + 1 vector |
| **Cache size (LTX-2)** | 0 | 0.5 MB | 1 MB |
| **Extra compute (kept)** | None | sum reduction | sum + mean reduction |
| **Extra compute (skip)** | None | 1 add | 1 add + d MADs |
| **V load for skip** | No | No | No (cached) |
| **Accuracy** | Wrong l, missing acc | Better l, missing acc | Better l, approximate acc |

## Relation to VSA

V2.5 arrives at a similar idea to VSA's compression branch from a different
direction:

| | V2.5 | VSA |
|---|---|---|
| When to approximate | After skip decision (known unimportant tiles) | Before attention (all tiles) |
| V approximation | mean(V) cached from last kept step | mean(V) computed fresh each step |
| Weight | Real sum(p) from softmax | Block-level attention score |
| Blending | Fused into Flash Attention's l and acc | Separate branch with learned gate |
| Training needed | No | Yes (gate weight) |

## Implementation Path

1. **Python simulation**: Replay tile data to measure accuracy of V1 vs V2 vs
   V2.5 at various sparsity levels (30%, 50%, 70%, 90%). Use captured tile
   data from `examples/diffusers/sparsity/tile_data/`.

2. **Triton kernel prototype**: Implement Flash Attention tile loop with V2.5
   logic. Forward only first, add backward later (backward uses full attention).

3. **TRT-LLM kernel**: Modify skip path in `epilogue.h` to read cached sum_p
   and v_mean, update l and acc.
