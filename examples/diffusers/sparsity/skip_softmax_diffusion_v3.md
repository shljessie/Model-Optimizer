# Skip-Softmax Diffusion V3: Majority-Vote Tile Skipping

## Summary

V2.51 solved the approximation quality problem (fresh pool-K weights + fresh
v_mean), but LTX-2 still achieves only ~5% skip rate because the **all-128-
rows-must-agree** rule fails on short-sequence, uniform-attention models.

V3 attacks the skip-rate problem directly:
- **Majority vote**: skip a tile if >= 90% of rows agree, instead of requiring
  100% unanimity. The ~10% dissenting rows get the V2.51 pool-K approximation.
- **row_max fix**: always update `row_max` from actual tile scores (not just
  `approx_score`) to prevent cascading staleness for dissenting rows.

No new math tricks on the gap or the scores — just a pragmatic relaxation of
the decision rule.

## Recap: What Each Version Improves

```
V1:    Skip decision works   → skipped tiles contribute zero       → quality loss
V2.5:  Skipped tiles improve → stale cached sum_p + stale v_mean   → staleness problem
V2.51: Staleness fixed       → fresh pool-K + fresh v_mean         → quality restored
V3:    Skip RATE fixed       → majority vote (90% agreement)       → more tiles skipped
```

Where:
- V1-V2.51 improve WHAT HAPPENS to skipped tiles (approximation quality)
- V3 improves HOW MANY tiles get skipped (skip decision rule)

V3 builds on top of V2.51 — it uses the same pool-K + v_mean approximation
for skipped tiles, but fires it much more often.

## The Problem: Why LTX-2 Can't Skip

### Sequence Length Comparison

| Model   | Video self-attn seq_k | KV tiles (BLOCK_N=64) | Attention pattern  |
|---------|-----------------------|-----------------------|--------------------|
| WAN 2.2 | ~19,200               | ~300                  | Peaky/concentrated |
| LTX-2   | 3,840                 | ~60                   | Uniform            |

### The Unanimity Bottleneck

The V1-V2.51 skip decision requires ALL 128 query rows to agree:

```python
tile_row_max = tl.max(scores, 1)                          # [128] per-row max
can_skip = tile_row_max < (row_max + skip_threshold_log2)  # [128] per-row decision
all_skip = tl.min(can_skip.to(tl.int32)) == 1              # ALL must agree
```

On LTX-2:
- Attention is nearly uniform across ~60 KV tiles
- Each tile gets ~1/60 of the weight for most rows
- Within any 128-row block, there is almost always at least one row whose
  score peak falls in the current tile
- That single row vetoes the skip for all 128 rows

Probability of unanimity drops exponentially with row count:
```
P(all 128 agree) = P(single row agrees)^128   (assuming independence)

If P(single row) = 0.977 → P(all 128) ≈ 0.05   (5% skip rate)
If P(single row) = 0.95  → P(all 128) ≈ 0.001  (essentially zero)
```

### Why Majority Vote Helps

With 90% majority (skip if >= 115 of 128 rows agree):
```
P(>= 115 of 128 agree) with P(single row) = 0.95

This is P(Binomial(128, 0.95) >= 115) ≈ 0.98   (98% skip rate!)
```

Even at P(single row) = 0.90:
```
P(>= 115 of 128 agree) with P(single row) = 0.90 ≈ 0.56  (56% skip rate)
```

The unanimity requirement turns a 95% per-row agreement into a 5% tile skip
rate. Majority vote at 90% turns that same 95% per-row agreement into ~98%
tile skip rate. The improvement is dramatic because the `min` operator is
catastrophically sensitive to outliers, while `sum >= threshold` is robust.

## Key Definitions

For clarity, here are the terms used throughout this document:

```
scores = Q_tile @ K_tile^T * qk_scale          # [BLOCK_M, BLOCK_N] = [128, 64]
                                                 # attention scores for this Q-tile vs this KV-tile

tile_row_max = max(scores, dim=1)               # [BLOCK_M] = [128]
                                                 # best score each query row sees in THIS KV tile

row_max                                          # [BLOCK_M] = [128]
                                                 # running cumulative max score each query row has
                                                 # seen across ALL KV tiles processed so far
                                                 # (Flash Attention online softmax state)

gap[i] = tile_row_max[i] - row_max[i]           # always <= 0 for previously-seen tiles
                                                 # how far below the running peak this tile is

exp(gap[i]) = softmax weight upper bound         # THE fundamental relationship
                                                 # gap = -1 → exp(-1) = 0.37 (37% of peak)
                                                 # gap = -3 → exp(-3) = 0.05 (5% of peak)
                                                 # gap = -5 → exp(-5) = 0.007 (0.7% of peak)
```

The gap directly tells you the tile's maximum softmax contribution for that
row. This is an exact mathematical identity, not an approximation.

### Threshold Semantics

The threshold in the kernel comparison
`can_skip = tile_row_max < (row_max + skip_threshold_log2)` is a **raw
log2-space value** added directly to `row_max`. It should be **negative**:

- `-5.0` means "skip tiles whose max is 5 bits below the running max",
  i.e., the tile's peak score is 2^5 = 32x smaller than the row's best.
- **More negative = more conservative** (fewer tiles skipped, higher quality).
- **Less negative / closer to 0 = more aggressive** (more tiles skipped,
  faster but riskier).

There are two CLI flags for setting this value in experiments:

- **`--raw-threshold <value>`** (recommended for experiments): the value
  goes directly into the kernel as `skip_threshold_log2` with **no
  transformation**. What you pass is exactly what the kernel sees.
- **`--threshold <value>`** (normalized): this value is multiplied by
  `log2(seq_k)` before being passed to the kernel, i.e.,
  `skip_threshold_log2 = threshold * log2(seq_k)`. The normalization
  exists for cross-resolution portability (a single `--threshold` value
  produces comparable sparsity across different sequence lengths), but it
  adds a seq_k-dependent transformation that can be confusing during
  debugging. Prefer `--raw-threshold` when tuning on a fixed resolution.

## V3 Algorithm

### Change 1: Majority Vote (90% Agreement)

Replace the unanimity requirement with a supermajority:

```python
# V2.51 (current):
can_skip = tile_row_max < (row_max + skip_threshold_log2)   # [128] per-row
all_skip = tl.min(can_skip.to(tl.int32)) == 1               # ALL 128 must agree

# V3:
can_skip = tile_row_max < (row_max + skip_threshold_log2)   # [128] per-row (SAME gap check)
num_skip = tl.sum(can_skip.to(tl.int32))                     # count agreeing rows
all_skip = num_skip >= MAJORITY_NUMER                         # e.g. >= 115 (90% of 128)
```

The per-row gap check is **identical** to V2.51. The ONLY change is replacing
`min == 1` (unanimity) with `sum >= 115` (90% majority). The gap formula,
the threshold, the log(seq_k) normalization — all unchanged.

MAJORITY_NUMER is a `tl.constexpr` computed at kernel launch:
`MAJORITY_NUMER = int(majority_pct * BLOCK_M)`, e.g. `int(0.9 * 128) = 115`.

### What Happens to Dissenting Rows

When a tile is skipped via majority vote, ~13 rows (10% of 128) disagreed —
they wanted to keep the tile because their gap was above the threshold.

These dissenting rows get the **V2.51 pool-K approximation**, exactly the same
as consenting rows:

```python
# Pool-K runs for ALL 128 rows, not just dissenters:
k_mean = mean(K_tile, dim=tokens)                     # [BLOCK_D], K already in SRAM
approx_score[i] = Q[i] @ k_mean * qk_scale           # [128] per-row approximate scores
p_approx[i] = exp2(approx_score[i] - m_new) * N       # [128] approximate weights
acc += p_approx[:, None] * v_mean[None, :]             # rank-1 update with fresh v_mean
```

For consenting rows (large gap, tile unimportant): the pool-K approximation
is accurate because the tile's contribution is small anyway.

For dissenting rows (small gap, tile matters more): the pool-K approximation
is less accurate. The error depends on how much the actual attention weights
within the tile differ from the pooled-K approximation.

### Change 2: Always Update row_max from Actual Scores

V2.51 updates `row_max` in the skip path using `approx_score` (the pool-K
estimate), which can be lower than the actual `tile_row_max`:

```python
# V2.51 current behavior in skip path:
m_new = tl.maximum(row_max, approx_score)    # approx_score may be < tile_row_max
row_max = m_new                               # row_max may be stale for some rows
```

V3 adds one line to ensure `row_max` reflects the actual scores:

```python
# V3: after the V2.51 pool-K path completes:
row_max = tl.maximum(row_max, tile_row_max)   # always use actual per-row max
```

**Why this matters for majority vote**: without this fix, a dissenting row's
`row_max` may not be updated to reflect the tile it was forced to skip. On
subsequent tiles, its gap computation uses a stale `row_max`, making future
skip decisions for that row increasingly wrong (cascading staleness).

The fix costs one element-wise max on [BLOCK_M] = 128 values — essentially free.

## Per-Tile Flow (V3)

```
Before tile loop:
  Precompute v_mean[j] for all KV tiles              ← same as V2.51

Tile j:
  1. Load K tile into SRAM                             (same as V2.51)
  2. BMM1: scores_j = Q @ K_j^T * qk_scale            (same as V2.51)
  3. tile_row_max = max(scores_j, dim=1)               (same as V2.51)
  4. can_skip = tile_row_max < (row_max + threshold)   (same as V2.51)

  ─── V3 SKIP DECISION (replaces V2.51's unanimity check) ───
  5. num_skip = sum(can_skip)                          ← count agreeing rows
  6. all_skip = (num_skip >= 115)                      ← 90% majority vote
  ────────────────────────────────────────────────────────

  If NOT all_skip (KEEP):
    7-11. Same as V2.51: full softmax, load V, BMM2
    row_max = m_new

  If all_skip (SKIP — V2.51 Pool-K + V3 row_max fix):
    7-11. Same as V2.51: pool-K, p_approx, v_mean load, rank-1 update
    row_max = m_new
    row_max = max(row_max, tile_row_max)               ← V3 FIX
```

### What Changed vs V2.51

| Step | V2.51 | V3 |
|------|-------|----|
| Gap check (per-row) | `tile_row_max < row_max + threshold` | **Same** (unchanged) |
| Agreement rule | `min(can_skip) == 1` (unanimous) | `sum(can_skip) >= 115` (90% majority) |
| row_max update (skip) | From `approx_score` only | `max(m_new, tile_row_max)` (actual scores) |
| Pool-K approximation | Unchanged | Unchanged |
| v_mean precompute | Unchanged | Unchanged |
| Threshold / calibration | Unchanged | Same threshold, same calibrator |

The KEEP path is completely unchanged. The SKIP path's pool-K computation is
unchanged. Only the tile-level decision and the row_max update are modified.

## Kernel Changes

### New Constexpr Flag

```python
APPLY_MAJORITY_VOTE: tl.constexpr   # False = V2.51 (unanimous), True = V3 (majority)
MAJORITY_NUMER: tl.constexpr         # int(0.9 * BLOCK_M), e.g. 115
```

### Pseudocode Diff in _attn_fwd_body

```python
if APPLY_SKIP_SOFTMAX:
    tile_row_max = tl.max(scores, 1)
    can_skip = tile_row_max < (row_max + skip_threshold_log2)    # SAME as V2.51

    if APPLY_MAJORITY_VOTE:
        # V3: majority vote
        num_skip = tl.sum(can_skip.to(tl.int32))
        all_skip = num_skip >= MAJORITY_NUMER
    else:
        # V2.51: unanimity
        all_skip = tl.min(can_skip.to(tl.int32)) == 1

    if not all_skip:
        # KEEP: unchanged
        ...
    elif APPLY_SKIP_V25:
        # SKIP: V2.51 pool-K path, unchanged
        ...
        # V3 FIX: update row_max from actual scores
        if APPLY_MAJORITY_VOTE:
            row_max = tl.maximum(row_max, tile_row_max)
```

That's it. The diff is ~5 lines in the kernel.

## Cost Analysis

### Per-Tile Overhead (V3 vs V2.51)

| Operation | V2.51 | V3 | Difference |
|-----------|-------|----|------------|
| Gap check (per-row) | Same | Same | 0 |
| Decision reduction | `tl.min` on [128] (~5 cycles) | `tl.sum` on [128] (~20 cycles) | +15 cycles |
| Scalar comparison | `== 1` | `>= 115` | 0 |
| row_max update (skip path) | From approx_score | +1 element-wise max | +~2 cycles |
| **Total overhead per tile** | | | **~17 cycles** |

This is negligible. A KEPT tile costs ~2000+ cycles (full softmax + V load
from HBM + BMM2). The V3 decision overhead is <1% of the kept-tile cost.

### No New Registers, No New Memory

| Resource | V2.51 | V3 | Change |
|----------|-------|----|--------|
| Extra registers per thread | 0 | 0 | None |
| L2 cache usage | v_mean (0.5 MB) | Same | None |
| HBM traffic per tile | K load + conditional V load | Same | None |
| Precompute kernels | v_mean | Same | None |

V3 adds zero memory overhead. It only changes the reduction operation on
`can_skip` (min → sum) and adds one element-wise max for row_max.

### Expected Skip Rate Improvement (LTX-2)

| | V2.51 (unanimous) | V3 (90% majority) |
|---|---|---|
| Skip rate | ~5% | **~30-50%** (depends on threshold) |
| Tiles skipped (of 60) | ~3 | **~18-30** |
| V loads saved per attention call | 48 KB | **288-480 KB** |
| BMM2 FLOPs saved | 6M | **36-60M** |
| Estimated attention speedup | ~2% | **~15-25%** |

Exact numbers depend on the threshold and the actual gap distribution of LTX-2.
These should be validated on captured Q/K/V data before deployment.

## Error Analysis

### Where Error Comes From

V3 introduces error ONLY through dissenting rows in skipped tiles. For a tile
skipped via majority vote where row `i` dissented:

1. **Row i's gap is above threshold** — the tile has non-negligible softmax
   weight for row i
2. **Row i gets pool-K approximation** instead of exact attention
3. **The pool-K error depends on intra-tile K variance**

### Bounding the Dissenting Row Error

For a dissenting row with gap `g` (where `g > threshold`, i.e., less negative):

The tile's maximum softmax contribution for that row is `exp(g)`.

The pool-K approximation error for that row is bounded by:

```
|error_i| <= exp(g) * BLOCK_N * max_j ||V[j] - v_mean||
```

Since dissenting rows have `g` close to the threshold (they are borderline —
they wanted to keep but were outvoted), and the threshold is already set to
ensure small softmax contributions, the error per dissenting row is bounded.

At 90% majority, at most 13 rows dissent per tile. The total output error
is the sum of per-row errors, which scales linearly with the number of
dissenters.

### Pool-K Underestimation (Jensen's Inequality)

Pool-K always underestimates the true total attention weight:

```
true:   sum_j exp(Q[i] @ K[j])
approx: N * exp(Q[i] @ mean(K))

By Jensen (exp is convex):  mean(exp(x)) >= exp(mean(x))
Therefore:                  true >= approx    (always)
```

The underestimation factor is approximately `exp(sigma^2 / 2)` where
`sigma^2 = Var(Q[i] @ K[j])` within the tile.

For consenting rows (large gap): the tile's total weight is tiny anyway,
so the underestimation doesn't matter.

For dissenting rows (small gap): the underestimation is more significant.
This is the main quality risk of V3 and should be monitored empirically.

### Cascading Error Prevention

The row_max fix (Change 2) prevents a subtle compounding problem:

Without the fix, a dissenting row's `row_max` may not reflect the skipped
tile's true maximum. This makes the row appear to have a larger gap in
subsequent tiles (stale row_max → artificially inflated gap → more skipping),
creating a positive feedback loop where the row's attention becomes
increasingly distorted.

The fix ensures `row_max` always reflects reality, breaking this feedback loop.

## Calibration

### No Changes Needed

V3 uses the **same per-row gap check** as V2.51:

```python
can_skip[i] = tile_row_max[i] < row_max[i] + skip_threshold_log2
```

The gap formula, the `gap / log(seq_k)` normalization, and the percentile
threshold computation are all identical. The only difference is that V3
requires 90% of rows to pass the check instead of 100%.

**The existing `PercentileThresholdCalibrator` works unchanged.** It collects
per-row normalized gaps and computes a threshold at the target percentile.
The same threshold produces higher tile-level sparsity under majority vote
because the tile decision is less conservative.

### Threshold Reuse

A threshold calibrated for V2.51 can be reused directly in V3. The per-row
gap semantics are identical — only the tile-level aggregation changes.

However, because majority vote increases the effective tile-level sparsity
for any given threshold, you may want to use a **more conservative threshold**
(lower target sparsity) with V3 to match V2.51's quality at higher speed.

### Recommended Calibration Strategy

1. Calibrate with V2.51's existing flow (unchanged)
2. Deploy with V3's majority vote using the same threshold
3. Measure actual tile-level sparsity and quality
4. If quality degrades, either:
   - Reduce `target_sparsity` in calibration (more conservative threshold)
   - Increase `MAJORITY_NUMER` (e.g., 120 instead of 115 = 94% majority)

## Tunable Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `skip_threshold_log2` | Calibrated (same as V2.51) | Same as V2.51 | Per-row gap threshold |
| `MAJORITY_NUMER` | 115 (90% of 128) | [96, 128] | Tile-level agreement requirement |
| `skip_first_last` | 2 | [0, 5] | Layers excluded from sparsity |

### Tuning MAJORITY_NUMER

| Value | Agreement % | Max dissenters | Behavior |
|-------|-------------|----------------|----------|
| 128 | 100% | 0 | Identical to V2.51 (unanimous) |
| 122 | 95% | 6 | Conservative V3, minimal quality risk |
| 115 | 90% | 13 | **Recommended**: good skip rate, bounded error |
| 96 | 75% | 32 | Aggressive, check quality carefully |
| < 96 | < 75% | > 32 | Not recommended — too many dissenters |

The right value depends on the model and quality requirements. Start with
115 and adjust based on empirical quality measurements.

## Comparison of All Versions

| | V1 | V2.51 | V3 |
|---|---|---|---|
| **Skip decision** | Absolute gap, unanimous | Absolute gap, unanimous | **Absolute gap, 90% majority** |
| **Gap check per-row** | Same | Same | **Same** |
| **Tile-level rule** | `min == 1` | `min == 1` | **`sum >= 115`** |
| **Skip approximation** | Zero | Pool-K + v_mean | Pool-K + v_mean (same as V2.51) |
| **LTX-2 skip rate** | ~5% | ~5% | **~30-50%** |
| **WAN 2.2 skip rate** | ~75% | ~75% | ~75% (already good, unchanged) |
| **row_max update (skip)** | Not updated | From approx_score | **From actual tile_row_max** |
| **Extra registers** | 0 | 0 | 0 |
| **Extra per-tile cycles** | 0 | 0 | ~17 |
| **Cache** | None | v_mean (0.5 MB) | v_mean (0.5 MB, unchanged) |
| **Threshold** | Same | Same | **Same** (reusable from V2.51) |
| **Quality risk** | High (zero approx) | Low (fresh approx) | Low-medium (dissenting rows) |

## Files to Modify

| File | Changes |
|------|---------|
| `modelopt/torch/kernels/triton_fa.py` | Add `APPLY_MAJORITY_VOTE` and `MAJORITY_NUMER` constexpr to `_attn_fwd_body`. Replace `tl.min` with `tl.sum` + comparison when flag is set. Add `row_max = max(row_max, tile_row_max)` in skip path. Pass flags through `_attention_v25_forward` and `attention()`. |
| `modelopt/torch/sparsity/attention_sparsity/methods/triton_skip_softmax_diffusion.py` | Add `enable_v3: bool` and `majority_pct: float` (default 0.9) to config. Compute `MAJORITY_NUMER = int(majority_pct * BLOCK_M)`. Pass to kernel. |
| `modelopt/torch/sparsity/attention_sparsity/kernels/ltx_triton_attention.py` | Thread `enable_v3` and `majority_pct` through to `attention()`. |
| `modelopt/torch/sparsity/attention_sparsity/kernels/diffusers_triton_attention.py` | Same as above. |
| `modelopt/torch/sparsity/attention_sparsity/config.py` | Add `enable_v3: bool = False` and `majority_pct: float = 0.9` to `SparseAttentionAttributeConfig`. |
| `examples/diffusers/sparsity/ltx2_skip_softmax.py` | Add `--enable-v3` and `--majority-pct` CLI flags, wire to config. |
| `tests/gpu/torch/sparsity/attention_sparsity/test_triton_fa.py` | Test majority vote counting, row_max update, output closeness to dense. |

## Validation Plan

1. **Offline analysis**: Load captured Q/K/V from `experiments/attn_input/step_*.pt`,
   simulate V3 skip decisions (just change `min` to `sum >= 115`), measure
   actual skip rate improvement and approximation error vs dense attention.

2. **Unit tests**: Verify `tl.sum(can_skip) >= N` produces correct skip decisions
   for known inputs. Verify `row_max` update prevents cascading staleness.

3. **End-to-end**: Run `ltx2_skip_softmax.py` with `--enable-v3`, compare:
   - Wall-clock time vs V2.51 and Triton baseline
   - Video quality (visual inspection + optional FVD/CLIP metrics)
   - Per-layer sparsity summary

4. **Sweep MAJORITY_NUMER**: Test 128 (V2.51), 122, 115, 108, 96 to find the
   quality-speed Pareto frontier for LTX-2.

5. **Backward compatibility**: Verify that `APPLY_MAJORITY_VOTE=False` produces
   bit-identical results to V2.51 (no regression).

6. **WAN 2.2 check**: Verify V3 does not degrade WAN 2.2 performance or quality
   (it should be equivalent since WAN already achieves high skip rates with
   unanimity).

## Future Work (V3.1+)

### Variance Correction for Dissenting Rows

The pool-K approximation can be improved with a second-order correction:

```python
# Precompute k_var[d] = Var(K_tile, dim=tokens) alongside v_mean
# Per skipped tile:
sigma_sq[i] = sum(Q[i]**2 * k_var)          # per-row score variance estimate
approx_score[i] += sigma_sq[i] / 2           # second-order correction (Jensen)
```

This corrects the systematic underestimation by pool-K, especially for
dissenting rows. Cost: one extra [BLOCK_D] vector precomputed per tile
(+512 bytes L2) and one extra dot product per row (+16K FLOPs per tile).

### Adaptive Majority Percentage

Instead of a fixed 90%, use a per-layer or per-denoising-step majority
percentage. Early layers and early denoising steps (noisier, more uniform
attention) may tolerate a lower majority (e.g., 75%), while later layers
and steps (sharper attention) should use higher majority (e.g., 95%).

### Per-Head Thresholds

Different attention heads have different sparsity patterns. Calibrating a
per-head threshold (instead of per-layer) could further improve the
quality-speed tradeoff without changing the majority vote mechanism.
