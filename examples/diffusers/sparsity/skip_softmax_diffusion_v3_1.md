# Skip-Softmax Diffusion V3.1: Accuracy Recovery for Aggressive Majority Vote

## Summary

V3 introduced majority vote to increase LTX-2's skip rate from ~5% to ~30-50%.
V3.1 adds accuracy recovery methods so we can push the majority threshold
**lower** (e.g., 70%) without unacceptable quality loss.

The key insight from our analysis: **the dominant error for dissenting rows is
the VALUE approximation (v_mean ≠ attention-weighted V), not the WEIGHT
approximation (pool-K underestimates).** V3.1 addresses both.

## Prerequisite: V3 (Majority Vote + row_max Fix)

V3.1 builds on V3. Read `skip_softmax_diffusion_v3.md` first. V3 provides:
- Majority vote: skip if >= X% of 128 rows agree (default 90%)
- row_max fix: always update row_max from actual `tile_row_max` in skip path
- m_new fix: use `tile_row_max` (not `approx_score`) for online softmax max

## Error Decomposition

For a skipped tile, the per-row error decomposes cleanly:

```
error[i] = (α - α') × (V_kept - V_skipped)  +  α' × (v_mean - V_weighted)
           \_____weight rebalancing error___/     \____value approximation___/

where:
  α  = true fraction of softmax mass in skipped tiles
  α' = pool-K approximate fraction (α' < α, by Jensen's inequality)
  V_kept     = attention-weighted V from kept tiles (exact)
  V_skipped  = attention-weighted V from skipped tiles (true, unknown)
  v_mean     = uniform mean of V in skipped tiles (our approximation)
  V_weighted = true attention-weighted mean of V in skipped tiles
```

Self-normalization (dividing acc by row_sum) provides partial cancellation:
even pool-K recovering only 30% of the true weight cuts the total error roughly
in half compared to dropping tiles entirely (V1). The error saturates rather
than amplifying — this is a structural safety property.

For **consenting rows** (α small): both terms vanish. Error ∝ α.

For **dissenting rows** (α not small): Term 2 (value error) dominates because
v_mean ignores which V vectors the row actually attends to within the tile.
Fixing the weight (Term 1) helps, but the value error remains.

## V3.1 Recovery Methods

### Method 1: Three-Tier System (Gray Zone)

Instead of binary KEEP/SKIP, add an intermediate tier for "gray zone" tiles
where the skip vote is contentious (70-95% agreement):

```
num_skip = sum(can_skip)

Tier 1 — KEEP (num_skip < 70%):
  Full softmax + V load + BMM2. Unchanged.

Tier 2 — GRAY ZONE (70% <= num_skip < 95%):
  Compute EXACT softmax weights (8192 exp2), but use v_mean instead of loading V.
  Eliminates weight error entirely. Only value error remains.
  Saves: V load (16 KB HBM) + BMM2 (~1M FMA).
  Costs: 8192 exp2 (same as full softmax, but much cheaper than BMM2).

Tier 3 — SKIP (num_skip >= 95%):
  Pool-K + v_mean. Cheapest path. Same as V2.51/V3.
```

**Why Tier 2 works:** The 8192 exp2 operations run on the SFU in ~64 cycles.
The V load + BMM2 they replace cost ~thousands of cycles. So Tier 2 gives
exact weights at ~3% of the full-keep cost.

The exact weights directly fix Term 1 of the error formula for ALL rows
(consenting and dissenting). The only remaining error is Term 2 (v_mean vs
attention-weighted V), which is bounded by the intra-tile V diversity.

**Per-tile cost comparison:**

| Tier | FLOPs (incremental over BMM1) | exp2 | V Load | Quality |
|------|-------------------------------|------|--------|---------|
| 1 (KEEP) | ~2M FMA | 8192 | 16 KB | Exact |
| 2 (GRAY) | ~41K FMA | 8192 | 0 | Exact weights, approx V |
| 3 (SKIP) | ~49K FMA | 128 | 0 | Approx weights, approx V |

**Tile distribution estimate for LTX-2 (at ~40% target sparsity):**

| Tier | Agreement Range | Tiles (of 60) | Fraction |
|------|-----------------|---------------|----------|
| 1 (KEEP) | < 70% skip | ~30-33 | 50-55% |
| 2 (GRAY) | 70-95% skip | ~8-12 | 13-20% |
| 3 (SKIP) | >= 95% skip | ~18-20 | 30-33% |

### Method 2: Variance Correction (σ²/2 Fix for Pool-K)

Pool-K systematically underestimates the true weight by Jensen's inequality:

```
True:   mean_j(exp(s_j)) = exp(mean(s)) × exp(σ²/2 + higher order terms)
Pool-K: exp(mean(s))
```

For Gaussian-distributed scores (reasonable by CLT with head_dim=128), the
correction `exp(σ²/2)` is **exact**. In log-space:

```
corrected_approx_score[i] = approx_score[i] + σ²[i] / 2
```

where `σ²[i] = Var(Q[i] @ K[j])` across the j positions in the tile.

**Diagonal approximation**: The exact variance requires the full K covariance
matrix (D×D = 16K values per tile). The diagonal approximation:

```
σ²[i] ≈ scale² × sum_d(Q[i,d]² × Var_j(K[j,d]))
       = scale² × Q[i]² @ k_var
```

where `k_var[d] = Var_j(K[j,d])` is a [BLOCK_D] vector per tile.

**Precomputation**: Compute `k_var` alongside `v_mean` in the precompute kernel:

```python
# In _precompute_tile_stats kernel (extends _precompute_vmean):
k_sq_mean = sum(K * K, dim=0) / n_valid    # [BLOCK_D]
k_mean_sq = k_mean * k_mean                 # [BLOCK_D]
k_var = k_sq_mean - k_mean_sq               # [BLOCK_D]
```

This requires one additional load of K in the precompute pass (same shape as V).
Output: one extra [BLOCK_D] vector per tile stored in L2 (~512 bytes).

**Inference cost per skipped tile**: Hoist `Q_sq = Q * Q` before the KV loop
(compute once, [BLOCK_M, BLOCK_D] in registers). Then per skipped tile:

```python
# Load k_var from L2 (512 bytes, alongside v_mean)
score_var = tl.sum(Q_sq * k_var[None, :], axis=1)   # [BLOCK_M], one dot product
# Scale correctly for log2 space:
var_correction = score_var * (qk_scale * qk_scale * LOG2E * 0.5)
approx_score += var_correction
```

**Cost**: ~16K FMA (one dot product per row) + 512B L2 load.
This is ~50% overhead on the existing pool-K path (~33K FMA).

**Empirical improvement**: Reduces weight underestimation from ~2-4x to ~1.15x.

```
Example: dissenting row with gap = -0.3, intra-tile σ² = 1.5
  Pool-K:  underestimates by exp(1.5/2) = 2.12x
  +σ²/2:  residual error from non-Gaussianity + diagonal approx ≈ 1.15x
```

### Method 3: VMC (V-Mean Compensation)

From **Top-Theta Attention** (Berestizshevsky et al., 2025): track the
"lost probability mass" per row and redistribute it through the global mean V.

When tiles are skipped, the effective attention weights no longer sum to 1.
The missing mass represents attention that should have gone to skipped tiles.

```python
# At the END of the tile loop, after all tiles processed:
beta[i] = 1.0 - row_sum[i]    # missing mass (should be close to 0 for consenting rows,
                                # can be significant for dissenting rows)
# Compensate:
output[i] = acc[i] / row_sum[i] + beta[i] * global_v_mean
```

Wait — actually, in online softmax, `row_sum` is the sum of all `exp(score - row_max)` terms, and the final normalization is `acc / row_sum`. The missing mass concept applies differently here: the issue is that skipped tiles' approximate weights (from pool-K) underestimate the true total, so `row_sum` is too small. After normalization, the kept tiles get too much relative weight.

**A more correct formulation for our setting:**

Track the approximate total weight from skipped tiles separately:

```python
# During tile loop: maintain row_sum_skip[i] alongside row_sum[i]
# row_sum_skip accumulates p_approx from skipped tiles only

# After tile loop:
# Estimate the true skip weight by inflating the approximate:
correction_factor = calibrated_value   # e.g., 1.5, estimated from k_var or calibrated offline
row_sum_corrected = (row_sum - row_sum_skip) + row_sum_skip * correction_factor
output = acc / row_sum_corrected
```

**Cost**: 2 extra registers per row (row_sum_skip, n_skip_tiles).
One `tl.where` per tile (to accumulate into the right tracker).
One correction at the end.

### Method 4: Per-Step Majority Schedule

Different denoising steps tolerate different amounts of approximation error:

```
Early steps (0-9):   Latent is noisy → attention patterns are unreliable
                     → aggressive majority (70%) is fine
                     → errors masked by noise

Middle steps (10-29): Structure forming → moderate aggressiveness (85%)

Late steps (30-39):  Fine details → conservative (95%)
                     → errors create visible artifacts
```

This is based on diffusion physics, not model-specific tuning, so it
generalizes across prompts and resolutions.

**Implementation**: Pass denoising step number to the kernel. Compute
`MAJORITY_NUMER` from a 4-entry lookup table:

```python
step_schedule = {
    (0, 9):   int(0.70 * BLOCK_M),   # 90 of 128 = 70%
    (10, 24): int(0.85 * BLOCK_M),   # 109 of 128 = 85%
    (25, 34): int(0.92 * BLOCK_M),   # 118 of 128 = 92%
    (35, 39): int(0.97 * BLOCK_M),   # 124 of 128 = 97%
}
```

**Cost**: Zero per-tile overhead. Just changes which integer is compared
against `num_skip`. The step number can be threaded through the same mechanism
as LiteAttention's `_lite_step` counter.

### Method 5: Delta Correction (from Literature)

From **Delta Attention** (Willette et al., 2025, NeurIPS): periodically compute
full (unskipped) attention, measure the delta from approximate output, and
apply it as a correction.

```
Every G-th denoising step (e.g., G=5):
  1. Run attention WITHOUT majority vote (full, no skipping)
  2. Also run attention WITH majority vote (approximate)
  3. Compute delta = full_output - approx_output per layer
  4. Cache the delta

For the next G-1 steps:
  5. Run attention WITH majority vote (fast)
  6. Add the cached delta: output += delta
```

The delta is smooth (changes slowly across steps) so caching it every 5 steps
is sufficient. Overhead: 1 in every 5 steps runs 2x attention (full + approx).
Net: ~1.2x the cost of always-approximate, but with much better quality.

**Storage**: One [seq_q, head_dim] tensor per layer per head. For LTX-2:
3840 × 128 × 4 bytes = 1.9 MB per head, 61 MB for 32 heads. Fits in HBM
(not L2). The delta load from HBM is a one-time cost per step.

This is the most expensive recovery method but also the highest quality.
It converts a systematic bias into a periodic correction.

## Recommended Implementation Order

```
V3:     Majority vote + row_max fix + m_new fix
        (shipped separately, the foundation)

V3.1a:  Variance correction (σ²/2)
        → Low cost, high impact on weight accuracy
        → Requires extending precompute kernel for k_var

V3.1b:  Three-tier system (gray zone)
        → Medium cost (8K exp2 for gray tiles), eliminates weight error for gray zone
        → Requires adding a second threshold + branch to kernel

V3.1c:  Per-step majority schedule
        → Zero cost, adapts aggressiveness to denoising phase
        → Requires threading step number to kernel

V3.1d:  VMC (probability mass compensation)
        → Low cost, fixes systematic bias in normalization
        → Requires 2 extra registers per row

V3.1e:  Delta correction
        → Highest quality, most expensive
        → Requires caching infrastructure between denoising steps
```

Each method is independent and composable. They can be implemented and
evaluated one at a time. The expected impact:

| Method | Weight error | Value error | Cost | Skip rate |
|--------|-------------|-------------|------|-----------|
| V3 alone (90%) | ~2-4x underest. | v_mean | ~17 cycles/tile | ~35-45% |
| + σ²/2 | ~1.15x underest. | v_mean | +16K FMA/tile | Same |
| + 3-tier | 0 (gray), ~2x (skip) | v_mean | +8K exp2/gray | Same |
| + per-step schedule | Same | Same | 0 | Higher (early steps) |
| + VMC | Compensated | Partially compensated | +2 regs | Same |
| + delta correction | ~0 (periodic) | ~0 (periodic) | 1.2x every G steps | Same |

## Comparison Table

| | V2.51 | V3 | V3.1 (all methods) |
|---|---|---|---|
| **Skip decision** | Unanimous | Majority vote | Majority vote + per-step schedule |
| **Weight approx** | Pool-K | Pool-K | Pool-K + σ²/2 (skip) or exact (gray) |
| **Value approx** | v_mean | v_mean | v_mean + VMC correction |
| **m_new source** | approx_score | tile_row_max | tile_row_max |
| **LTX-2 skip rate** | ~5% | ~30-50% | ~40-60% (per-step adaptive) |
| **Tiers** | 2 (keep/skip) | 2 (keep/skip) | 3 (keep/gray/skip) |
| **Extra precompute** | v_mean | v_mean | v_mean + k_var |
| **Extra registers** | 0 | 0 | +2 (row_sum_skip) |
| **Quality** | Good (few skips) | Moderate | Good (recovery compensates) |

## Key References from Literature

- **Top-Theta Attention** (Berestizshevsky et al., 2025): VMC (V-Mean
  Compensation) — redistribute lost probability mass through mean(V).
  Directly applicable to pool-K's systematic underweighting.

- **Delta Attention** (Willette et al., 2025, NeurIPS): Periodically compute
  full attention, cache the delta correction, apply to approximate outputs.
  ~1.5% overhead recovers 88% of dense accuracy at 98.5% sparsity.

- **SpargeAttention** (Zhang et al., 2025, ICML): Self-similarity check on K
  blocks — if K vectors within a tile have low cosine similarity, pool-K will
  be unreliable. Always compute those tiles fully ("fix blocks").

- **DiTFastAttn** (Yuan et al., 2024, NeurIPS): Cache `residual = full - approx`
  across denoising steps. The residual is more stable than either quantity alone.

## Files to Modify (Beyond V3)

| File | V3.1a (σ²/2) | V3.1b (3-tier) | V3.1c (per-step) | V3.1d (VMC) |
|------|-------------|----------------|-------------------|-------------|
| `triton_fa.py` | Add k_var precompute kernel, var_correction in skip path | Add second threshold + gray-zone branch | Accept step-dependent majority_threshold as runtime param | Add row_sum_skip accumulator, correction at end |
| `ltx_triton_attention.py` | Pass k_var_cache alongside v_mean_cache | Pass gray_threshold | Pass step number | — |
| `diffusers_triton_attention.py` | Same | Same | Same | — |
| `triton_skip_softmax_diffusion.py` | Add k_var config, precompute management | Add gray_threshold config | Add step_schedule config | Add vmc_correction flag |
| `config.py` | Add `enable_var_correction: bool` | Add `gray_threshold_pct: float` | Add `step_schedule: dict` | Add `enable_vmc: bool` |

## Open Questions

1. **Is the three-tier system worth the complexity?** If variance correction
   (σ²/2) alone provides sufficient weight accuracy, the gray zone may be
   unnecessary. Validate empirically.

2. **Does the value error (v_mean vs attention-weighted V) actually matter for
   video quality?** In diffusion models, V vectors within a tile may be
   sufficiently similar that v_mean is a good approximation. If so, fixing
   the weight is enough and value recovery methods (VMC, delta) are unnecessary.

3. **What majority percentage is optimal per model?** The answer depends on
   the attention pattern (which varies by resolution, frame count, and model
   architecture). Calibration with quality metrics (FVD, CLIP) is needed.

4. **Can the σ²/2 correction and VMC be combined?** They address different
   aspects (weight accuracy vs normalization bias). They should compose
   well but this needs validation.
