# Skip-Softmax Diffusion V2.51: Pool-K Fresh Weights + Precomputed Fresh V-Mean

## Summary

V2.5 used stale cached `sum_p` and `v_mean` from previous denoising steps for
skipped tiles. V2.51 replaces both with fresh values computed every step:
- **p_approx**: fresh per-row weight from pooled K (K already in SRAM from BMM1)
- **v_mean**: fresh from precompute kernel (loads V once upfront for all tiles)

## Recap: What Each Version Does

```
True:  output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂ + 0.007*v₃) / 1.027
V1:    output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂)             / 1.020   ← wrong l, missing v₃
V2.5:  output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂ + 0.006*v₃') / 1.026  ← stale p, stale v_mean
V2.51: output = (0.018*v₀ + 0.002*v₁ + 1.0*v₂ + 0.007*v̄₃) / 1.027   ← fresh p, fresh v_mean
```

Where:
- `v₃'` = stale cached v_mean from previous denoising step
- `v̄₃` = fresh mean(V₃) precomputed at current step
- `0.006` = stale cached sum_p from previous step
- `0.007` = fresh p_approx from `Q @ mean(K₃)`

## Algorithm

### Per-Tile Flow (Current Implementation)

```
Before tile loop:
  Precompute v_mean[j] = mean(V_tile_j) for all KV tiles  ← one V load pass

Tile j:
  1. Load K tile into SRAM (always needed)
  2. BMM1: scores_j = Q @ K_j^T * qk_scale       (always computed)
  3. tile_row_max = max(scores_j, dim=1)           (always computed)
  4. gap check: can_skip = tile_row_max < (row_max + threshold_log2)
  5. all_skip = all rows agree tile is negligible

  If NOT all_skip (KEEP):
    6. m_new = max(row_max, tile_row_max)
    7. p = exp2(scores - m_new)                    ← full BLOCK_M × BLOCK_N exp2
    8. Online softmax correction: row_sum, acc rescaled
    9. Load V from HBM
   10. BMM2: acc += p @ V

  If all_skip (SKIP — Pool-K path):
    6. k_mean = mean(K_tile, dim=tokens)           ← K already in SRAM, free reduction
       K is loaded as [BLOCK_D, BLOCK_N], so: k_mean = sum(K, dim=1) / n_valid
    7. approx_score = sum(Q * k_mean, dim=D) * qk_scale   ← [BLOCK_M] per-row scores
    8. m_new = max(row_max, approx_score)
    9. p_approx = exp2(approx_score - m_new) * n_valid_kv  ← BLOCK_M exp2 only (128, not 8192)
       The * n_valid_kv approximates: sum_j(exp(q·k_j)) ≈ N * exp(q·mean(k))
   10. Online softmax correction: row_sum, acc rescaled
   11. Load precomputed v_mean[j]                  ← from L2 cache (~0.5MB buffer)
   12. acc += p_approx[:, None] * v_mean[None, :]  ← fresh weight × fresh v_mean
       — NO full V load from HBM, NO full BMM2
```

### Implementation Details

The kernel is in `modelopt/torch/kernels/triton_fa.py`:
- `_precompute_vmean`: Triton kernel that computes `mean(V_tile)` for all KV tiles
- `_attn_fwd_body`: Main attention kernel with `APPLY_SKIP_V25` constexpr branch
- `_attention_v25_forward`: Python launcher that calls precompute then attention
- `attention()`: Public API, routes to V2.5 path when `v_mean_cache` is provided

Key parameters:
- `BLOCK_M=128, BLOCK_N=64, BLOCK_D=128` (for LTX-2 with head_dim=128)
- `qk_scale = sm_scale * log2(e)` (scores are in log2 space for exp2)
- `skip_threshold_log2 = -threshold * log2(seq_k)` (diffusion mode, normalized by seq len)

### Key Improvement Over V2.5

V2.51 eliminates ALL stale cached state. Both the attention weight and the V
approximation are computed fresh every denoising step:

| | V2.5 | V2.51 |
|---|---|---|
| **Weight (p)** | Stale cached scalar from prev step | Fresh per-row from pool-K |
| **v_mean** | Stale cached from prev step | Fresh precomputed every step |
| **Cache needed** | sum_p + v_mean per tile | v_mean only (no sum_p) |
| **Exponentials per skip** | 0 | 128 (vs 8192 full) |
| **Staleness** | Gets worse over consecutive skipped steps | None — always fresh |

### Why Pool K?

Full softmax for a skipped tile requires BLOCK_M x BLOCK_N = 8,192 exponentials.

Pool-K computes `Q @ mean(K)` → [BLOCK_M] scores → 128 exponentials.
This is 64x fewer exponentials while keeping per-query-row granularity
(each of 128 query rows gets its own weight for the skipped KV tile).

K is already loaded into SRAM for BMM1 (`Q @ K^T` for the skip decision),
so `mean(K)` is a free reduction — no extra HBM load.

### Why Precompute v_mean?

V is NOT loaded for skipped tiles (that's the main bandwidth saving).
Instead, v_mean is precomputed for ALL KV tiles in a separate kernel pass
before the attention kernel runs. The v_mean buffer (~0.5 MB for LTX-2)
fits in L2 cache, so reading v_mean for skipped tiles is essentially free.

Total V HBM bandwidth:
- 1x for precompute (all tiles, sequential coalesced read)
- ~(1-sparsity)x for kept tiles (inside attention kernel for BMM2)

At 75% sparsity: 1.0 + 0.25 = 1.25x (vs 1.0x for dense, 0.25x for V1)

### Relation to VSA

V2.51 arrives at a very similar design to VSA's compression branch:

| | V2.51 | VSA compression branch |
|---|---|---|
| K pooling | mean(K_tile) → [d] | mean(K_block) → [d] |
| V pooling | mean(V_tile) precomputed | mean(V_block) computed inline |
| Q pooling | None (keep full Q) | mean(Q_block) → [d] |
| Weight shape | [BLOCK_M] per-row | scalar per block |
| Blending | Fused in flash attention l/acc | Separate branch + learned gate |
| Training | Not needed | gate_compress learned |

V2.51 keeps full Q resolution (per-row weights) while VSA pools all three.
V2.51 is fused inside the flash attention tile loop; VSA is a separate branch.

## Cost Analysis

### Per Skipped Tile

| Operation | Full Attention | V1 (pure skip) | V2.51 (pool-K + v_mean) |
|-----------|---------------|-----------------|-------------------------|
| BMM1 (Q @ K^T) | BLOCK_M x BLOCK_N x d | Same | Same |
| Softmax exp | BLOCK_M x BLOCK_N = 8192 | 0 | **BLOCK_M = 128** (64x fewer) |
| BMM2 (p @ V) | BLOCK_M x BLOCK_N x d | 0 | **BLOCK_M x d** (outer product, BLOCK_N x fewer) |
| V load from HBM | BLOCK_N x d x 2 bytes | 0 | 0 (precomputed) |
| v_mean load | — | — | d x 4 bytes (L2 hit) |
| l update | per-row sum | not updated | per-row approx |
| acc update | full BMM2 | not updated | rank-1 outer product |

### Precompute Overhead

- One pass over all V tiles: `num_kv_tiles x BLOCK_N x d` reads from HBM
- Output: `num_kv_tiles x d` writes (tiny)
- For LTX-2: ~0.5 MB output buffer, fits in L2

### Cache Storage

| Buffer | Shape (LTX-2) | Size |
|--------|---------------|------|
| v_mean | [1, 32, 60, 128] | 0.5 MB |
| sum_p | Not needed | 0 |
| **Total** | | **0.5 MB** |

## Comparison of All Versions

| | V1 | V2.5 | V2.51 |
|---|---|---|---|
| **Skip → l** | Not updated | += stale cached sum_p | += fresh p_approx (per-row) |
| **Skip → acc** | Not updated | += stale_sum_p * stale_v_mean | += fresh_p * fresh_v_mean |
| **Weight freshness** | — | Stale (from prev kept step) | Fresh (pool-K every step) |
| **V freshness** | — | Stale (from prev kept step) | Fresh (precomputed every step) |
| **Exp per skip** | 0 | 0 | 128 |
| **Cache** | None | sum_p + v_mean | v_mean only |
| **Training** | No | No | No |

## Files

| File | Role |
|------|------|
| `modelopt/torch/kernels/triton_fa.py` | Triton kernel: `_precompute_vmean`, `_attn_fwd_body` (APPLY_SKIP_V25 branch), `_attention_v25_forward`, `attention()` |
| `modelopt/torch/sparsity/attention_sparsity/kernels/ltx_triton_attention.py` | LTX-2 integration: v_mean_cache allocation, `_ltx_triton_attention`, `_TritonLTXAttentionWrapper` |
| `modelopt/torch/sparsity/attention_sparsity/kernels/diffusers_triton_attention.py` | Diffusers integration: v_mean_cache allocation, Triton attention dispatch |
| `modelopt/torch/sparsity/attention_sparsity/methods/triton_skip_softmax_diffusion.py` | Method class: `enable_v25` flag, threshold management, kernel config dispatch |
| `examples/diffusers/sparsity/run_experiments.py` | Experiment runner with fixed-threshold bypass |
| `examples/diffusers/sparsity/capture_attn_inputs.py` | Capture Q/K/V inputs for offline analysis |
| `examples/diffusers/sparsity/STUDIES.md` | Captured data structure, loading code, experiment summary |
| `tests/gpu/torch/sparsity/attention_sparsity/test_triton_fa.py` | Unit tests for V2.5 path |
