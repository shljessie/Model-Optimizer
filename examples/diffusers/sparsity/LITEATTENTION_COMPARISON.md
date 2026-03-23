# LiteAttention vs Skip-Softmax: Algorithm Comparison

## Skip Decision

### ModelOpt Skip-Softmax (this repo)

Relative comparison against the **cumulative maximum** (global running max):

```
gap = cummax[j] - block_max[j]
skip tile j if:  gap >= threshold * log(seq_k)
```

- `cummax[j]` = max of block_max[0..j] (all tiles seen so far)
- `block_max[j]` = max attention score in tile j
- Threshold is a single scalar, calibrated via percentile

### LiteAttention

Absolute comparison against a **fixed threshold** in log2-space:

```
skip tile j if:  max_score[j] < threshold
```

- `max_score[j]` = max QK score in tile j (log2 scale)
- No relative comparison against other tiles
- Threshold is per-layer, calibrated via binary search targeting output error

## Key Algorithmic Differences

| | Skip-Softmax (ModelOpt) | LiteAttention |
|---|---|---|
| **Reference** | Cumulative max (global) | Absolute threshold (fixed) |
| **Conservativeness** | More conservative (cummax >= any single tile) | Less conservative (flat cutoff) |
| **Temporal state** | Stateless (recompute mask each forward) | Stateful (skip list carried across timesteps) |
| **Normalization** | `gap / log(seq_k)` for seq-length invariance | None (log2-space threshold is absolute) |
| **Calibration target** | Sparsity percentage (e.g., "skip 20%") | Output error (e.g., "L1 < 5%") |
| **Calibration method** | Percentile of all normalized gaps | Binary search per-layer |
| **Threshold scope** | One global scalar | Per-layer, optionally per-timestep |

## Temporal Skip List (LiteAttention only)

LiteAttention exploits **temporal coherence** in diffusion models:

1. **Timestep t**: Full attention, record which tiles are below threshold
2. **Timestep t+1**: Skip tiles marked in timestep t, record new decisions
3. Double-buffered skip list `[2, B, H, qtiles, ktiles+2]` (int16, ping-pong)

Tiles marked as unimportant at one timestep are physically bypassed at the
next. This eliminates profiling overhead since the skip decision is a byproduct
of QK computation that already happened.

Skip-softmax has no temporal state -- it recomputes the mask from scratch
every forward pass.

## Conservativeness

Skip-softmax is strictly more conservative. Since `cummax >= block_max[j-1]`,
the gap against cummax is always >= the gap against any single previous tile.
Skip-softmax will never skip a tile that a previous-tile-only comparison would
keep.

## Quantization (LiteAttention)

LiteAttention supports INT8 Q/K quantization with per-tile symmetric scaling:

**Q**: Per-tile (`kBlockM` rows x `head_dim`), one scale per tile.
```
tile_max = max(|Q[tile]|)
Q_int8 = round(Q * 127 / tile_max)
q_descale = tile_max * log2(e) / (127 * sqrt(head_dim))
```

**K**: Same granularity, but with **channel-wise mean centering**:
```
k_mean[d] = mean(K[:, :, h, d])       # per-channel across seq_len
K_centered = K - k_mean
tile_max = max(|K_centered[tile]|)
K_int8 = round(K_centered * 127 / tile_max)
```

Mean centering is free because softmax is shift-invariant:
`softmax(Q(K-mu)^T) = softmax(QK^T - Q*mu^T) = softmax(QK^T)`.
It reduces K's dynamic range for better INT8 utilization.

**V is NOT quantized** (stays BF16).

**FP8** uses standard FA3 path with coarser per-head granularity (one scale
for all seq positions), all three Q/K/V quantized.

| | INT8 (LiteAttention) | FP8 (standard FA3) |
|---|---|---|
| Granularity | Per-tile | Per-head |
| Q descale shape | `[B, H, num_q_tiles]` | `[B, H]` |
| K descale shape | `[B, H, num_k_tiles]` | `[B, H]` |
| V quantized? | No | Yes |
| Mean smoothing | Yes (K) | No |

## Training Support

| | Skip-Softmax (ModelOpt) | LiteAttention |
|---|---|---|
| Forward grad flow | Yes (`masked_fill(-inf)` is differentiable) | Yes (non-skipped tiles computed normally) |
| Backward pass | Yes (standard autograd) | Disabled by default |
| Training feasible? | Yes, but no compute savings | Experimental |

## Must-Do / Must-Skip Lists (LiteAttention only)

LiteAttention supports forcing specific token ranges to always be computed
or always be skipped:

```python
must_do_list = [0, 128, 500, 640]   # always compute positions 0-127 and 500-639
must_skip_list = [1000, 1024]        # always skip positions 1000-1023 (e.g., padding)
```

These override the threshold-based skip decision. Skip-softmax has no
equivalent mechanism.
