# Bypass Distillation (Blockwise Local Distillation)

Bypass distillation (also called **Blockwise Local Distillation / BLD**) is an optional pipeline
stage that trains alternative transformer block configurations using per-block knowledge
distillation from the teacher model. It significantly improves the quality of aggressively
compressed models by producing better "puzzle pieces" for the MIP solver.

## When to use bypass

Bypass distillation is most beneficial for **aggressive compression**. For mild FFN pruning
(e.g., keeping most of the intermediate width), weight-initialization-based pruning alone often
provides a reasonable starting point and bypass may not be essential. Use bypass when:

- **Heavy FFN pruning**: the target `intermediate_size` is significantly smaller than the
  teacher's (e.g., ≤ 1/8 of the teacher width). For example, on Llama-3.1-8B
  (`intermediate_size=14336`), bypass is strongly recommended for sizes ≤ 1792.
- **KV head compression**: `num_key_value_heads` is being significantly reduced. The
  `AverageKV` initialisation provides a useful starting point but bypass distillation recovers
  additional accuracy.
- **Attention no-op blocks**: when a full attention block is removed (`no_op: true`), bypass
  trains the co-located FFN to compensate for the missing attention.

## Time cost

Bypass distillation is a full training loop. Plan for several hours per configuration when using
~1B training tokens on H100 GPUs. Total time scales with
`len(bypass.configs) × training_tokens`. This is comparable to lightweight fine-tuning.

## Sequential execution

Each entry in `bypass.configs` trains **sequentially** (one config at a time). There is no
parallelism across configurations. Distribute jobs across different runs if time is a
constraint.

## Enabling bypass

In your concrete model YAML, uncomment the bypass line:

```yaml
defaults:
  - Llama-3_1-8B
  - bypass: defaults   # remove the comment to enable bypass distillation
  - _self_
```

A shared `bypass/defaults.yaml` is located at
[`configs/bypass/defaults.yaml`](configs/bypass/defaults.yaml). It is used by all models.
Adjust `training.training_tokens` (default is 10K tokens for sanity-check runs; set to `1e+9`
for production runs) and the `auto_configs` or `configs` settings to match your compression
targets.

## Decoupled vs. coupled BLD

**Decoupled BLD** trains only one sub-block type at a time while keeping the other frozen:

| `keys_to_learn` | What is trained |
|---|---|
| `subblock_ffn` | FFN weights only (attention frozen) |
| `subblock_attention` | Attention weights only (FFN frozen) |
| `subblock_mamba` | Mamba SSM weights (hybrid models, e.g. NemotronH) |
| `entire_block` | Full transformer block (coupled BLD) |

**Coupled BLD** (`keys_to_learn: entire_block`) trains the whole block end-to-end and can
capture interactions between attention and FFN. It is more expensive and can be harder to
optimise. Decoupled BLD is recommended as a first step and often sufficient.

Typical decoupled workflow:
1. Run `keys_to_learn: subblock_ffn` for all FFN sizes you want in the replacement library.
2. Optionally run `keys_to_learn: subblock_attention` for blocks where KV heads are reduced.

## Training multiple configurations

Use `bypass.configs` to train multiple block configurations sequentially:

```yaml
bypass:
  training:
    training_tokens: 1e+9   # ~1B tokens per config
  configs:
    - model_config_overrides:
        ffn:
          - intermediate_size: 1792  # aggressive — bypass strongly recommended
        attention:
          - num_key_value_heads: null
      keys_to_learn: subblock_ffn
    - model_config_overrides:
        ffn:
          - intermediate_size: 3584
        attention:
          - num_key_value_heads: null
      keys_to_learn: subblock_ffn
```

> **Note:** Always include `num_key_value_heads: null` under `attention:` even when not
> changing KV heads. Omitting it when `no_op: true` is set on another field can cause
> a config parsing issue.

Trained checkpoints are automatically symlinked into `$PUZZLE_DIR/ckpts/` where the replacement
library builder picks them up in the next pipeline stage.

## Auto-generating configs from the pruning search space

Instead of listing each config manually, use `bypass.auto_configs` to generate configs
automatically from the pruning search space. The default (`auto_configs.attn: true`) trains
one attention-only bypass per KV-head reduction specified in `pruning.n_heads_in_group_list`:

```yaml
bypass:
  auto_configs:
    attn: true   # one subblock_attention config per pruned kv-head count
    ffn: false   # set true: one subblock_ffn config per size in pruning.intermediate_size_list
    blk: false   # set true: cartesian product (FFN size × kv-head count), entire_block BLD
  training:
    training_tokens: 1e+9
```

Teacher-size subblocks are automatically excluded (no redundant training). For `blk`, all
combinations where **both** FFN and attention are at teacher values are skipped.

All three flags can be combined. Order of generated configs: FFN → attn → blk.

## Attention no-op + FFN-only bypass

A common aggressive compression pattern removes entire attention blocks (`no_op: true`) and
trains only the FFN in those blocks. Example config:

```yaml
configs:
  - model_config_overrides:
      ffn:
        - intermediate_size: 12288
      attention:
        - num_key_value_heads: null
          no_op: true
    keys_to_learn: subblock_ffn
```

When attention is removed, only the FFN parameters are trained. The bypass code automatically
skips attention-related weights (including model-specific ones such as Qwen3's `q_norm`/`k_norm`)
during student weight initialisation.

## Weights & Biases logging

Enable W&B to track per-block distillation loss and validation metrics:

```yaml
bypass:
  wandb_log: true
  wandb:
    project: my-puzzletron-project
    entity: my-org
```

W&B logs iteration number, token count, learning rate, and per-block loss at each log interval.
If `wandb` is not installed, logging is silently disabled.
