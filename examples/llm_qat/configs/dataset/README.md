# Dataset Blend Configuration

Dataset blends are defined in YAML files that specify which datasets to mix,
how to sample from them, and how to tokenize them.

See [`blend_example.yaml`](blend_example.yaml) for a runnable example with all options.

## Blend YAML Structure

Blend YAML files contain only the `sources` list. Processing parameters
(`train_samples`, `eval_samples`, `cache_dir`, `shuffle`, etc.) are set via
`DataArguments` in the training config YAML or CLI flags.

## Per-Source Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `hf_path` | yes | - | HuggingFace dataset path or local path |
| `ratio` | yes | - | Relative weight (normalized across all sources) |
| `split` | yes | - | Split(s) to load (auto train/eval). See below |
| `dataset_kwargs` | no | `{}` | Extra kwargs passed to `datasets.load_dataset()` (e.g. `{name: "3.0.0"}`) |
| `apply_chat_template` | no | `true` | If true, expects OpenAI messages format |
| `chat_key` | no | `"messages"` | Key containing conversations |
| `category` | no | `""` | Label for logging |

## Split Modes

Loads the specified split(s), pools them, then auto-splits into train/eval.

```yaml
# Single split
split: train

# Comma-separated (equal weight per split)
split: code,math,stem

# Dict (weighted per split: 3:2:1 ratio)
split:
  code: 3
  math: 2
  stem: 1
```

## Dataset Kwargs

Pass any extra keyword arguments to `datasets.load_dataset()` via `dataset_kwargs`:

```yaml
# HF config name (e.g. cnn_dailymail)
dataset_kwargs: {name: "3.0.0"}

# Multiple kwargs
dataset_kwargs:
  name: "3.0.0"
  trust_remote_code: true
  revision: main
```

## Streaming and Shuffle

All HuggingFace datasets are loaded with `streaming=True` to avoid downloading
entire datasets.

- `shuffle: true` - Reservoir sampling: `dataset.shuffle(buffer_size=N).take(n)`.
  Accurate but slower with large buffers.
- `shuffle: false` - Take first N samples: `dataset.take(n)`. Fast and deterministic.

## Pre-tokenize and Cache

Pre-tokenize the dataset before training to avoid repeated work:

```sh
python dataset_utils.py \
    --config configs/train/qad_nvfp4.yaml \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct
```

The cached dataset is saved to `cache_dir` (or an auto-generated temp path).
Subsequent runs with the same config and tokenizer reuse the cache.

## Adding New Datasets

Add a source entry to your blend YAML:

```yaml
sources:
  # Chat dataset (OpenAI messages format)
  - hf_path: your/dataset
    split: train
    ratio: 1000

  # Dataset with different chat key
  - hf_path: your/sharegpt-dataset
    split: train
    ratio: 500
    chat_key: conversations

  # Plain text dataset (pretraining-style)
  - hf_path: your/text-corpus
    split: train
    ratio: 500
    apply_chat_template: false
```
