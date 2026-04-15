# calibration-data

**Description:** Use this skill when the user needs to select, prepare, or validate calibration data for post-training quantization (PTQ). Triggers on prompts like "pick calibration data", "prepare calib dataset", "how many calibration samples", "what dataset for quantization", or "calibration data for PTQ".

Do not trigger on: general dataset download requests, fine-tuning data preparation, or evaluation benchmark setup.

## Prerequisites

### Environment Assumptions
- OS: Linux or macOS
- Python >= 3.10
- ModelOpt installed (`pip install nvidia-modelopt`)
- HuggingFace `datasets` library installed (`pip install datasets`)
- Model checkpoint available locally or accessible via HuggingFace Hub
- GPU with at least 16 GB VRAM for calibration forward passes

### Setup Steps
- [ ] Set `HF_TOKEN` environment variable if using gated datasets (e.g. Llama, Gemma)
- [ ] Confirm model tokenizer is available at the checkpoint path
- [ ] Verify at least 512 calibration samples are available in the chosen dataset
- [ ] Run `python -c "import modelopt; print(modelopt.__version__)"` to confirm ModelOpt is installed

---

## Step 1 — Identify the model task type

**Goal:** Determine what kind of model is being quantized so the right calibration domain is used.

Ask the user (or infer from the checkpoint path):
- Is this a **language model** (LLM), **vision-language model** (VLM), or **code model**?
- What is the intended deployment task? (chat, summarization, code completion, etc.)

**Expected Outcome:** You know the model type and intended task before choosing a dataset.

---

## Step 2 — Select a calibration dataset

**Goal:** Choose a calibration dataset that matches the model's deployment domain.

| Model Type | Recommended Dataset | HuggingFace ID |
|------------|---------------------|----------------|
| General LLM (chat) | Pile subset | `mit-han-lab/pile-val-backup` |
| Code model | CodeParrot | `codeparrot/github-code` |
| Instruction-tuned LLM | Alpaca | `tatsu-lab/alpaca` |
| VLM | COCO captions | `HuggingFace/COCO` |

If the user has a custom dataset, use it if it:
- Contains representative samples from the deployment domain
- Has at least 512 samples
- Can be tokenized by the model's tokenizer without errors

**Expected Outcome:** A dataset ID or local path is confirmed.

---

## Step 3 — Prepare calibration samples

**Goal:** Load and tokenize calibration samples into the format ModelOpt expects.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<checkpoint_path>")
dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")

def get_calib_dataloader(num_samples=512, seq_len=512):
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        enc = tokenizer(
            example["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )
        samples.append(enc["input_ids"])
    return samples

calib_data = get_calib_dataloader()
```

**Expected Outcome:**
- `calib_data` is a list of 512 tensors of shape `[1, 512]`
- No tokenizer errors; all samples successfully encoded

---

## Step 4 — Validate calibration data quality

**Goal:** Confirm the data is suitable before running PTQ (avoids wasted quantization runs).

Run a quick sanity check:

```python
# Check no empty or all-padding samples
import torch

pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
for i, sample in enumerate(calib_data):
    non_pad = (sample != pad_id).sum().item()
    if non_pad < 10:
        print(f"Warning: sample {i} is nearly all padding ({non_pad} real tokens)")

print(f"Calibration data ready: {len(calib_data)} samples")
```

**Expected Outcome:**
- Fewer than 5% of samples flagged as nearly all-padding
- Print confirms sample count matches requested `num_samples`

---

## Example Prompts

### Happy path
> "I'm quantizing Llama-3.1-8B to FP8 for chat. What calibration data should I use and how do I prepare it?"

Expected: Walk through Steps 1–4, recommend `mit-han-lab/pile-val-backup`, produce a ready-to-use dataloader.

### Edge case
> "I only have 128 samples from my proprietary customer support logs. Can I use those for calibration?"

Expected: Warn that 128 samples is below the recommended 512; suggest augmenting with a public dataset from the same domain; still show how to use custom data.

### Off-topic / negative
> "Download the MMLU benchmark and run accuracy evaluation on my quantized model."

Expected: Do NOT trigger this skill. This is an evaluation task — redirect to the `evaluation` skill.
