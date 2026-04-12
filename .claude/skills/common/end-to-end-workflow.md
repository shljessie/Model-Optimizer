# End-to-End Workflow: PTQ → Deploy → Eval

This document ties together the three domain skills (PTQ, Deployment, Evaluation) for the common workflow of quantizing a model, deploying it, and evaluating accuracy.

## Pipeline Overview

```text
PTQ (quantize)          → Deployment (serve)         → Evaluation (benchmark)
─────────────────         ──────────────────           ────────────────────────
hf_ptq.py                vLLM / SGLang / TRT-LLM      NEL (SLURM or JET)
  ↓                         ↓                            ↓
NVFP4/FP8 checkpoint      OpenAI-compatible API        MMLU, GSM8K, GPQA scores
  (safetensors)            (http://host:8000)           (results.yml)
```

## Workspace Continuity

All three stages share the same workspace directory. The PTQ output becomes the deployment input, and eval results land alongside:

```text
workspaces/model-name-format/
  output/              ← PTQ checkpoint (safetensors + config.json)
  eval_results/        ← NEL evaluation artifacts (results.yml per task)
  eval_config.yaml     ← NEL config for evaluation
  scripts/             ← Custom run scripts (if needed)
  logs/                ← SLURM job logs
```

When starting a deployment or evaluation step, always check for an existing workspace from a prior PTQ run:

```bash
ls workspaces/
```

## Unsupported Models

Models not in the verified support matrices require extra work at each stage:

| Stage | What can go wrong | Reference |
|-------|-------------------|-----------|
| **PTQ** | Unknown architecture, FP8 source checkpoint, VLM structure | `ptq/references/unsupported-models.md` |
| **Deployment** | Missing architecture mapping, weight key mismatches, quant/unquant layer confusion | `deployment/references/unsupported-models.md` |
| **Evaluation** | Framework patches needed in deployment container, gated datasets, cluster storage | `evaluation/references/nel-ci-guide.md` |

Each stage has its own debug loop (run → read error → diagnose → patch → re-run). Fixes from one stage often inform the next — e.g., if PTQ required a transformers upgrade, deployment and evaluation will too.

## NEL Evaluation with Custom Deployments

When the serving framework needs runtime patches (e.g., transformers upgrade, model handler fix), override `deployment.command` in the NEL config to inject fixes before serving:

```yaml
deployment:
  command: >-
    pip install "transformers>=5.0.0.dev0" --pre -q &&
    sed -i 's/old_pattern/new_pattern/' /path/to/framework/file.py &&
    ${deployment.base_command}
```

This works with both NEL SLURM executor and NEL CI (via `NEL_DEPLOYMENT_COMMAND`).

## Decision: NEL SLURM Executor vs NEL CI (JET)

| Factor | NEL SLURM executor | NEL CI (JET) |
|--------|-------------------|--------------|
| **When to use** | Iterative debugging, checkpoint on non-JET cluster, custom patches needed | Production evals, MLflow tracking, reproducible configs |
| **Checkpoint location** | Any cluster you have SSH access to | Must be on JET cluster `/lustre/` storage |
| **Secrets (HF_TOKEN, NGC)** | Provide your own via `host:` env vars | Managed centrally via JET secrets |
| **Container patches** | Override `deployment.command` | Use `NEL_DEPLOYMENT_COMMAND` |
| **MLflow export** | Manual setup | Automatic |
| **Gated datasets** | Your HF account needs access | Handled by `COMPEVAL_HF_TOKEN` |
