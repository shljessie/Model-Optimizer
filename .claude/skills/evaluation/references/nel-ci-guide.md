# NEL CI Evaluation Guide

NEL CI is the recommended entry point for running evaluations on NVIDIA JET infrastructure. This guide covers patterns for evaluating quantized checkpoints using both the NEL SLURM executor (direct) and the NEL CI GitLab pipeline.

Reference repo: `gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nemo-evaluator-launcher-ci`

---

## 1. Two Execution Paths

| Path | When to use | How it works |
|------|-------------|--------------|
| **NEL SLURM executor** | You have SSH access to the cluster, checkpoint is on cluster storage | `nel run --config config.yaml` from your workstation; NEL SSHes to cluster and submits sbatch jobs |
| **NEL CI GitLab pipeline** | You want managed infrastructure, MLflow export, reproducible configs | Trigger via GitLab API or UI; JET orchestrates everything |

### NEL SLURM executor

Best for iterative development and debugging. Run from any machine with SSH access to the cluster:

```bash
export DUMMY_API_KEY=dummy
export HF_TOKEN=<your_token>

nel run --config eval_config.yaml \
    -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10  # test first
```

### NEL CI trigger

Best for production evaluations with MLflow tracking. See the trigger script pattern in section 4.

---

## 2. Cluster Reference

| Cluster | GPUs/Node | Architecture | Max Walltime | Storage | Notes |
|---------|-----------|-------------|--------------|---------|-------|
| oci-hsg | 4 | GB200 | 4 hours | `/lustre/` | Set `tensor_parallel_size=4` |
| cw | 8 | H100 | — | `/lustre/` | — |
| oci-nrt | 8 | H100 | — | `/lustre/` | Numerics configs |
| dlcluster | 4 (B100 partition) | B100 | 8 hours | `/home/omniml_data_*` | No `/lustre/`; use local NFS paths |

**Important**: `deployment.tensor_parallel_size` determines how many GPUs are requested. If this exceeds the cluster's GPUs per node, the job fails with a memory allocation error.

---

## 3. Checkpoint Availability

The checkpoint must be on a filesystem accessible from the cluster's **compute nodes** (not just login nodes).

| Cluster type | Accessible storage | NOT accessible |
|-------------|-------------------|----------------|
| JET clusters (oci-hsg, cw, oci-nrt) | `/lustre/fsw/...` | Workstation paths (`/home/scratch.*`), NFS mounts from other clusters |
| dlcluster | `/home/omniml_data_*`, `/home/scratch.*` | `/lustre/` (not available) |

If the checkpoint is on a workstation, **copy it to cluster storage first**:

```bash
rsync -av /path/to/local/checkpoint \
    <cluster-login>:/lustre/fsw/portfolios/coreai/users/$USER/checkpoints/
```

**Cross-cluster copy** (e.g., dlcluster → oci-hsg): If the two clusters can't SSH to each other directly, pipe through your workstation without staging to disk:

```bash
ssh user@source-cluster "tar czf - -C /path/to/checkpoint ." | \
    ssh user@target-cluster "tar xzf - -C /lustre/.../checkpoints/model-name"
```

After copying, set permissions for svc-jet: `chmod -R 777 /lustre/.../checkpoints/model-name`

For dlcluster, the checkpoint paths are directly accessible since the NFS mounts are shared between login and compute nodes.

---

## 4. NEL CI Trigger Pattern

For JET clusters, trigger evaluations via the GitLab API.

### Simple deployment (standard models)

For models that work with stock vLLM/SGLang, use `NEL_DEPLOYMENT_COMMAND` directly:

```bash
export GITLAB_TOKEN=<your_gitlab_token>

curl -k --request POST \
  --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "ref": "main",
    "variables": [
      {"key": "NEL_CONFIG_PATH", "value": "configs/AA/minimax_m2_5_lbd_lax.yaml"},
      {"key": "NEL_ACCOUNT", "value": "coreai_dlalgo_modelopt"},
      {"key": "NEL_CLUSTER", "value": "oci-hsg"},
      {"key": "NEL_CHECKPOINT_OR_ARTIFACT", "value": "/lustre/.../checkpoint"},
      {"key": "NEL_DEPLOYMENT_IMAGE", "value": "vllm/vllm-openai:v0.19.0"},
      {"key": "NEL_DEPLOYMENT_COMMAND", "value": "vllm serve /checkpoint --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --quantization modelopt_fp4 --trust-remote-code --served-model-name my-model"},
      {"key": "NEL_OTHER_OVERRIDES", "value": "deployment.tensor_parallel_size=4 execution.walltime=04:00:00"},
      {"key": "NEL_HF_HOME", "value": "/lustre/.../cache/huggingface"},
      {"key": "NEL_VLLM_CACHE", "value": "/lustre/.../cache/vllm"},
      {"key": "NEL_CLUSTER_OUTPUT_DIR", "value": "/lustre/.../nv-eval-rundirs"}
    ]
  }' \
  "https://gitlab-master.nvidia.com/api/v4/projects/221804/pipeline"
```

### Complex deployment (unsupported models needing runtime patches)

If the model needs runtime patches (e.g., transformers upgrade, framework source fixes), **do NOT put multi-step commands in `NEL_DEPLOYMENT_COMMAND`** — Hydra's override parser will break on nested quotes, `&&`, `$()`, etc.

Instead, use the **wrapper script pattern**: place a `serve.sh` in the checkpoint directory on the cluster, then point `NEL_DEPLOYMENT_COMMAND` to it.

**Step 1** — Write wrapper script to the checkpoint directory on the cluster:

```bash
ssh <cluster-login> 'cat > /lustre/.../checkpoint/serve.sh << '"'"'EOF'"'"'
#!/bin/bash
set -e
pip install "transformers>=5.0.0.dev0" "huggingface_hub>=0.32.0" --pre -q
# Patch vLLM for ministral3 support (example)
MISTRAL3_PY=$(find /usr/local/lib -path "*/vllm/model_executor/models/mistral3.py" 2>/dev/null | head -1)
sed -i "s/old_pattern/new_pattern/" "$MISTRAL3_PY"
exec vllm serve /checkpoint --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 --quantization modelopt_fp4 \
    --trust-remote-code --served-model-name my-model --gpu-memory-utilization 0.9
EOF
chmod 777 /lustre/.../checkpoint/serve.sh'
```

**Step 2** — Set `NEL_DEPLOYMENT_COMMAND` to the wrapper:

```json
{"key": "NEL_DEPLOYMENT_COMMAND", "value": "bash /checkpoint/serve.sh"}
```

This works because the checkpoint is mounted at `/checkpoint` inside the container. The script is Hydra-safe (no special characters in the override value).

### Custom configs with `NEL_CONFIG_BASE64`

When using a custom config (not from the repo), use `NEL_CONFIG_BASE64` instead of `NEL_CONFIG_PATH`. This requires setting `NEL_UNTRUSTED_EVAL=true`:

```python
import json, base64, subprocess, os

with open("my_config.yaml") as f:
    config_b64 = base64.b64encode(f.read().encode()).decode()

payload = {
    "ref": "main",
    "variables": [
        {"key": "NEL_CONFIG_BASE64", "value": config_b64},
        {"key": "NEL_ACCOUNT", "value": "coreai_dlalgo_modelopt"},
        {"key": "NEL_CLUSTER", "value": "oci-hsg"},
        {"key": "NEL_CHECKPOINT_OR_ARTIFACT", "value": "/lustre/.../checkpoint"},
        {"key": "NEL_DEPLOYMENT_IMAGE", "value": "vllm/vllm-openai:v0.19.0"},
        {"key": "NEL_DEPLOYMENT_COMMAND", "value": "bash /checkpoint/serve.sh"},
        {"key": "NEL_UNTRUSTED_EVAL", "value": "true"},
        # ... other variables
    ]
}

# Use Python to construct JSON (avoids shell escaping issues with curl)
token = os.environ["GITLAB_TOKEN"]
subprocess.run(
    ["curl", "-k", "--request", "POST",
     "--header", f"PRIVATE-TOKEN: {token}",
     "--header", "Content-Type: application/json",
     "--data", json.dumps(payload),
     "https://gitlab-master.nvidia.com/api/v4/projects/221804/pipeline"],
)
```

> **Tip**: Use Python (not bash) to construct the JSON payload for `curl`. Shell escaping of base64 strings and nested quotes is error-prone.

---

## 5. Environment Variables

### SLURM executor format

Env vars in NEL SLURM configs require explicit prefixes:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `host:VAR_NAME` | Read from the host environment where `nel run` is executed | `host:HF_TOKEN` |
| `lit:value` | Literal string value | `lit:dummy` |

```yaml
evaluation:
  env_vars:
    DUMMY_API_KEY: host:DUMMY_API_KEY
    HF_TOKEN: host:HF_TOKEN
```

### JET executor format

JET configs reference JET secrets with `$SECRET_NAME`:

```yaml
execution:
  env_vars:
    evaluation:
      HF_TOKEN: $COMPEVAL_HF_TOKEN
```

### Gated datasets

Tasks that download gated HuggingFace datasets (e.g., GPQA, HLE) need `HF_TOKEN` passed to the evaluation container.

**NEL CI (JET)**: Handled automatically — the `COMPEVAL_HF_TOKEN` JET secret is pre-configured by the eval platform team. No user action needed; you don't even need personal access to the gated dataset.

**NEL SLURM executor**: You must provide your own HF token, AND your HuggingFace account must have been granted access to the gated dataset (e.g., request access at <https://huggingface.co/datasets/Idavidrein/gpqa> for GPQA).

```yaml
evaluation:
  env_vars:
    HF_TOKEN: host:HF_TOKEN  # SLURM executor — reads from your local env
  tasks:
    - name: simple_evals.gpqa_diamond
      env_vars:
        HF_TOKEN: host:HF_TOKEN
```

---

## 6. Serving Framework Notes

### vLLM

- Binds to `0.0.0.0` by default — health checks work out of the box
- For NVFP4: `--quantization modelopt_fp4`
- For unsupported models (e.g., ministral3): may need custom `deployment.command` to patch the framework before serving (see `deployment/references/unsupported-models.md`)

### SGLang

- **Must include `--host 0.0.0.0`** — SGLang defaults to `127.0.0.1` which blocks health checks from the eval client
- Must include `--port 8000` to match NEL's expected port
- For NVFP4: `--quantization modelopt_fp4`

---

## 7. Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` pulling eval container | NGC credentials not set on cluster | Set up `~/.config/enroot/.credentials` with NGC API key |
| `PermissionError: /hf-cache/...` | HF cache dir not writable by svc-jet | Set `NEL_HF_HOME` to your own `chmod 777` directory |
| Health check stuck at `000` | Server binding to localhost | Add `--host 0.0.0.0` to deployment command (SGLang) |
| `Memory required by task is not available` | TP size exceeds GPUs/node | Set `tensor_parallel_size` to match cluster (4 for oci-hsg, dlcluster B100) |
| TIMEOUT after eval completes | Walltime too short for eval + MLflow export | Set `execution.walltime=04:00:00` |
| Gated dataset auth failure | `HF_TOKEN` not passed to eval container | Add `env_vars.HF_TOKEN` at evaluation or task level |
| `NEL_OTHER_OVERRIDES` splits `extra_args` | Space-separated parsing breaks multi-flag values | Use `NEL_DEPLOYMENT_COMMAND` instead |
| Checkpoint not found in container | Path not on cluster compute-node filesystem | Copy checkpoint to `/lustre/` (or cluster-accessible path) first |
| `trusted_eval` type mismatch in MLflow export | NEL writes boolean `true` instead of string `"true"` | Fix with `sed -i "s/trusted_eval: true/trusted_eval: 'true'/"` in export config |
| `LexerNoViableAltException` in Hydra | `NEL_DEPLOYMENT_COMMAND` contains quotes, `&&`, `$()` | Use wrapper script pattern (section 4): put script in checkpoint dir, set command to `bash /checkpoint/serve.sh` |
| `Bad Request` from GitLab API trigger | Shell escaping mangled the JSON payload | Use Python to construct JSON (section 4) instead of bash heredocs/string interpolation |
| `The model <path> does not exist` (404) | Eval client uses checkpoint path as model_id instead of served_model_name | Add `deployment.served_model_name=<name>` to `NEL_OTHER_OVERRIDES` to match `--served-model-name` in your serve command |

---

## 8. Directory Setup for JET Clusters

Before running evaluations on a JET cluster, create writable directories:

```bash
ssh <cluster-login>
mkdir -p /lustre/fsw/portfolios/coreai/users/$USER/cache/huggingface
mkdir -p /lustre/fsw/portfolios/coreai/users/$USER/cache/vllm
mkdir -p /lustre/fsw/portfolios/coreai/users/$USER/nv-eval-rundirs
chmod 777 /lustre/fsw/portfolios/coreai/users/$USER/cache/huggingface
chmod 777 /lustre/fsw/portfolios/coreai/users/$USER/cache/vllm
chmod 777 /lustre/fsw/portfolios/coreai/users/$USER/nv-eval-rundirs
```

`chmod 777` is required because `svc-jet` (JET service account) runs containers and needs write access.
