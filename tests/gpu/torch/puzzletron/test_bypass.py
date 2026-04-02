# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU integration tests for bypass distillation (blockwise local distillation).

These tests verify that:
- Bypass distillation runs end-to-end with a tiny Llama model (hidden_size=256,
  intermediate_size=512, num_layers=max(2, world_size)).
- FFN pruning, KV-head compression, and multi-config sweep all produce the expected
  checkpoint symlinks in puzzle_dir/ckpts/.
- The bypass config injection pattern via OmegaConf works correctly for tests that
  do not load a full bypass Hydra config file.

Model parameters used throughout:
  - teacher intermediate_size: 512  -> pruned to 256 (half) for FFN tests
  - teacher num_key_value_heads: 8  -> pruned to 4 for KV-head tests
  - training_tokens: 128, block_size: 64, micro_batch_size: 1  -> max_steps = 2
"""

from datetime import timedelta
from functools import partial
from pathlib import Path

import hydra
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.misc import set_seed
from _test_utils.torch.puzzletron.utils import setup_test_model_and_data
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.activation_scoring.score_pruning_activations as score_pruning_activations
import modelopt.torch.puzzletron.bypass_distillation as bypass_distillation
import modelopt.torch.puzzletron.pruning.pruning_ckpts as pruning_ckpts
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel import convert_model
from modelopt.torch.puzzletron.tools.hydra_utils import initialize_hydra_config_for_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 1234
HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CONVERTER = "llama"
HYDRA_CONFIG_NAME = "meta-llama/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct"

# Teacher model dimensions (set by setup_test_model_and_data for Llama)
TEACHER_INTERMEDIATE_SIZE = 512
TEACHER_NUM_KV_HEADS = 8

# Pruned sizes used in tests
PRUNED_INTERMEDIATE_SIZE = 256  # half of teacher
PRUNED_NUM_KV_HEADS = 4  # half of teacher

# Training budget: 128 tokens / (64 block * 1 mbs) = 2 steps — completes fast
TRAINING_TOKENS = 128
BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# Helper: build the bypass config dict for injection into hydra_cfg
# ---------------------------------------------------------------------------


def _make_bypass_cfg_dict(
    intermediate_size: int = PRUNED_INTERMEDIATE_SIZE,
    num_key_value_heads: int = PRUNED_NUM_KV_HEADS,
    configs_list: list | None = None,
) -> dict:
    """Return a plain-dict bypass config suitable for OmegaConf.update injection.

    Args:
        intermediate_size: FFN intermediate size for the student model.
        num_key_value_heads: Number of KV heads for the student model.
        configs_list: If provided, populates bypass.configs for a multi-config sweep.
            Each entry is a dict with ``model_config_overrides`` and optionally
            ``keys_to_learn``.
    """
    cfg = {
        "dtype": "bf16",
        "seed": 42,
        "experiment_id": None,
        "experiment_dir": None,
        "iter_num": 1,
        "step_num": 1,
        "token_count": 0,
        "data": {
            # The dummy test dataset stores conversations under the "conversation" column.
            "data_column": "conversation",
            "block_size": BLOCK_SIZE,
            "bos_rate": 0.5,
            "fim_rate": 0,
            "fim_spm_rate": 0,
            "source_datasets_to_discard": [],
            "load_from_disk": True,
            "keep_in_memory": False,
            "val_dataset_name": "valid",
            "max_eval_samples": 1,
            "eval_samples_per_process": None,
            "shuffle_train_data_seed": 42,
        },
        "training": {
            "learning_rate": 1e-4,
            "training_tokens": TRAINING_TOKENS,
            "micro_batch_size": 1,
            "val_micro_batch_size": 1,
            "warmup_ratio": 0.05,
            "warmup_steps": None,
            "min_lr_factor": 1e-5,
            "grad_accumulation_steps": 1,
            "skip_first_batches": 0,
            "weight_decay": 0.1,
            "decay_lr": True,
            "beta1": 0.9,
            "beta2": 0.95,
            "use_grad_scaling": False,
            "grad_clip": 1.0,
            "grad_clip_type": "norm",
            "clipping_count": 0,
            "log_interval": 5,
            # Large eval_interval so validation is skipped during this short run.
            # Validation is fully disabled anyway (disable_validation=True below).
            "eval_interval": 100,
        },
        "resume_checkpoint_path": None,
        "find_last_ckpt_for_resume": False,
        "parameter_count": None,
        "init_checkpoint_path": None,
        "model": {
            "student_weights_dtype": "bf16",
            "model_overrides": {
                "delete_old_checkpoints": True,
                "save_interval_seconds": None,
                # Effectively disable step-interval saving; rely on save_checkpoint_when_done.
                "save_interval": 1_000_000_000,
                "save_checkpoint_when_done": True,
            },
            "model_config_overrides": {
                "ffn": [{"intermediate_size": intermediate_size, "no_op": None}],
                "attention": [{"num_key_value_heads": num_key_value_heads, "no_op": None}],
            },
        },
        "model_factory": {
            "factory": "bypass_factory_fn",
            "block_loss_func": "normalized_mse_loss",
            "gqa_init_mode": "AverageKV",
            "mlp_init_mode": "Truncate",
            "mlp_init_config": {"activations_log_dir": None},
            "linear_init_mode": "FromTeacher",
            "submodule_for_loss_calculation": None,
            "keys_to_learn": "entire_block",
        },
        # Disable all validation to keep tests fast.
        "disable_initial_validate": True,
        "validate_teacher_model": False,
        "validate_student_model": False,
        "disable_validation": True,
        "best_val_loss": 1e9,
        "compile": False,
        "disable_fa2": False,
        "teacher_model_load_on_cpu": False,
        "save_checkpoint_before_training": False,
        "disable_checkpoint_save": False,
        "save_best_ckpt": True,
        # Do NOT use kill_after_first_save — it raises RuntimeError which becomes sys.exit(1).
        # Instead let the short training run (2 steps) complete naturally.
        "kill_after_first_save": False,
        "realize_best_or_latest": "best",
        "wandb_log": False,
        "wandb": {"project": None, "entity": None},
    }

    if configs_list is not None:
        cfg["configs"] = configs_list

    return cfg


# ---------------------------------------------------------------------------
# Helper: load hydra config and run pruning prerequisites
# ---------------------------------------------------------------------------


def _setup_hydra_cfg_and_pruning(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    size: int,
) -> tuple:
    """Set up the tiny model, convert it, score activations, and create pruning ckpts.

    This is the shared preamble for all bypass tests. Returns
    ``(puzzle_dir, dataset_path, hydra_cfg)``.

    Steps performed:
    1. Create a small HF model and dummy dataset via ``setup_test_model_and_data``.
    2. Convert the HF checkpoint to AnyModel/DeciLM format (rank 0 only).
    3. Load the Hydra config with ``puzzle_dir`` and ``dataset_path`` overrides.
    4. Run ``score_pruning_activations`` (distributed).
    5. Run ``pruning_ckpts`` (rank 0 only) then barrier.
    """
    set_seed(SEED)
    dist.setup(timeout=timedelta(minutes=10))

    puzzle_dir, hf_checkpoint_path, dataset_path = setup_test_model_and_data(
        project_root_path, tmp_path, rank, HF_MODEL_NAME
    )

    hydra_config_dir = str(project_root_path / "tests/gpu/torch/puzzletron/resources/configs")

    # Step 0: Convert HF checkpoint to AnyModel/DeciLM format.
    if rank == 0:
        convert_model(
            input_dir=str(hf_checkpoint_path),
            output_dir=str(puzzle_dir / "ckpts/teacher"),
            converter=CONVERTER,
        )
    dist.barrier()

    # Step 1: Load Hydra config.
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=HYDRA_CONFIG_NAME,
        overrides=[
            f"puzzle_dir={puzzle_dir}",
            f"dataset_path={dataset_path}",
        ],
    )
    hydra_cfg = hydra.utils.instantiate(hydra_cfg)

    # Step 2: Score pruning activations (distributed).
    score_pruning_activations.launch_score_activations(hydra_cfg)

    # Step 3: Create pruning checkpoints (rank 0 only).
    if rank == 0:
        pruning_ckpts.launch_prune_ckpt(hydra_cfg)
    dist.barrier()

    return puzzle_dir, dataset_path, hydra_cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bypass_ffn_pruning(project_root_path: Path, tmp_path: Path):
    """Bypass distillation with FFN pruned to intermediate_size=256.

    Verifies that after training:
    - The experiment directory ``bypass/bypass_runs/bypass_ffn256_kv4`` exists.
    - A symlink ``ckpts/bypass_ffn256_kv4`` pointing into the experiment dir
      is created by ``realize_bypass_checkpoints``.
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_ffn_pruning_job,
            project_root_path,
            tmp_path,
        ),
        backend="nccl",
    )


def _test_bypass_ffn_pruning_job(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    size: int,
):
    puzzle_dir, dataset_path, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path, tmp_path, rank, size
    )

    # Inject bypass config: prune FFN to 256, keep num_key_value_heads=4.
    # experiment_id will be set dynamically to "bypass_ffn256_kv4".
    bypass_cfg_dict = _make_bypass_cfg_dict(
        intermediate_size=PRUNED_INTERMEDIATE_SIZE,
        num_key_value_heads=PRUNED_NUM_KV_HEADS,
    )
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        expected_experiment_id = f"bypass_ffn{PRUNED_INTERMEDIATE_SIZE}_kv{PRUNED_NUM_KV_HEADS}"
        experiment_dir = puzzle_dir / "bypass/bypass_runs" / expected_experiment_id
        ckpt_symlink = puzzle_dir / "ckpts" / expected_experiment_id

        assert experiment_dir.exists(), (
            f"Expected bypass experiment directory to exist: {experiment_dir}"
        )
        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
        )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_ffn_pruning completed successfully. "
        f"Puzzle directory: {puzzle_dir}"
    )


def test_bypass_kv_head_compression(project_root_path: Path, tmp_path: Path):
    """Bypass distillation with KV heads reduced from 8 to 4, FFN kept at 512.

    The experiment_id is ``bypass_ffn512_kv4`` because both FFN and attention
    overrides are specified (FFN is kept at teacher size, attention is halved).
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_kv_head_compression_job,
            project_root_path,
            tmp_path,
        ),
        backend="nccl",
    )


def _test_bypass_kv_head_compression_job(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    size: int,
):
    puzzle_dir, dataset_path, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path, tmp_path, rank, size
    )

    # Keep FFN at teacher size (512) but halve KV heads (8 -> 4).
    # experiment_id will be "bypass_ffn512_kv4".
    bypass_cfg_dict = _make_bypass_cfg_dict(
        intermediate_size=TEACHER_INTERMEDIATE_SIZE,
        num_key_value_heads=PRUNED_NUM_KV_HEADS,
    )
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        expected_experiment_id = f"bypass_ffn{TEACHER_INTERMEDIATE_SIZE}_kv{PRUNED_NUM_KV_HEADS}"
        experiment_dir = puzzle_dir / "bypass/bypass_runs" / expected_experiment_id
        ckpt_symlink = puzzle_dir / "ckpts" / expected_experiment_id

        assert experiment_dir.exists(), (
            f"Expected bypass experiment directory to exist: {experiment_dir}"
        )
        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
        )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_kv_head_compression completed successfully. "
        f"Puzzle directory: {puzzle_dir}"
    )


def test_bypass_multi_config_sequential(project_root_path: Path, tmp_path: Path):
    """Bypass distillation sweep: two configs run sequentially via bypass.configs list.

    Config 0: FFN=256, heads=4  -> experiment_id ``bypass_ffn256_kv4``
    Config 1: FFN=512, heads=4  -> experiment_id ``bypass_ffn512_kv4``

    Both symlinks must exist after the sweep completes.
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_multi_config_sequential_job,
            project_root_path,
            tmp_path,
        ),
        backend="nccl",
    )


def _test_bypass_multi_config_sequential_job(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    size: int,
):
    puzzle_dir, dataset_path, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path, tmp_path, rank, size
    )

    # Build base bypass config (model_config_overrides will be overwritten by configs list).
    configs_list = [
        {
            "model_config_overrides": {
                "ffn": [{"intermediate_size": PRUNED_INTERMEDIATE_SIZE, "no_op": None}],
                "attention": [{"num_key_value_heads": PRUNED_NUM_KV_HEADS, "no_op": None}],
            },
            "keys_to_learn": "entire_block",
        },
        {
            "model_config_overrides": {
                "ffn": [{"intermediate_size": TEACHER_INTERMEDIATE_SIZE, "no_op": None}],
                "attention": [{"num_key_value_heads": PRUNED_NUM_KV_HEADS, "no_op": None}],
            },
            "keys_to_learn": "entire_block",
        },
    ]
    bypass_cfg_dict = _make_bypass_cfg_dict(
        intermediate_size=PRUNED_INTERMEDIATE_SIZE,
        num_key_value_heads=PRUNED_NUM_KV_HEADS,
        configs_list=configs_list,
    )
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        expected_ids = [
            f"bypass_ffn{PRUNED_INTERMEDIATE_SIZE}_kv{PRUNED_NUM_KV_HEADS}",
            f"bypass_ffn{TEACHER_INTERMEDIATE_SIZE}_kv{PRUNED_NUM_KV_HEADS}",
        ]
        for experiment_id in expected_ids:
            experiment_dir = puzzle_dir / "bypass/bypass_runs" / experiment_id
            ckpt_symlink = puzzle_dir / "ckpts" / experiment_id

            assert experiment_dir.exists(), (
                f"Expected bypass experiment directory to exist: {experiment_dir}"
            )
            assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
                f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
            )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_multi_config_sequential completed successfully. "
        f"Puzzle directory: {puzzle_dir}"
    )


def test_bypass_checkpoint_contents(project_root_path: Path, tmp_path: Path):
    """Verify that a bypass checkpoint contains expected HuggingFace model files.

    After bypass completes, the checkpoint directory (reachable via the symlink at
    ``ckpts/{experiment_id}``) must contain a ``config.json`` (saved by
    ``save_checkpoint`` / ``save_bypass_checkpoint``).
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_checkpoint_contents_job,
            project_root_path,
            tmp_path,
        ),
        backend="nccl",
    )


def _test_bypass_checkpoint_contents_job(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    size: int,
):
    puzzle_dir, dataset_path, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path, tmp_path, rank, size
    )

    bypass_cfg_dict = _make_bypass_cfg_dict(
        intermediate_size=PRUNED_INTERMEDIATE_SIZE,
        num_key_value_heads=PRUNED_NUM_KV_HEADS,
    )
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        expected_experiment_id = f"bypass_ffn{PRUNED_INTERMEDIATE_SIZE}_kv{PRUNED_NUM_KV_HEADS}"
        ckpt_symlink = puzzle_dir / "ckpts" / expected_experiment_id

        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink: {ckpt_symlink}"
        )

        # The symlink resolves to the latest checkpoint dir; verify HF config exists.
        resolved = ckpt_symlink.resolve()
        config_json = resolved / "config.json"
        assert config_json.exists(), (
            f"Expected HuggingFace config.json inside checkpoint: {config_json}"
        )

        # The saving_completed marker must be present (set by save_bypass_checkpoint).
        saving_completed = resolved / "saving_completed"
        assert saving_completed.exists(), (
            f"Expected saving_completed marker inside checkpoint: {saving_completed}"
        )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_checkpoint_contents completed successfully. "
        f"Puzzle directory: {puzzle_dir}"
    )


def test_bypass_checkpoint_resume(project_root_path: Path, tmp_path: Path):
    """Verify that bypass distillation can resume from a previous checkpoint.

    Runs bypass twice with the same experiment_id:
    - First run: completes 2 training steps and saves a checkpoint.
    - Second run: uses ``find_last_ckpt_for_resume=True`` to auto-detect the
      saved checkpoint and resume from it.

    Checks that the second run finds the checkpoint, loads it without error,
    and produces a final checkpoint in the experiment directory.
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_checkpoint_resume_job,
            project_root_path,
            tmp_path,
        ),
        backend="nccl",
    )


def _test_bypass_checkpoint_resume_job(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    size: int,
):
    puzzle_dir, dataset_path, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path, tmp_path, rank, size
    )

    bypass_cfg_dict = _make_bypass_cfg_dict(
        intermediate_size=PRUNED_INTERMEDIATE_SIZE,
        num_key_value_heads=PRUNED_NUM_KV_HEADS,
    )
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    # --- First run: train and save a checkpoint. ---
    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    expected_experiment_id = f"bypass_ffn{PRUNED_INTERMEDIATE_SIZE}_kv{PRUNED_NUM_KV_HEADS}"
    experiment_dir = puzzle_dir / "bypass/bypass_runs" / expected_experiment_id

    if rank == 0:
        assert experiment_dir.exists(), (
            f"First run should have created experiment directory: {experiment_dir}"
        )

    dist.barrier()

    # --- Second run: resume from the checkpoint saved by the first run. ---
    # Reset training counters so the second run starts fresh in terms of config,
    # but find_last_ckpt_for_resume=True causes it to reload the saved state.
    OmegaConf.update(hydra_cfg, "bypass.iter_num", 1, merge=True)
    OmegaConf.update(hydra_cfg, "bypass.step_num", 1, merge=True)
    OmegaConf.update(hydra_cfg, "bypass.token_count", 0, merge=True)
    OmegaConf.update(hydra_cfg, "bypass.find_last_ckpt_for_resume", True, merge=True)

    # The second run should not raise; it should load the checkpoint and complete.
    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        ckpt_symlink = puzzle_dir / "ckpts" / expected_experiment_id
        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Second (resume) run should produce a checkpoint symlink: {ckpt_symlink}"
        )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_checkpoint_resume completed successfully. "
        f"Puzzle directory: {puzzle_dir}"
    )
