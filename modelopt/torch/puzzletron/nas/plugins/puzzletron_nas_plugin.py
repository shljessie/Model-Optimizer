# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Puzzletron NAS plugin for the Modelopt framework (based on Puzzle algorithm: https://arxiv.org/abs/2411.19146).

It is used by mtn.convert() to convert a model from HF format to Puzzletron heterogeneous format + do pruning scoring
and save pruned checkpoints, and by mtn.search() to perform the MIP-based NAS search.
"""

import datetime
from pathlib import Path

import hydra
import torch
from torch import nn

import modelopt.torch.puzzletron.bypass_distillation as bypass_distillation
import modelopt.torch.puzzletron.mip.mip_and_realize_models as mip_and_realize_models
import modelopt.torch.puzzletron.pruning.pruning_ckpts as pruning_ckpts
import modelopt.torch.puzzletron.scoring.scoring as scoring
import modelopt.torch.utils.distributed as dist
from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    MetadataDict,
    ModeDescriptor,
    RestoreEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher, SearchStateDict
from modelopt.torch.puzzletron import build_library_and_stats
from modelopt.torch.puzzletron.activation_scoring import score_pruning_activations
from modelopt.torch.puzzletron.anymodel.converter import ConverterFactory
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
from modelopt.torch.puzzletron.tools.hydra_utils import initialize_hydra_config_for_dir
from modelopt.torch.puzzletron.tools.logger import mprint


class PuzzletronModel(nn.Module):
    pass  # No model implementation is needed for the puzzletron mode


class PuzzletronConfig(ModeloptBaseConfig):
    """Configuration for Puzzletron NAS algorithm."""

    # Input model path to compress in the HF format
    input_model_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Hydra config directory containing the search space definition
    hydra_config_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Hydra config name containing the search space definition
    hydra_config_name: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Directory to save the compressed model and intermediate results
    puzzle_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Dataset path to use for scoring in prunining and NAS search
    dataset_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )


def _total_steps(hydra_cfg) -> int:
    """Return total pipeline step count: 9 with bypass, 8 without.

    Steps:
      1  starting (main.py)
      2  convert model
      3  score pruning activations
      4  prune checkpoints
     [5  bypass distillation — only when bypass is configured]
      5/6  build replacement library & subblock stats
      6/7  calculate one block scores
      7/8  MIP and realize models
      8/9  completed (main.py)
    """
    return 9 if hydra_cfg.get("bypass", None) is not None else 8


def convert_puzzletron_model(model: nn.Module, config: PuzzletronConfig) -> ConvertReturnType:
    """1. Convert the model from HF format to AnyModel format.
    2. Score the pruning activations.
    3. Prune the model and save pruned checkpoints.
    4. (Optional) Run bypass distillation.

    The output of this step will be used by mnt.search() to perform the NAS search.
    """
    # Required for mtn.search() to read NAS configuration
    model.hydra_config_dir = config.hydra_config_dir
    model.hydra_config_name = config.hydra_config_name
    model.puzzle_dir = config.puzzle_dir
    model.dataset_path = config.dataset_path

    # Load hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=config.hydra_config_dir,
        config_name=config.hydra_config_name,
        overrides=[
            f"puzzle_dir={config.puzzle_dir}",
            f"dataset_path={config.dataset_path}",
        ],
    )
    # Instantiate nested Hydra configs (e.g., pruning_mixin, hook_class)
    hydra_cfg = hydra.utils.instantiate(hydra_cfg)

    has_bypass = hydra_cfg.get("bypass", None) is not None
    n = _total_steps(hydra_cfg)

    puzzle_dir = Path(config.puzzle_dir)

    # Step 2: Convert HuggingFace model to Puzzletron heterogeneous format
    hf_ckpt_teacher_dir = "ckpts/teacher"  # TODO: make it configurable
    teacher_dir = puzzle_dir / hf_ckpt_teacher_dir
    convert_marker = puzzle_dir / "convert.complete"
    if dist.is_master():
        # Check durable marker first; fall back to artifact existence for backward compat
        already_done = convert_marker.exists() or (teacher_dir / "config.json").exists()
        if already_done:
            mprint(
                f"Puzzletron Progress 2/{n}: teacher checkpoint already exists, skipping conversion"
            )
        else:
            mprint(
                f"Puzzletron Progress 2/{n}: converting model to Puzzletron heterogeneous format (single-gpu)"
            )

            # Get descriptor and converter from the hydra config
            descriptor_name = hydra_cfg.descriptor
            descriptor = ModelDescriptorFactory.get(descriptor_name)
            converter = ConverterFactory.get(descriptor_name)

            # Auto-download from HuggingFace if path doesn't exist locally.
            # input_model_path is only used on rank 0 (conversion is single-process);
            # other ranks wait at the dist.barrier() below and never read this variable.
            input_model_path = config.input_model_path
            if not Path(input_model_path).exists():
                from huggingface_hub import snapshot_download

                if input_model_path.startswith("https://huggingface.co/"):
                    model_id = "/".join(input_model_path.rstrip("/").split("/")[-2:])
                else:
                    model_id = input_model_path  # assume HF model ID like "org/model-name"
                mprint(
                    f"Downloading HuggingFace model '{model_id}' — this may take several minutes "
                    f"for large models. Other ranks are waiting at a barrier."
                )
                input_model_path = snapshot_download(repo_id=model_id)
                mprint(f"Downloaded to: {input_model_path}")

            converter.convert(
                descriptor=descriptor,
                input_dir=Path(input_model_path),
                output_dir=teacher_dir,
            )
            convert_marker.touch()
    dist.barrier()

    # Step 3: Score pruning activations (distributed processing)
    activations_log_dir = Path(hydra_cfg.pruning.activations_log_dir)
    score_marker = puzzle_dir / "score_activations.complete"
    # Check durable marker first; fall back to artifact existence for backward compat
    already_scored = score_marker.exists() or (
        activations_log_dir.exists() and any(activations_log_dir.glob("rank_*.pth"))
    )
    if already_scored:
        mprint(
            f"Puzzletron Progress 3/{n}: pruning activation scores already exist, skipping scoring"
        )
        dist.barrier()
    else:
        mprint(f"Puzzletron Progress 3/{n}: scoring pruning activations (multi-gpu)")
        score_pruning_activations.launch_score_activations(hydra_cfg)
        if dist.is_master():
            score_marker.touch()
        dist.barrier()

    # Step 4: Prune the model and save pruned checkpoints (single process)
    pruned_ckpts_dir = Path(hydra_cfg.pruning.pruned_ckpts_output_dir)
    prune_marker = puzzle_dir / "prune.complete"
    if dist.is_master():
        # Check durable marker first; fall back to artifact existence for backward compat
        already_pruned = prune_marker.exists() or (
            pruned_ckpts_dir.exists() and any(pruned_ckpts_dir.iterdir())
        )
        if already_pruned:
            mprint(f"Puzzletron Progress 4/{n}: pruned checkpoints already exist, skipping pruning")
        else:
            mprint(
                f"Puzzletron Progress 4/{n}: pruning the model and saving pruned checkpoints (single-gpu)"
            )
            pruning_ckpts.launch_prune_ckpt(hydra_cfg)
            prune_marker.touch()
    dist.barrier()

    # Step 5: Bypass distillation (optional, distributed processing)
    if has_bypass:
        mprint(f"Puzzletron Progress 5/{n}: running bypass distillation (multi-gpu)")
        bypass_distillation.launch_bypass_distillation(hydra_cfg)

    return model, {}


def restore_puzzletron_model(
    model: nn.Module, config: PuzzletronConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore is not needed for the puzzletron mode as we are not saving any model state"""
    return model


@NASModeRegistry.register_mode
class PuzzletronDescriptor(ModeDescriptor):
    """Descriptor for the Puzzletron mode."""

    @property
    def name(self) -> str:
        """String identifier for this mode."""
        return "puzzletron"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Configuration class for this mode."""
        return PuzzletronConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Return the associated searcher implementation."""

        return PuzzletronSearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """Entrypoint to convert a model."""
        return convert_puzzletron_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """Entrypoint to restore a model."""
        return restore_puzzletron_model

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode.
        For now, this will be a no-op as there is no modelopt's concept of search space defined
        for the puzzletron algorithm.
        """
        return "export_nas"


class PuzzletronSearcher(BaseSearcher):
    """Runs NAS search for the Puzzletron mode."""

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Not needed for the puzzletron mode as we are not saving any model state"""
        return {}

    def run_search(self) -> None:
        # Load hydra config
        hydra_cfg = initialize_hydra_config_for_dir(
            config_dir=self.model.hydra_config_dir,
            config_name=self.model.hydra_config_name,
            overrides=[
                f"puzzle_dir={self.model.puzzle_dir}",
                f"dataset_path={self.model.dataset_path}",
            ],
        )
        # Instantiate nested Hydra configs (e.g., pruning_mixin, hook_class)
        hydra_cfg = hydra.utils.instantiate(hydra_cfg)

        has_bypass = hydra_cfg.get("bypass", None) is not None
        n = _total_steps(hydra_cfg)
        # With bypass:    library=6, scoring=7, mip=8  (out of 9)
        # Without bypass: library=5, scoring=6, mip=7  (out of 8)
        library_step = 6 if has_bypass else 5
        scoring_step = 7 if has_bypass else 6
        mip_step = 8 if has_bypass else 7

        # Build replacement library and subblock statistics (single process)
        puzzle_dir = Path(self.model.puzzle_dir)
        replacement_library_path = puzzle_dir / "replacement_library.json"
        subblock_stats_path = puzzle_dir / hydra_cfg.calc_subblock_stats.subblock_stats_filename
        library_marker = puzzle_dir / "library.complete"
        if dist.is_master():
            # Check durable marker first; fall back to artifact existence for backward compat
            already_built = library_marker.exists() or (
                replacement_library_path.exists() and subblock_stats_path.exists()
            )
            if already_built:
                mprint(
                    f"Puzzletron Progress {library_step}/{n}: replacement library and subblock stats already exist, skipping"
                )
            else:
                mprint(
                    f"Puzzletron Progress {library_step}/{n}: building replacement library and subblock statistics (single-gpu)"
                )
                build_library_and_stats.launch_build_library_and_stats(hydra_cfg)
                library_marker.touch()
        dist.barrier()

        # Calculate one block scores (distributed processing)
        mprint(f"Puzzletron Progress {scoring_step}/{n}: calculating one block scores (multi-gpu)")
        scoring.launch_scoring(hydra_cfg)

        # MIP search and realize models (distributed processing)
        mprint(f"Puzzletron Progress {mip_step}/{n}: running MIP and realizing models (multi-gpu)")
        mip_and_realize_models.launch_mip_and_realize_model(hydra_cfg)
