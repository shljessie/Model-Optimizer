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

"""Checkpoint utilities for bypass distillation."""

import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Type, Union

import torch
from omegaconf import DictConfig
from tqdm import tqdm

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import save_checkpoint
from modelopt.torch.puzzletron.tools.logger import aprint, mprint
from modelopt.torch.puzzletron.tools.robust_json import json_dump

from .stitched_model_factory import StitchedModuleDescriptor


def find_latest_run_dir(run_parent_dir: Union[str, Path]) -> str | None:
    """Find the latest checkpoint directory within a run parent directory."""
    run_parent_dir = Path(run_parent_dir)

    # Check for the "latest" directory
    latest_dir = run_parent_dir / "latest"
    if latest_dir.exists() and (latest_dir / "saving_completed").exists():
        return str(latest_dir)

    # If "latest" doesn't exist, look explicitly into directories with `*iter-*`
    candidate_dirs = [d for d in run_parent_dir.glob("*iter-*") if d.is_dir()]

    if not candidate_dirs:
        return None

    def get_iter_num(dir_name):
        match = re.search(r"iter-(\d+)", dir_name.name)
        return int(match.group(1)) if match else 0

    checkpoint_dirs = sorted(candidate_dirs, key=get_iter_num, reverse=True)
    for latest_dir in checkpoint_dirs:
        if (latest_dir / "saving_completed").exists():
            return str(latest_dir)
    return None


def find_best_run_dir(run_parent_dir: Union[str, Path]) -> str | None:
    """Find the best-validation checkpoint directory within a run parent directory.

    Returns the ``best-iter-*`` directory with the highest iteration number that has a
    ``saving_completed`` marker.  Falls back to ``None`` when no best checkpoint exists
    (e.g. validation was disabled or no improvement was recorded).
    """
    run_parent_dir = Path(run_parent_dir)
    best_dirs = [d for d in run_parent_dir.glob("best-iter-*") if d.is_dir()]
    if not best_dirs:
        return None

    def get_iter_num(d):
        m = re.search(r"iter-(\d+)", d.name)
        return int(m.group(1)) if m else 0

    for best_dir in sorted(best_dirs, key=get_iter_num, reverse=True):
        if (best_dir / "saving_completed").exists():
            return str(best_dir)
    return None


def load_local_state(
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    checkpoint_path: str | Path,
    verbose=True,
) -> None:
    """Load local state from a checkpoint.

    Loads both optimizer and state dicts into stitched module descriptors.
    Modifies stitched_module_descriptors in place.
    """
    device = torch.device(f"cuda:{dist.local_rank()}")
    load_dir = Path(checkpoint_path)

    if not load_dir.exists():
        raise RuntimeError(f'Can\'t load local state. "{load_dir}" does not exist.')

    for stitched_module_name, stitched_module_descriptor in stitched_module_descriptors.items():
        stitched_module = stitched_module_descriptor.stitched_module
        optimizer = stitched_module_descriptor.optimizer

        state_dict_path = load_dir / "stitched" / f"{stitched_module_name}.state_dict.pth"
        if verbose:
            mprint(f"Loading state dict for module {stitched_module_name} from {state_dict_path}")
        loaded_state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
        # Use strict=False so parameters absent in the checkpoint (e.g. newly added adapter
        # keys not yet saved) retain their initialised values rather than raising an error.
        stitched_module.load_state_dict(loaded_state_dict, strict=False)
        del loaded_state_dict

        if optimizer is not None:
            optimizer_state_path = (
                load_dir / "stitched" / f"{stitched_module_name}.optimizer_state.pth"
            )
            if verbose:
                mprint(
                    f"Loading optimizer state for module {stitched_module_name} from {optimizer_state_path}"
                )
            loaded_optimizer_state = torch.load(
                optimizer_state_path, map_location=device, weights_only=True
            )
            optimizer.load_state_dict(loaded_optimizer_state)
            del loaded_optimizer_state


def _save_local_file(obj, save_path: Path | str, overwrite=True):
    save_path = Path(save_path)
    if save_path.exists():
        if not overwrite:
            mprint(f'WARNING: Local save path "{save_path}" already exists. Skipping')
            return
    torch.save(obj, save_path)


def _save_local_state(
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    checkpoint_dir: Path | str,
    overwrite=True,
    verbose=True,
) -> None:
    save_dir = Path(checkpoint_dir) / "stitched"

    if dist.is_master():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Main process creates the directory, so we must wait for it to finish
    dist.barrier()

    for stitched_module_name, stitched_module_descriptor in tqdm(
        stitched_module_descriptors.items()
    ):
        optimizer = stitched_module_descriptor.optimizer

        state_dict_path = save_dir / f"{stitched_module_name}.state_dict.pth"
        if verbose:
            aprint(f"Saving state dict for module {stitched_module_name} to {state_dict_path}")
        state_dict = {
            **stitched_module_descriptor.owned_parameters,
            **stitched_module_descriptor.owned_buffers,
        }
        _save_local_file(state_dict, state_dict_path, overwrite=overwrite)

        if optimizer is not None:
            optimizer_state_path = save_dir / f"{stitched_module_name}.optimizer_state.pth"
            if verbose:
                mprint(
                    f"Saving optimizer state for module {stitched_module_name} to {optimizer_state_path}"
                )
            _save_local_file(optimizer.state_dict(), optimizer_state_path, overwrite=overwrite)

    dist.barrier()


def save_bypass_checkpoint(
    cfg: DictConfig,
    descriptor: Type[ModelDescriptor],
    model: torch.nn.Module,
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    checkpoint_dir: Path | str,
    reference_checkpoint_dir: Optional[Path] = None,
) -> None:
    """Save a bypass distillation checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    mprint("Starting checkpoint save")
    mprint(f"Saving checkpoint to {checkpoint_dir}")

    # Save stitched module states
    _save_local_state(
        stitched_module_descriptors=stitched_module_descriptors,
        checkpoint_dir=checkpoint_dir,
        overwrite=cfg.bypass.model.model_overrides.delete_old_checkpoints,
        verbose=dist.is_master() and False,
    )
    # Save as HF checkpoint
    save_checkpoint(model=model, checkpoint_dir=checkpoint_dir, descriptor=descriptor)

    if dist.is_master():
        # Create 'latest' symlink
        latest_symlink = Path(cfg.bypass.experiment_dir) / "latest"
        latest_symlink.unlink(missing_ok=True)
        latest_symlink.symlink_to(checkpoint_dir.name)
        # Save config args json
        json_dump(cfg.bypass, checkpoint_dir / "args.json")
        # Save completed file
        completed_file = checkpoint_dir / "saving_completed"
        completed_file.touch()

    dist.barrier()
    mprint("Checkpoint save done")
