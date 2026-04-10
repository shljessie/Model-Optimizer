#!/usr/bin/env python3
"""Split fused MoE expert weights for vLLM compatibility.

ModelOpt with transformers 5.x exports fused 3D tensors:
  experts.gate_up_proj  [num_experts, 2*intermediate, hidden]
  experts.down_proj     [num_experts, hidden, intermediate]

vLLM's modelopt FP4 loader expects per-expert split:
  experts.{id}.gate_proj.weight  [intermediate, hidden]
  experts.{id}.up_proj.weight    [intermediate, hidden]
  experts.{id}.down_proj.weight  [hidden, intermediate]

This script converts between the two formats.

Usage:
  python split_fused_moe_weights.py <checkpoint_dir> [--output <output_dir>]
"""

import argparse
import json
import os
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


def split_fused_weights(ckpt_dir: str, output_dir: str | None = None):
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir) if output_dir else ckpt_dir

    if output_dir != ckpt_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single shard — no index file
        shard_files = list(ckpt_dir.glob("model*.safetensors"))
        if len(shard_files) != 1:
            raise FileNotFoundError(f"Expected single safetensors file, found {len(shard_files)}")
        index = None
        shard_map = {shard_files[0].name: shard_files[0].name}
    else:
        index = json.load(open(index_path))
        shard_map = {}
        for key, shard in index["weight_map"].items():
            shard_map.setdefault(shard, [])
            shard_map[shard].append(key)

    # Check if conversion is needed
    all_keys = set()
    if index:
        all_keys = set(index["weight_map"].keys())
    else:
        weights = load_file(str(shard_files[0]))
        all_keys = set(weights.keys())

    fused_keys = [k for k in all_keys if ".experts.gate_up_proj" in k or ".experts.down_proj" in k]
    if not fused_keys:
        print("No fused MoE expert weights found. Nothing to do.")
        return

    # Detect num_experts from config
    config = json.load(open(ckpt_dir / "config.json"))
    # Try various config paths for num_experts
    num_experts = (
        config.get("num_experts")
        or config.get("num_local_experts")
        or config.get("text_config", {}).get("num_experts")
        or config.get("text_config", {}).get("num_local_experts")
    )
    if num_experts is None:
        raise ValueError("Could not find num_experts in config.json")

    print(f"Found {len(fused_keys)} fused MoE keys, {num_experts} experts")

    # Process each shard
    new_weight_map = {}
    total_split = 0

    if index:
        shards_to_process = set(index["weight_map"][k] for k in fused_keys)
    else:
        shards_to_process = {shard_files[0].name}

    for shard_name in sorted(shards_to_process):
        shard_path = ckpt_dir / shard_name
        print(f"\nProcessing {shard_name}...")
        weights = load_file(str(shard_path))

        new_weights = {}
        for key, tensor in weights.items():
            if ".experts.gate_up_proj" in key:
                # Split gate_up_proj [num_experts, 2*intermediate, ...] → per-expert gate_proj + up_proj
                prefix = key.replace(".experts.gate_up_proj", "")
                suffix = ""

                # Handle weight vs scale vs scale_2
                # experts.gate_up_proj → experts.{id}.gate_proj.weight
                # experts.gate_up_proj_scale → experts.{id}.gate_proj.weight_scale
                # experts.gate_up_proj_scale_2 → experts.{id}.gate_proj.weight_scale_2
                is_scale_2 = key.endswith("_scale_2")
                is_scale = key.endswith("_scale") and not is_scale_2

                if is_scale_2:
                    base_key = key.replace("_scale_2", "")
                    weight_suffix = "_scale_2"
                elif is_scale:
                    base_key = key.replace("_scale", "")
                    weight_suffix = "_scale"
                else:
                    base_key = key
                    weight_suffix = ""

                base_prefix = base_key.replace(".experts.gate_up_proj", "")

                if tensor.dim() == 1:
                    # Per-expert scalar (e.g., scale_2): [num_experts * 2] → split
                    assert tensor.shape[0] == num_experts * 2, (
                        f"Expected {num_experts * 2} for scale_2, got {tensor.shape}")
                    for eid in range(num_experts):
                        gate_name = f"{base_prefix}.experts.{eid}.gate_proj.weight{weight_suffix}"
                        up_name = f"{base_prefix}.experts.{eid}.up_proj.weight{weight_suffix}"
                        new_weights[gate_name] = tensor[eid * 2].unsqueeze(0)
                        new_weights[up_name] = tensor[eid * 2 + 1].unsqueeze(0)
                        total_split += 2

                elif tensor.dim() == 3:
                    # [num_experts, 2*intermediate, hidden] → split dim 1
                    intermediate = tensor.shape[1] // 2
                    for eid in range(num_experts):
                        gate_name = f"{base_prefix}.experts.{eid}.gate_proj.weight{weight_suffix}"
                        up_name = f"{base_prefix}.experts.{eid}.up_proj.weight{weight_suffix}"
                        new_weights[gate_name] = tensor[eid, :intermediate]
                        new_weights[up_name] = tensor[eid, intermediate:]
                        total_split += 2

                elif tensor.dim() == 2:
                    # [num_experts, 2] → per-expert scalar pair
                    for eid in range(num_experts):
                        gate_name = f"{base_prefix}.experts.{eid}.gate_proj.weight{weight_suffix}"
                        up_name = f"{base_prefix}.experts.{eid}.up_proj.weight{weight_suffix}"
                        new_weights[gate_name] = tensor[eid, 0:1]
                        new_weights[up_name] = tensor[eid, 1:2]
                        total_split += 2

                else:
                    print(f"  WARNING: unexpected dim {tensor.dim()} for {key}, keeping as-is")
                    new_weights[key] = tensor

            elif ".experts.down_proj" in key:
                # Split down_proj [num_experts, hidden, intermediate] → per-expert
                is_scale_2 = key.endswith("_scale_2")
                is_scale = key.endswith("_scale") and not is_scale_2

                if is_scale_2:
                    base_key = key.replace("_scale_2", "")
                    weight_suffix = "_scale_2"
                elif is_scale:
                    base_key = key.replace("_scale", "")
                    weight_suffix = "_scale"
                else:
                    base_key = key
                    weight_suffix = ""

                base_prefix = base_key.replace(".experts.down_proj", "")

                if tensor.dim() == 3:
                    for eid in range(num_experts):
                        new_name = f"{base_prefix}.experts.{eid}.down_proj.weight{weight_suffix}"
                        new_weights[new_name] = tensor[eid]
                        total_split += 1
                elif tensor.dim() == 1:
                    for eid in range(num_experts):
                        new_name = f"{base_prefix}.experts.{eid}.down_proj.weight{weight_suffix}"
                        new_weights[new_name] = tensor[eid].unsqueeze(0)
                        total_split += 1
                else:
                    print(f"  WARNING: unexpected dim {tensor.dim()} for {key}, keeping as-is")
                    new_weights[key] = tensor
            else:
                new_weights[key] = tensor

        # Save
        out_path = output_dir / shard_name
        save_file(new_weights, str(out_path))
        print(f"  Saved {len(new_weights)} keys to {out_path}")

        for k in new_weights:
            new_weight_map[k] = shard_name

    # Copy non-MoE shards if output dir is different
    if output_dir != ckpt_dir and index:
        all_shards = set(index["weight_map"].values())
        for shard_name in all_shards - shards_to_process:
            import shutil
            src = ckpt_dir / shard_name
            dst = output_dir / shard_name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"Copied {shard_name}")
            # Add keys from this shard
            weights = load_file(str(src))
            for k in weights:
                new_weight_map[k] = shard_name

    # Update index
    if index:
        # Rebuild with non-MoE keys from original
        for k, shard in index["weight_map"].items():
            if k not in new_weight_map and ".experts.gate_up_proj" not in k and ".experts.down_proj" not in k:
                new_weight_map[k] = shard

        new_index = {
            "metadata": index.get("metadata", {}),
            "weight_map": dict(sorted(new_weight_map.items())),
        }
        out_index = output_dir / "model.safetensors.index.json"
        json.dump(new_index, open(out_index, "w"), indent=2)
        print(f"\nUpdated index: {len(new_index['weight_map'])} keys")

    # Copy non-weight files
    if output_dir != ckpt_dir:
        for f in ckpt_dir.iterdir():
            if f.suffix not in (".safetensors",) and f.name != "model.safetensors.index.json":
                dst = output_dir / f.name
                if not dst.exists():
                    import shutil
                    shutil.copy2(f, dst)

    print(f"\nDone. Split {total_split} tensors from {len(fused_keys)} fused keys.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split fused MoE weights for vLLM")
    parser.add_argument("checkpoint_dir", help="Path to the quantized checkpoint")
    parser.add_argument("--output", help="Output directory (default: in-place)")
    args = parser.parse_args()

    split_fused_weights(args.checkpoint_dir, args.output)
