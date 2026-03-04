#!/usr/bin/env python3
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

"""Training script to verify sparse24 backward correctness with loss curves.

Creates a tiny Llama model, trains it with dense, sparse24, and eager attention,
and plots loss curves side-by-side to visually confirm gradients work.

Usage:
    python train_sparse24.py
    python train_sparse24.py --steps 50 --seq_len 256 --lr 1e-3
    python train_sparse24.py --save_plot loss_curves.png
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from modelopt.torch.sparsity.attention_sparsity.kernels import (
    register_triton_attention,
    set_sparse24,
)


def create_model(
    tmpdir: Path,
    hidden_size=128,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=128,
    max_position_embeddings=512,
    vocab_size=256,
):
    """Create and save a tiny Llama model, return the path."""
    model_dir = tmpdir / "tiny_llama"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir)
    return model_dir


def load_model(model_dir: Path, attn_implementation: str, dtype=torch.float32):
    """Load model with given attn_implementation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        device_map="cuda",
    )
    model.train()
    return model


def train_loop(model, input_ids, labels, steps=30, lr=1e-3):
    """Run training loop, return list of loss values and grad norms."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    grad_norms = []

    for step in range(steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()

        # Compute total grad norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        optimizer.step()
        losses.append(loss.item())
        grad_norms.append(total_norm)

        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:3d}: loss={loss.item():.4f}  grad_norm={total_norm:.4f}")

    return losses, grad_norms


def main(args):
    assert torch.cuda.is_available(), "GPU required"
    register_triton_attention()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_dir = create_model(
            tmpdir,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            intermediate_size=args.hidden_size,
            max_position_embeddings=args.max_pos_embeddings,
            vocab_size=args.vocab_size,
        )

        # Generate random input (same for all modes)
        torch.manual_seed(123)
        input_ids = torch.randint(0, args.vocab_size, (1, args.seq_len), device="cuda")
        labels = input_ids.clone()

        print(
            f"Config: seq_len={args.seq_len}, hidden={args.hidden_size}, "
            f"layers={args.num_layers}, heads={args.num_heads}, "
            f"kv_heads={args.num_kv_heads}, steps={args.steps}, lr={args.lr}"
        )
        print(f"Input shape: {input_ids.shape}\n")

        results = {}

        # 1) Eager (PyTorch native) — baseline
        print("=== Training: eager (PyTorch native) ===")
        model = load_model(model_dir, attn_implementation="eager")
        losses, grad_norms = train_loop(model, input_ids, labels, steps=args.steps, lr=args.lr)
        results["eager"] = {"losses": losses, "grad_norms": grad_norms}
        del model
        torch.cuda.empty_cache()

        # 2) Triton dense (no sparsity)
        print("\n=== Training: triton dense ===")
        model = load_model(model_dir, attn_implementation="modelopt_triton")
        losses, grad_norms = train_loop(model, input_ids, labels, steps=args.steps, lr=args.lr)
        results["triton_dense"] = {"losses": losses, "grad_norms": grad_norms}
        del model
        torch.cuda.empty_cache()

        # 3) Triton sparse24 (skip_diagonal_blocks=True)
        print("\n=== Training: triton sparse24 (skip_diag=True) ===")
        model = load_model(model_dir, attn_implementation="modelopt_triton")
        set_sparse24(model, apply_sparse24=True, skip_diagonal_blocks=True)
        losses, grad_norms = train_loop(model, input_ids, labels, steps=args.steps, lr=args.lr)
        results["sparse24_skip_diag"] = {"losses": losses, "grad_norms": grad_norms}
        del model
        torch.cuda.empty_cache()

        # 4) Triton sparse24 (skip_diagonal_blocks=False)
        print("\n=== Training: triton sparse24 (skip_diag=False) ===")
        model = load_model(model_dir, attn_implementation="modelopt_triton")
        set_sparse24(model, apply_sparse24=True, skip_diagonal_blocks=False)
        losses, grad_norms = train_loop(model, input_ids, labels, steps=args.steps, lr=args.lr)
        results["sparse24_full"] = {"losses": losses, "grad_norms": grad_norms}
        del model
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, data in results.items():
        l0, lf = data["losses"][0], data["losses"][-1]
        decreased = "YES" if lf < l0 else "NO"
        pct = (l0 - lf) / l0 * 100
        print(f"  {name:25s}: {l0:.4f} -> {lf:.4f}  (decrease={decreased}, {pct:+.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "eager": "#1f77b4",
        "triton_dense": "#2ca02c",
        "sparse24_skip_diag": "#ff7f0e",
        "sparse24_full": "#d62728",
    }
    labels_map = {
        "eager": "Eager (PyTorch native)",
        "triton_dense": "Triton dense",
        "sparse24_skip_diag": "Sparse24 (skip_diag=True)",
        "sparse24_full": "Sparse24 (skip_diag=False)",
    }

    # Loss curve
    ax = axes[0]
    for name, data in results.items():
        ax.plot(data["losses"], label=labels_map[name], color=colors[name], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Grad norm curve
    ax = axes[1]
    for name, data in results.items():
        ax.plot(data["grad_norms"], label=labels_map[name], color=colors[name], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Norms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sparse24 Training Verification  "
        f"(seq={args.seq_len}, hidden={args.hidden_size}, layers={args.num_layers})",
        fontsize=13,
    )
    plt.tight_layout()

    save_path = args.save_plot
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Sequence length (must be > 128 for sparsity to apply)",
    )
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--max_pos_embeddings", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument(
        "--save_plot", type=str, default="sparse24_training.png", help="Path to save the plot"
    )
    main(parser.parse_args())
