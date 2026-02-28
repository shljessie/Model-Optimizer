"""Wan2.2 training strategy.

Handles the conversion between the unified batch format [B, C, F, H, W]
and WanModel's list-of-tensors interface, plus noise application and loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...interfaces import StrategyOutputs


class WanStrategy:
    """Flow matching training strategy for Wan2.2.

    WanModel expects List[Tensor] inputs (one per batch item) with shapes:
    - x: List[[C_in, F, H, W]]
    - context: List[[L, C=4096]]
    - t: [B] timesteps
    - seq_len: int (max sequence length after patchification)

    The patch embedding is Conv3d with kernel (1,2,2), stride (1,2,2),
    so the patchified sequence length = F * (H/2) * (W/2).
    """

    MOCK_LATENT_SHAPE: tuple[int, ...] = (48, 4, 32, 32)
    MOCK_TEXT_EMBED_DIM: int = 4096

    def __init__(self, patch_size: tuple[int, int, int] = (1, 2, 2)) -> None:
        self.patch_size = patch_size

    def prepare_inputs(
        self,
        batch: dict[str, Tensor],
        noise: Tensor,
        timesteps: Tensor,
    ) -> StrategyOutputs:
        latents = batch["latents"]      # [B, C, F, H, W]
        text_embeds = batch["text_embeds"]  # [B, L, D]

        b, _c, n_f, n_h, n_w = latents.shape

        # Flow matching noise application: noisy = (1 - sigma) * clean + sigma * noise
        sigmas = timesteps.view(-1, 1, 1, 1, 1)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        # Velocity prediction target: noise - clean
        targets = noise - latents

        # Compute max patchified sequence length
        pt, ph, pw = self.patch_size
        seq_len = (n_f // pt) * (n_h // ph) * (n_w // pw)

        # Convert batch tensors to per-item lists (WanModel interface)
        x_list = list(noisy_latents.unbind(0))
        ctx_list = list(text_embeds.unbind(0))

        # Map sigma in [0,1] -> timestep range [0, 1000) for sinusoidal embeddings.
        # Keep as float -- WanModel.forward handles expansion to [B, seq_len] internally.
        t = timesteps * 1000.0

        forward_kwargs = {
            "x": x_list,
            "t": t,
            "context": ctx_list,
            "seq_len": seq_len,
        }

        loss_mask = torch.ones(b, dtype=torch.float32, device=latents.device)

        return StrategyOutputs(
            noisy_input=x_list,
            targets=targets,
            loss_mask=loss_mask,
            forward_kwargs=forward_kwargs,
        )

    def forward_model(self, model: nn.Module, inputs: StrategyOutputs) -> Tensor:
        output_list = model(**inputs.forward_kwargs)
        # Stack per-item outputs into a single tensor [B, C_out, F, H', W']
        return torch.stack(output_list)

    def compute_task_loss(
        self, pred: Tensor, targets: Tensor, loss_mask: Tensor
    ) -> Tensor:
        loss = (pred - targets).pow(2).mean()
        return loss


def create_wan_strategy(**kwargs) -> WanStrategy:
    return WanStrategy()
