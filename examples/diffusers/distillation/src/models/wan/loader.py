"""Wan2.2 model loader."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WanModelLoader:
    def load_transformer(
        self,
        path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        from wan.modules.model import WanModel

        logger.info(f"Loading WanModel from {path}")
        model = WanModel.from_pretrained(str(path))
        model = model.to(device=device, dtype=dtype)
        return model
