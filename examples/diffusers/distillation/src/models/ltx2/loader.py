"""LTX-2 model loader.

Uses SingleGPUModelBuilder from ltx-core with the LTXModelConfigurator
and COMFY key renaming map for checkpoint loading.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class LTX2ModelLoader:
    def load_transformer(
        self,
        path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer.model_configurator import (
            LTXV_MODEL_COMFY_RENAMING_MAP,
            LTXModelConfigurator,
        )

        logger.info(f"Loading LTX-2 transformer from {path}")
        model = SingleGPUModelBuilder(
            model_path=str(path),
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        ).build(
            device=torch.device(device) if isinstance(device, str) else device,
            dtype=dtype,
        )
        return model
