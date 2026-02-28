"""Wan2.2 inference pipeline for validation video generation and data preprocessing.

Wraps the official Wan2.2 T5 encoder, VAE, and denoising loop to work with
the unified trainer's cached-embeddings protocol.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from torch import Tensor

from ...interfaces import CachedEmbeddings, free_gpu_memory

logger = logging.getLogger(__name__)


class WanInferencePipeline:
    """Inference pipeline for Wan2.2 models.

    Manages T5 text encoder and VAE lifecycles:
    - T5 is loaded, used for embedding, then permanently deleted.
    - VAE stays on CPU during training, briefly moves to GPU for decode/encode.
    """

    def __init__(self) -> None:
        self._text_encoder = None
        self._vae = None
        self._config = None

    def load_components(
        self, model_config, device: str, dtype: torch.dtype
    ) -> None:
        from wan.configs.wan_ti2v_5B import ti2v_5B
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_2 import Wan2_2_VAE

        # Accept either a ModelConfig object or a plain path string
        path = str(getattr(model_config, "model_path", model_config))
        self._config = ti2v_5B

        t5_path = os.path.join(path, self._config.t5_checkpoint)
        t5_tokenizer = self._config.t5_tokenizer
        self._text_encoder = T5EncoderModel(
            text_len=self._config.text_len,
            dtype=dtype,
            device=torch.device("cpu"),
            checkpoint_path=t5_path,
            tokenizer_path=t5_tokenizer,
        )

        vae_path = os.path.join(path, self._config.vae_checkpoint)
        self._vae = Wan2_2_VAE(vae_pth=vae_path, device=device)

        logger.info("Wan inference components loaded (T5 + VAE)")

    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompt: str,
        device: str,
    ) -> list[CachedEmbeddings]:
        assert self._text_encoder is not None, "Call load_components() first"

        self._text_encoder.model.to(device)
        cached = []
        with torch.no_grad():
            for prompt in prompts:
                ctx_pos = self._text_encoder([prompt], torch.device(device))
                ctx_neg = self._text_encoder([negative_prompt], torch.device(device))
                cached.append(CachedEmbeddings(
                    positive={"context": ctx_pos[0].cpu()},
                    negative={"context": ctx_neg[0].cpu()},
                ))
        self._text_encoder.model.cpu()
        return cached

    def unload_text_encoder(self) -> None:
        if self._text_encoder is not None:
            del self._text_encoder
            self._text_encoder = None
        free_gpu_memory()
        logger.info("T5 text encoder unloaded")

    def offload_to_cpu(self) -> None:
        if self._vae is not None:
            self._vae.model.cpu()
        free_gpu_memory()

    def encode_videos(
        self,
        videos: list[Tensor],
        device: str,
    ) -> list[Tensor]:
        assert self._vae is not None, "Call load_components() first"
        self._vae.model.to(device)
        with torch.no_grad():
            latents = self._vae.encode([v.to(device) for v in videos])
        self._vae.model.cpu()
        free_gpu_memory()
        return [z.cpu() for z in latents]

    def generate(
        self,
        model: nn.Module,
        cached_embeds: list[CachedEmbeddings],
        config: dict,
        device: str,
    ) -> list[Tensor]:
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        assert self._vae is not None
        assert self._config is not None

        width = config.get("width", 512)
        height = config.get("height", 320)
        num_frames = config.get("num_frames", 33)
        num_steps = config.get("num_inference_steps", 30)
        guidance_scale = config.get("guidance_scale", 5.0)
        seed = config.get("seed", 42)
        shift = self._config.sample_shift

        vae_stride = self._config.vae_stride
        n_f = (num_frames - 1) // vae_stride[0] + 1
        n_h = height // vae_stride[1]
        n_w = width // vae_stride[2]
        z_dim = self._vae.model.z_dim

        patch_size = self._config.patch_size
        seq_len = n_f * (n_h // patch_size[1]) * (n_w // patch_size[2])

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(num_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps

        videos = []
        generator = torch.Generator(device=device).manual_seed(seed)

        for emb in cached_embeds:
            context = [emb.positive["context"].to(device)]
            context_null = [emb.negative["context"].to(device)]

            latent = torch.randn(
                z_dim, n_f, n_h, n_w, dtype=torch.float32, device=device, generator=generator
            )
            latents = [latent]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                for t in timesteps:
                    timestep = torch.stack([t])
                    # Per-token timestep expansion (simplified from official code)
                    timestep_expanded = timestep.expand(1, seq_len)

                    noise_pred_cond = model(latents, t=timestep_expanded, context=context, seq_len=seq_len)[0]
                    noise_pred_uncond = model(latents, t=timestep_expanded, context=context_null, seq_len=seq_len)[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                        return_dict=False, generator=generator
                    )[0]
                    latent = temp_x0.squeeze(0)
                    latents = [latent]

            # VAE decode
            self._vae.model.to(device)
            with torch.no_grad():
                decoded = self._vae.decode([latent])
            self._vae.model.cpu()

            # decoded[0] is [C, F, H, W] in [-1, 1] -> [0, 1]
            video = ((decoded[0] + 1.0) / 2.0).clamp(0, 1).float().cpu()
            videos.append(video)

        free_gpu_memory()
        return videos
