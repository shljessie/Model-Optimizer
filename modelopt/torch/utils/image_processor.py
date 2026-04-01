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

# Adapted from tensorrt_llm/quantization/image_processing.py
"""Utility classes for image processing."""

from typing import Any

import torch


class BaseImageProcessor:
    """Base class for image processors."""

    def __init__(self, tokenizer, device="cuda"):
        """Constructor."""
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, **kwargs):
        """Call the tokenizer."""
        return self.tokenizer(**kwargs)

    def preprocess_function(self, examples):
        """Preprocess function."""
        raise NotImplementedError("Each image processor must implement its own preprocess method")

    def collate_function(self, examples):
        """Collate function to process images during data loading."""
        raise NotImplementedError("Each image processor must implement its own collate method")

    def _collate_first_item(self, batch, long_keys=(), float_keys=(), dtype=None):
        """Shared collate helper: validates batch_size=1, converts lists to tensors.

        Args:
            batch: List of sample dicts from the DataLoader.
            long_keys: Keys to convert via torch.LongTensor.
            float_keys: Keys to convert via torch.tensor with optional dtype cast.
            dtype: Optional dtype for float_keys tensors.

        Returns:
            Dict of tensors moved to self.device.
        """
        if len(batch) != 1:
            raise ValueError(f"{type(self).__name__} currently supports batch_size=1 only.")
        first = batch[0]
        result = {}
        for key in long_keys:
            if first.get(key) is not None:
                result[key] = torch.LongTensor(first[key]).to(self.device)
        for key in float_keys:
            if first.get(key) is not None:
                t = torch.tensor(first[key])
                if dtype is not None:
                    t = t.to(dtype)
                result[key] = t.to(self.device)
        return result


# A light Encapsulation for Huggingface MllamaImageProcessor


class MllamaImageProcessor(BaseImageProcessor):
    """Image processor for Mllama."""

    def preprocess_function(self, examples):
        """Preprocess function."""
        # Prepare prompts in a generic chat format
        question = examples.get("question", "Describe this image.")

        if examples["image"] is not None:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": question}],
                        }
                    ],
                    add_generation_prompt=True,
                )
            else:
                prompt = f"<|image|><|begin_of_text|>{question}"

            # Process images using the processor's image processor
            values = self.tokenizer(text=prompt, images=examples["image"], return_tensors="pt").to(
                self.device
            )
        else:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": question}],
                        }
                    ],
                    add_generation_prompt=True,
                )
            else:
                prompt = question

            values = self.tokenizer(text=prompt, images=None, return_tensors="pt").to(self.device)

            values["pixel_values"] = None
            values["aspect_ratio_ids"] = None
            values["aspect_ratio_mask"] = None
            values["cross_attention_mask"] = None

        return values

    def collate_function(self, batch):
        """Collate function to process images during data loading."""
        batch[0]["input_ids"] = torch.LongTensor(batch[0]["input_ids"]).to(self.device)
        batch[0]["attention_mask"] = torch.LongTensor(batch[0]["attention_mask"]).to(self.device)

        if batch[0]["pixel_values"] is not None:
            batch[0]["pixel_values"] = torch.Tensor(batch[0]["pixel_values"]).to(self.device)
            batch[0]["aspect_ratio_ids"] = torch.LongTensor(batch[0]["aspect_ratio_ids"]).to(
                self.device
            )
            batch[0]["aspect_ratio_mask"] = torch.LongTensor(batch[0]["aspect_ratio_mask"]).to(
                self.device
            )
            batch[0]["cross_attention_mask"] = torch.LongTensor(
                batch[0]["cross_attention_mask"]
            ).to(self.device)

        return batch[0]


class Qwen3OmniTextProcessor(BaseImageProcessor):
    """Text-only processor for Qwen3-Omni that applies proper conversation template.

    This processor wraps raw text in the Qwen3-Omni conversation format and applies
    the chat template before tokenization. Use this for text-only calibration datasets.

    See: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking
    """

    def __init__(self, processor, device="auto", dtype=None):
        """Constructor.

        Args:
            processor: The Qwen3OmniMoeProcessor (from AutoProcessor.from_pretrained).
            device: Device to move tensors to.
            dtype: dtype for float tensors (e.g., torch.bfloat16). If None, uses default.
        """
        super().__init__(processor, device)
        self.dtype = dtype

    def preprocess_function(self, text: str) -> dict:
        """Preprocess a single text sample by applying conversation template.

        Args:
            text: Raw text string from dataset.

        Returns:
            Dictionary with tokenized inputs.
        """
        # Build conversation in Qwen format (text-only)
        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        formatted_text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        # Tokenize with the processor (no multimodal inputs)
        values = self.tokenizer(
            text=formatted_text,
            audio=None,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
        )

        return values

    def collate_function(self, batch):
        """Collate function to process text inputs during data loading."""
        return self._collate_first_item(
            batch,
            long_keys=("input_ids", "attention_mask"),
        )


class _Qwen3OmniProcessorMixin:
    """Shared preprocessing logic for Qwen3-Omni image/video processors."""

    tokenizer: Any
    process_mm_info: Any
    use_audio_in_video: Any

    def _tokenize_conversation(self, conversation):
        """Tokenize a Qwen3-Omni conversation and return processor outputs.

        Args:
            conversation: List of conversation dicts in Qwen format.

        Returns:
            Processor output dict with tensors.
        """
        text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        audios, images, videos = self.process_mm_info(
            conversation, use_audio_in_video=self.use_audio_in_video
        )
        return self.tokenizer(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

    @staticmethod
    def _serialize_for_arrow(values, all_keys):
        """Convert processor outputs to lists for Arrow serialization.

        Args:
            values: Processor output dict (may contain tensors).
            all_keys: List of keys to include in the result (ensures consistent schema).

        Returns:
            Dict with all_keys initialized to None, populated from values.
        """
        result = dict.fromkeys(all_keys)
        for key, val in values.items():
            if val is not None and hasattr(val, "tolist"):
                result[key] = val.tolist()
            elif val is not None:
                result[key] = val
        return result


class Qwen3OmniImageProcessor(_Qwen3OmniProcessorMixin, BaseImageProcessor):
    """Image processor for Qwen3-Omni multimodal model."""

    _ALL_KEYS = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "audio_features",
        "audio_feature_lens",
        "video_grid_thw",
    ]

    def __init__(self, tokenizer, device="auto", dtype=None, use_audio_in_video=False):
        """Constructor."""
        super().__init__(tokenizer, device)
        self.dtype = dtype
        self.use_audio_in_video = use_audio_in_video
        # Try to import qwen_omni_utils for multimodal processing
        try:
            from qwen_omni_utils import process_mm_info

            self.process_mm_info = process_mm_info
        except ImportError:
            raise ImportError(
                "qwen_omni_utils is required for Qwen3OmniImageProcessor. "
                "Please install it from https://github.com/QwenLM/Qwen3-Omni"
            )

    def preprocess_function(self, examples):
        """Preprocess function for Qwen3-Omni."""
        question = examples.get("question", "Describe this image.")

        # Build conversation in Qwen format
        content = []
        if examples.get("image") is not None:
            content.append({"type": "image", "image": examples["image"]})
        if examples.get("audio") is not None:
            content.append({"type": "audio", "audio": examples["audio"]})
        if examples.get("video") is not None:
            content.append({"type": "video", "video": examples["video"]})
        content.append({"type": "text", "text": question})

        conversation = [{"role": "user", "content": content}]
        values = self._tokenize_conversation(conversation)
        return self._serialize_for_arrow(values, self._ALL_KEYS)

    def collate_function(self, batch):
        """Collate function to process inputs during data loading."""
        return self._collate_first_item(
            batch,
            long_keys=(
                "input_ids",
                "attention_mask",
                "image_grid_thw",
                "audio_feature_lens",
                "video_grid_thw",
            ),
            float_keys=("pixel_values", "audio_features"),
            dtype=self.dtype,
        )
