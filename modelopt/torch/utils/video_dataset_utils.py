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

"""Utility functions for getting samples and forward loop function for video datasets."""

import os
import tempfile
from typing import Any

import torch
from torch.utils.data import DataLoader

from .image_processor import BaseImageProcessor, _Qwen3OmniProcessorMixin

# Use dict to store the config for each dataset.
SUPPORTED_VIDEO_DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "finevideo": {
        "config": {"path": "HuggingFaceFV/finevideo", "split": "train", "streaming": True}
    },
}

__all__ = [
    "Qwen3OmniVideoProcessor",
    "get_supported_video_datasets",
    "get_video_dataset_dataloader",
]


def _get_video_dataset(dataset_name: str, num_samples: int):
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        A hugging face Dataset.
    """
    if dataset_name in SUPPORTED_VIDEO_DATASET_CONFIG:
        from datasets import Dataset, load_dataset

        config = SUPPORTED_VIDEO_DATASET_CONFIG[dataset_name]["config"]
        is_streaming = config.get("streaming", False)

        dataset = load_dataset(**config)

        if is_streaming:
            # For streaming datasets, use take() and convert to list then Dataset
            samples = list(dataset.take(num_samples))
            return Dataset.from_list(samples)
        else:
            return dataset.select(range(num_samples))
    else:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_video_datasets()}."
        )


def get_supported_video_datasets() -> list[str]:
    """Retrieves a list of video datasets supported.

    Returns:
        A list of strings, where each string is the name of a supported dataset.

    Example usage:

    .. code-block:: python

        from modelopt.torch.utils import get_supported_video_datasets

        print("Supported video datasets:", get_supported_video_datasets())
    """
    return list(SUPPORTED_VIDEO_DATASET_CONFIG.keys())


def get_video_dataset_dataloader(
    dataset_name: str = "finevideo",
    processor: "Qwen3OmniVideoProcessor" = None,
    batch_size: int = 1,
    num_samples: int = 512,
    cache_dir: str | None = None,
) -> DataLoader:
    """Get a dataloader with the dataset name and processor of the target model.

    Args:
        dataset_name: Name of the dataset to load.
        processor: Processor used for encoding video and text data.
        batch_size: Batch size of the returned dataloader.
        num_samples: Number of samples from the dataset.
        cache_dir: Directory to cache the processed dataset. Defaults to a temp directory.
            If the cache exists, it will be loaded instead of reprocessing.

    Returns:
        An instance of dataloader.
    """
    assert processor is not None, "Please provide a valid processor."

    # Default cache_dir to temp directory
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "modelopt_video_dataset_cache")

    processed_dataset = None

    # Try to load from cache (use torch.save/load to avoid Arrow 32-bit offset overflow)
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, f"{dataset_name}_n{num_samples}_processed.pt")
        if os.path.exists(cache_path):
            try:
                from datasets import Dataset

                # weights_only=False is safe here: the cache file is self-generated at line 151
                processed_samples = torch.load(cache_path, weights_only=False)
                processed_dataset = Dataset.from_list(processed_samples)
                print(f"Loaded processed dataset from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")
                processed_dataset = None

    # Process dataset if not loaded from cache
    if processed_dataset is None:
        from datasets import Dataset

        dataset = _get_video_dataset(dataset_name, num_samples=num_samples)

        # Process samples manually to avoid Arrow 32-bit offset overflow
        # (dataset.map() uses Arrow internally which can't handle large nested lists)
        processed_samples = []
        for i, sample in enumerate(dataset):
            processed = processor.preprocess_function(sample)
            processed_samples.append(processed)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples...")

        processed_dataset = Dataset.from_list(processed_samples)

        # Save to cache using torch.save to avoid Arrow 32-bit offset overflow
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(processed_samples, cache_path)
            print(f"Saved processed dataset to cache: {cache_path}")

    # Create DataLoader with the custom collate function
    return DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=processor.collate_function,
    )


class Qwen3OmniVideoProcessor(_Qwen3OmniProcessorMixin, BaseImageProcessor):
    """Video processor for Qwen3-Omni multimodal model with finevideo dataset support."""

    def __init__(self, tokenizer, device="cuda", dtype=None, use_audio_in_video=True):
        """Constructor.

        Args:
            tokenizer: The Qwen3OmniMoeProcessor for tokenizing and processing inputs.
            device: Device to move tensors to.
            dtype: dtype for float tensors (e.g., torch.bfloat16). If None, uses default.
            use_audio_in_video: Whether to extract and use audio from video files.
        """
        super().__init__(tokenizer, device)
        self.dtype = dtype
        self.use_audio_in_video = use_audio_in_video
        self._temp_dir = tempfile.mkdtemp(prefix="qwen3omni_video_")
        self._video_counter = 0
        # Try to import qwen_omni_utils for multimodal processing
        try:
            from qwen_omni_utils import process_mm_info

            self.process_mm_info = process_mm_info
        except ImportError:
            raise ImportError(
                "qwen_omni_utils is required for Qwen3OmniVideoProcessor. "
                "Please install it from https://github.com/QwenLM/Qwen3-Omni"
            )

    def _save_video_bytes_to_file(self, video_bytes: bytes) -> str:
        """Save video bytes to a temporary file and return the path.

        Args:
            video_bytes: Raw video bytes (e.g., from finevideo's 'mp4' field).

        Returns:
            Path to the temporary video file.
        """
        video_path = os.path.join(self._temp_dir, f"video_{self._video_counter}.mp4")
        self._video_counter += 1
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        return video_path

    _ALL_KEYS = [
        "input_ids",
        "attention_mask",
        "pixel_values_videos",
        "video_grid_thw",
        "video_second_per_grid",
        "feature_attention_mask",
        "input_features",
    ]

    def preprocess_function(self, examples):
        """Preprocess function for Qwen3-Omni with video support.

        Handles both standard video paths and raw video bytes (finevideo format).
        """
        # Get question/prompt - finevideo has metadata in 'json' field
        if "json" in examples and examples["json"] is not None:
            metadata = examples["json"]
            category = metadata.get("content_fine_category", "")
            question = (
                f"Describe what is happening in this video in detail. Category hint: {category}"
            )
        else:
            question = examples.get("question", "Describe this video in detail.")

        # Build conversation in Qwen format
        content = []

        # Handle video - check for raw bytes (finevideo format) or path
        video_path = None
        if examples.get("mp4") is not None:
            video_path = self._save_video_bytes_to_file(examples["mp4"])
        elif examples.get("video") is not None:
            video_path = examples["video"]

        if video_path is not None:
            content.append({"type": "video", "video": video_path})

        content.append({"type": "text", "text": question})

        conversation = [{"role": "user", "content": content}]
        values = self._tokenize_conversation(conversation)
        return self._serialize_for_arrow(values, self._ALL_KEYS)

    def collate_function(self, batch):
        """Collate function to process inputs during data loading."""
        result = self._collate_first_item(
            batch,
            long_keys=(
                "input_ids",
                "attention_mask",
                "video_grid_thw",
                "feature_attention_mask",
            ),
            float_keys=("pixel_values_videos", "video_second_per_grid", "input_features"),
            dtype=self.dtype,
        )
        # Pass use_audio_in_video flag to model.generate() for Qwen3Omni
        result["use_audio_in_video"] = self.use_audio_in_video
        return result

    def cleanup(self):
        """Clean up temporary video files."""
        import shutil

        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    def __del__(self):
        """Ensure temporary files are cleaned up when the processor is garbage collected."""
        self.cleanup()
