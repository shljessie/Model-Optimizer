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

"""Graph surgery module for post-processing ONNX models.

This module provides utilities for performing graph-level transformations on ONNX models
after export. Common use cases include:

- Replacing standard attention patterns with GroupQueryAttention (GQA) for LLMs
- Adding cross-attention KV cache outputs to encoder models
- Converting model precision (e.g., FP16 to BF16)
- Transposing DequantizeLinear weights for column-major storage optimization
- Graph cleanup and optimization

CLI Usage::

    python -m modelopt.onnx.graph_surgery <command> [options]

Available commands:

Replace attention with GQA (for FP16/BF16 LLMs)::

    python -m modelopt.onnx.graph_surgery replace-gqa \
        --input model.onnx \
        --output model_gqa.onnx \
        --model-id meta-llama/Llama-2-7b-hf

Replace attention with GQA (for INT4/AWQ quantized LLMs)::

    python -m modelopt.onnx.graph_surgery replace-gqa \
        --input model.onnx \
        --output model_gqa.onnx \
        --model-id meta-llama/Llama-3.1-8B

Add cross-attention KV cache to encoder::

    python -m modelopt.onnx.graph_surgery add-cross-kv \
        --input encoder_model.onnx \
        --output encoder_with_kv.onnx \
        --model-id openai/whisper-large-v3-turbo

Convert FP16 to BF16::

    python -m modelopt.onnx.graph_surgery convert-bf16 \
        --input model_fp16.onnx \
        --output model_bf16.onnx

Transpose DequantizeLinear weights (column-major optimization)::

    python -m modelopt.onnx.graph_surgery transpose-dq \
        --input model_quantized.onnx \
        --output model_quantized_transposed.onnx

Analyze attention pattern::

    python -m modelopt.onnx.graph_surgery analyze \
        --input model.onnx \
        --layer 0

For full options on any command, run::

    python -m modelopt.onnx.graph_surgery <command> --help
"""

from .dq_transpose import transpose_dequantize_linear_weights
from .encoder_cross_kv import add_cross_kv_to_encoder
from .gqa_replacement import replace_attention_with_gqa
from .utils.dtype_conversion import convert_fp16_to_bf16

__all__ = [
    "add_cross_kv_to_encoder",
    "convert_fp16_to_bf16",
    "replace_attention_with_gqa",
    "transpose_dequantize_linear_weights",
]
