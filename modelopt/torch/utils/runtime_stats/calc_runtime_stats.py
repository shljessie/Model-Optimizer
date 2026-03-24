import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    SubblockConfig,
)
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import DeciLMForCausalLM


def create_benchmark_model(
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    block_config: BlockConfig | None,
    repeat_block_n_times: int = 10,
) -> DeciLMForCausalLM:

    block_configs = [
        BlockConfig(
            attention=AttentionConfig(no_op=False, num_key_value_heads=num_attention_heads),
            ffn=FFNConfig(no_op=False, intermediate_size=hidden_size, moe=None),
            parallel_blocks=None,
        )
    ]

    if block_config:
        block_configs.extend([block_config] * repeat_block_n_times)

    print("|||| Creating benchmark model...")
    model_config = DeciLMConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=len(block_configs),
        block_configs=block_configs,
        head_dim=None,  # Compute from hidden_size // num_attention_heads instead of using default 128
        # this is required for trt-llm convertion to know which model classes to use to the checkpoint
        auto_map={
            "AutoConfig": "configuration_decilm.DeciLMConfig",
            "AutoModelForCausalLM": "modeling_decilm.DeciLMForCausalLM",
        },
    )
    print(f"|||| Created DeciLM config with {len(block_configs)} layers")

    model = DeciLMForCausalLM(model_config)
    return model


def _save_model(model: DeciLMForCausalLM, tokenizer_path: Path, output_path: Path) -> None:

    model.to(dtype=torch.bfloat16).save_pretrained(output_path)
    AutoTokenizer.from_pretrained(tokenizer_path).save_pretrained(output_path)


def _generate_dataset(
    tokenizer_path: str,
    synth_dataset_num_requests: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    output_dir: str,
) -> None:

    args = [
        f"--tokenizer={tokenizer_path}",
        "--stdout",
        "token-norm-dist",
        f"--num-requests={synth_dataset_num_requests}",
        f"--input-mean={prefill_seq_len}",
        f"--output-mean={generation_seq_len}",
    ]
    output_filepath = output_dir / "dataset.txt"
    with open(output_filepath, "w") as f:
        subprocess.run(
            ["python", "/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"] + args,
            stdout=f,
            check=True,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    master_puzzle_dir: str
    tokenizer_path: str
    synth_dataset_num_requests: int
    backend: str
    repeat_block_n_times: int = 10
    prefill_seq_len: int = 100
    generation_seq_len: int = 100


def calc_subblock_runtime(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig,
) -> float:

    if not isinstance(subblock_config, AttentionConfig) and not isinstance(
        subblock_config, FFNConfig
    ):
        raise Exception("Runtime stats:Not supported subblock type")

    model = create_benchmark_model(
        runtime_config.vocab_size,
        runtime_config.hidden_size,
        runtime_config.num_attention_heads,
        block_config=subblock_config.to_blockconfig(),
        repeat_block_n_times=runtime_config.repeat_block_n_times,
    )
    model_tmpdir = Path(tempfile.mkdtemp())
    _save_model(model, Path(runtime_config.tokenizer_path), model_tmpdir)

    # synth_dataset_tmpdir = Path(tempfile.mkdtemp())
    # _generate_dataset(runtime_config.tokenizer_path,
    #                   runtime_config.synth_dataset_num_requests,
    #                   runtime_config.prefill_seq_len,
    #                   runtime_config.generation_seq_len,
    #                   synth_dataset_tmpdir)

    # import argparse

    # def run_vllm_latency_benchmark():

    #     from vllm.benchmarks.latency import add_cli_args
    #     from vllm.benchmarks.latency import main as vllm_main

    #     parser = argparse.ArgumentParser()
    #     add_cli_args(parser)
    #     args = parser.parse_args(
    #         [
    #             "--model",
    #             "meta-llama/Llama-3.1-8B-Instruct",
    #             "--input-len",
    #             "32",
    #             "--output-len",
    #             "128",
    #             "--batch-size",
    #             "8",
    #         ]
    #     )
    #     vllm_main(args)

    # print("|||| Running vllm latency benchmark...")
    # run_vllm_latency_benchmark()
    # print("|||| VLLM latency benchmark completed.")
    # # args = argparse.Namespace(
    # #     model=model_tmpdir,
    # #     tokenizer=runtime_config.tokenizer_path,
    # #     input_len=runtime_config.prefill_seq_len,
    # #     output_len=runtime_config.generation_seq_len,
    # #     batch_size=1,  # runtime_config.batch_size,
    # # )

    # # from vllm.benchmarks.latency import main as run_vllm_latency_benchmark

    # # run_vllm_latency_benchmark(args)

    subblock_total_runtime_ms = 1.0

    return subblock_total_runtime_ms


def calc_baseline_runtime(runtime_config: RuntimeConfig, subblock_config: SubblockConfig) -> float:
    return 0.1


def calc_runtime_for_subblocks(
    subblock_config_set: set[SubblockConfig],
    runtime_stats_config: DictConfig,
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    master_puzzle_dir: str,
    tokenizer_path: str,
    synth_dataset_num_requests: int,
    prefill_seq_len: int,
    generation_seq_len: int,
) -> tuple[dict[SubblockConfig, float], float]:

    repeat_block_n_times = 10
    runtime_config = RuntimeConfig(
        vocab_size,
        hidden_size,
        num_attention_heads,
        master_puzzle_dir,
        tokenizer_path,
        synth_dataset_num_requests,
        repeat_block_n_times,
        prefill_seq_len,
        generation_seq_len,
    )

    runtime_by_subblock_dict = {}

    for subblock_config in tqdm(
        sorted(subblock_config_set),
        desc=(
            f"Computing runtime_by_subblock_dict [hidden_size={hidden_size}, "
            f"num_subblocks={len(subblock_config_set)}]"
        ),
    ):
        if subblock_config.no_op:
            total_runtime_ms = 0.0
        else:
            subblock_total_runtime_ms = calc_subblock_runtime(runtime_config, subblock_config)
            baseline_runtime_ms = calc_baseline_runtime(runtime_config, subblock_config)
            total_runtime_ms = subblock_total_runtime_ms - baseline_runtime_ms

        runtime_by_subblock_dict[subblock_config] = total_runtime_ms

    no_block_runtime_ms = 0.0  # TODO: Implement this

    return runtime_by_subblock_dict, no_block_runtime_ms
