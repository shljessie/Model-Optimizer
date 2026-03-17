from collections import defaultdict

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import SubblockConfig


def calc_runtime_for_subblocks(
    subblock_config_set: set[SubblockConfig],
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    master_puzzle_dir: str,
    tokenizer_path: str,
    synth_dataset_num_requests: int,
    backend: str,
    prefill_seq_len: int,
    generation_seq_len: int,
) -> tuple[dict[SubblockConfig, float], float]:
    return defaultdict(float), 0.01
