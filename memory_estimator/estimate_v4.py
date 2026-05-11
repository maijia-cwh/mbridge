# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""DeepSeek-V4 Memory Estimation Entry Point.

Usage:
    from estimate_v4 import estimate_v4_from_config
    result = estimate_v4_from_config(v4_config)
"""

import argparse
import json
import os
from typing import Optional

from moe_mem_estimator.base import (
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_global_config,
    set_pipeline_model_parallel_rank,
)
from moe_mem_estimator.dsv4 import DSV4Config, DSV4Model

NUM_BYTES_IN_GIGABYTE = 1024 * 1024 * 1024


def patch_parallel_states_v4(config: DSV4Config):
    """Set up parallel state functions for V4 estimation."""
    from mbridge.core.parallel_states import ParallelStates

    ParallelStates.get_default_parallel_states = lambda: ParallelStates(
        tp_size=config.tp_size,
        pp_size=config.pp_size,
        ep_size=config.ep_size,
        cp_size=config.cp_size,
        vpp_size=None,
        etp_size=config.etp_size,
    )


def build_v4_model(config: DSV4Config, pre_process: bool = True, post_process: bool = True) -> DSV4Model:
    """Build the V4 model estimator."""
    return DSV4Model(config, pre_process=pre_process, post_process=post_process)


def report_memory_usage_one_pp_rank(
    input_shape: list[int],
    config: DSV4Config,
    pp_rank: int = 0,
    pp_size: int = 1,
) -> tuple[list[int], dict]:
    """Report memory usage for one PP rank."""
    set_pipeline_model_parallel_rank(pp_rank)

    pre_process = is_pipeline_first_stage()
    post_process = is_pipeline_last_stage()

    model = build_v4_model(config, pre_process=pre_process, post_process=post_process)

    # Parameter counting
    num_parameter_this_shard = model.num_parameter()
    num_parameter_this_shard_sparse = model.num_parameter_sparse()

    # Activation counting
    num_activation = model.num_activation(input_shape)
    output_shape = model.mock_forward(input_shape)

    # MoE activation breakdown
    num_activation_mlp = 0
    for block in model.blocks:
        num_activation_mlp += block.moe.num_activation(input_shape)

    # Number of in-flight microbatches (pipeline parallelism)
    if pp_size == 1:
        num_microbatch = 1
    else:
        num_microbatch = pp_size - pp_rank

    # Recompute handling
    if config.recompute_granularity == "full":
        recompute_num_layers = config.recompute_num_layers
        num_layers = model.num_layers

        common_act = (
            model._num_act_pre
            + model._num_act_between_layers * num_layers * num_microbatch
        )

        if config.recompute_method == "block":
            num_layers_with_loss = num_layers - recompute_num_layers
            if num_layers_with_loss == 0:
                peak1 = common_act + model._num_act_post
                peak2 = common_act + model._num_act_per_layer
                num_activation = max(peak1, peak2)
            else:
                num_activation = (
                    common_act
                    + model._num_act_post
                    + model._num_act_per_layer
                    * num_layers_with_loss
                    * num_microbatch
                )
        elif config.recompute_method == "uniform":
            peak1 = common_act + model._num_act_post
            peak2 = common_act + model._num_act_per_layer
            num_activation = max(peak1, peak2)
    else:
        # Selective or no recompute
        num_activation = (
            (num_activation - model._num_act_post)
            * num_microbatch
            + model._num_act_post
        )

    # Context parallelism
    num_activation = num_activation / config.cp_size

    # Bytes per parameter
    world_size = config.world_size
    dp_size = config.data_parallel_size

    if config.use_distributed_optimizer:
        num_bytes_per_parameter_dense = 6 + 12 / dp_size
    else:
        num_bytes_per_parameter_dense = 18

    ep_size = config.ep_size
    etp_size = config.etp_size
    if ep_size * etp_size > 1:
        moe_dp = world_size // (config.pp_size * ep_size * etp_size)
        num_bytes_per_parameter_moe = 6 + 12 / max(moe_dp, 1)
    else:
        num_bytes_per_parameter_moe = num_bytes_per_parameter_dense

    # Memory calculation
    weight_grad_memory = num_parameter_this_shard * 6 / NUM_BYTES_IN_GIGABYTE
    weight_grad_optim_memory = (
        (num_parameter_this_shard - num_parameter_this_shard_sparse)
        * num_bytes_per_parameter_dense
        + num_parameter_this_shard_sparse * num_bytes_per_parameter_moe
    ) / NUM_BYTES_IN_GIGABYTE
    activation_memory = num_activation * 2 / NUM_BYTES_IN_GIGABYTE  # FP16
    total_memory = weight_grad_optim_memory + activation_memory

    # Print report
    print(f"\n--- PP Rank {pp_rank} ---")
    print(f"  Layers: {model.num_layers} (offset: {model.layer_offset})")
    print(f"  Parameters: {num_parameter_this_shard / 1e9:.2f}B"
          f" (sparse/MoE: {num_parameter_this_shard_sparse / 1e9:.2f}B)")
    print(f"  Activation: {num_activation / 1e9:.2f}B elements"
          f" (MoE: {num_activation_mlp / 1e9:.2f}B)")
    print(f"  Weight+Grad: {weight_grad_memory:.2f} GB")
    print(f"  Weight+Grad+Optim: {weight_grad_optim_memory:.2f} GB")
    print(f"  Activation: {activation_memory:.2f} GB")
    print(f"  Total: {total_memory:.2f} GB")

    # Attention type breakdown
    csa_count = sum(1 for b in model.blocks if b.attention.is_csa)
    hca_count = sum(1 for b in model.blocks if b.attention.is_hca)
    win_count = sum(1 for b in model.blocks if b.attention.is_window)
    print(f"  Attention types: Window={win_count}, CSA={csa_count}, HCA={hca_count}")

    # Per-layer parameter breakdown
    if len(model.blocks) > 0:
        block = model.blocks[0]
        attn_params = block.attention.num_parameter()
        moe_params = block.moe.num_parameter()
        hc_params = block.hc_attn.num_parameter() + block.hc_ffn.num_parameter()
        norm_params = block.attn_norm.num_parameter() + block.ffn_norm.num_parameter()
        print(f"  Per-layer breakdown (layer 0):")
        print(f"    Attention: {attn_params / 1e6:.2f}M")
        print(f"    MoE: {moe_params / 1e6:.2f}M"
              f" (routed: {block.moe.routed_experts.num_parameter() * block.moe.num_local_experts / 1e6:.2f}M"
              f" shared: {block.moe.shared_expert.num_parameter() / 1e6:.2f}M"
              f" gate: {block.moe.gate.num_parameter() / 1e6:.2f}M)")
        print(f"    HyperConnection: {hc_params / 1e6:.2f}M")
        print(f"    Norm: {norm_params / 1e6:.2f}M")

    report = {
        "pp_rank": pp_rank,
        "num_layers": model.num_layers,
        "parameters_b": num_parameter_this_shard / 1e9,
        "parameters_sparse_b": num_parameter_this_shard_sparse / 1e9,
        "activation_b": num_activation / 1e9,
        "weight_grad_gb": round(weight_grad_memory, 2),
        "weight_grad_optim_gb": round(weight_grad_optim_memory, 2),
        "activation_gb": round(activation_memory, 2),
        "total_gb": round(total_memory, 2),
        "attention_types": {"window": win_count, "csa": csa_count, "hca": hca_count},
    }

    return output_shape, report


def estimate_v4_from_config(config: DSV4Config):
    """Estimate memory usage for DeepSeek-V4 model.

    Args:
        config: DSV4Config with all model and parallelism parameters

    Returns:
        Tuple of (aggregated_reports, raw_reports)
    """
    # Set up parallel states
    patch_parallel_states_v4(config)

    input_shape = [config.micro_batch_size, config.seq_length]

    # Set global config for base.py functions
    # Create a minimal namespace that maps V4 config fields to what base.py expects
    import types

    _cfg = types.SimpleNamespace(
        tensor_model_parallel_size=config.tp_size,
        expert_tensor_parallel_size=config.etp_size,
        pipeline_model_parallel_size=config.pp_size,
        expert_model_parallel_size=config.ep_size,
        virtual_pipeline_model_parallel_size=None,
    )
    set_global_config(_cfg)

    pp_size = config.pp_size
    cli_reports = []

    if pp_size > 1:
        current_input_shape = input_shape[:]
        for pp_rank in range(pp_size):
            print(f"\n==========[Pipeline_Parallelism_Rank={pp_rank}]==========")
            output_shape, rpt = report_memory_usage_one_pp_rank(
                current_input_shape, config, pp_rank, pp_size
            )
            current_input_shape = output_shape
            cli_reports.append(rpt)
    else:
        set_pipeline_model_parallel_rank(0)
        _, rpt = report_memory_usage_one_pp_rank(input_shape, config)
        cli_reports.append(rpt)

    # Summary
    print("\n===== Summary (per PP rank) =====")
    for r in cli_reports:
        print(
            f"PP{r['pp_rank']}  total {r['total_gb']} GB  "
            f"(weight_grad {r['weight_grad_gb']} GB  "
            f"weight_grad_optim {r['weight_grad_optim_gb']} GB  "
            f"act {r['activation_gb']} GB)  "
            f"attn_types: {r['attention_types']}"
        )

    return cli_reports, cli_reports


def estimate_v4_from_hf_config(
    hf_config_path: str,
    num_gpus: int = 8,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    cp: int = 1,
    etp: int = 1,
    mbs: int = 1,
    seq_len: int = 4096,
    use_distributed_optimizer: bool = True,
    recompute_granularity: str = "selective",
    recompute_method: str = "uniform",
    recompute_num_layers: int = 1,
    recompute_modules: Optional[list] = None,
    num_layers_in_first_pipeline_stage: Optional[int] = None,
    num_layers_in_last_pipeline_stage: Optional[int] = None,
    no_1f1b: bool = False,
    unfused_attn: bool = False,
):
    """Estimate V4 memory from a HuggingFace config.json path.

    Args:
        hf_config_path: Path to directory containing config.json, or path to config.json itself
        ... other args: parallelism and training configuration

    Returns:
        Tuple of (aggregated_reports, raw_reports)
    """
    # Load config
    if os.path.isdir(hf_config_path):
        config_path = os.path.join(hf_config_path, "config.json")
    else:
        config_path = hf_config_path

    with open(config_path) as f:
        hf_config = json.load(f)

    # Compute parallelism
    parallel_product = tp * pp * cp
    if parallel_product == 0:
        raise ValueError("TP, PP, CP cannot be zero")

    if num_gpus % parallel_product != 0:
        raise ValueError(
            f"Number of GPUs ({num_gpus}) must be divisible by TP*PP*CP ({parallel_product})"
        )

    dp_size = num_gpus // parallel_product

    # Create V4 config
    v4_config = DSV4Config.from_hf_config(
        hf_config,
        tp_size=tp,
        pp_size=pp,
        ep_size=ep,
        cp_size=cp,
        etp_size=etp,
        micro_batch_size=mbs,
        seq_length=seq_len,
        use_distributed_optimizer=use_distributed_optimizer,
        data_parallel_size=dp_size,
        world_size=num_gpus,
        recompute_granularity=recompute_granularity,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        recompute_modules=recompute_modules or [],
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        unfused_attn=unfused_attn,
    )

    return estimate_v4_from_config(v4_config)


if __name__ == "__main__":
    import asyncio

    # Default: estimate with the provided V4 config
    config_path = os.path.join(
        os.path.dirname(__file__), "config", "DeepSeek-V4", "config.json"
    )
    if not os.path.exists(config_path):
        # Try the deepseekv4 directory
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "deepseekv4", "config.json"
        )

    result = estimate_v4_from_hf_config(
        config_path,
        num_gpus=512,
        tp=8,
        pp=1,
        ep=8,
        cp=8,
        mbs=1,
        seq_len=131072,
    )
    print(json.dumps(result[0], indent=2))