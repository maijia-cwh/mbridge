import argparse
import asyncio  # ← 新增
import glob
import json
import os
import tempfile
from typing import Optional

import requests
from estimate_013 import estimate_from_config

from megatron.core import parallel_state as mpu
from pydantic import BaseModel, field_validator

from mbridge import AutoBridge

SUPPORTED_MODELS = [
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "moonshotai/Moonlight-16B-A3B",
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "deepseek-ai/DeepSeek-V3",
    "XiaomiMiMo/MiMo-7B-RL",
]



async def get_supported_models():
    """Return the list of HF model identifiers supported by the UI."""
    return SUPPORTED_MODELS



async def get_remote_hf_config(model_path: str):
    """Fetch the HuggingFace config.json for the given model id."""
    url = f"https://huggingface.co/{model_path}/raw/main/config.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Failed to fetch config from {url}: {str(e)}"}


class MBridgeEstimateConfig(BaseModel):
    hf_model_path: str
    custom_hf_config: Optional[dict] = None  # Renamed for clarity

    # Hardware & Training
    num_gpus: int = 8
    mbs: int = 1
    seq_len: int = 4096
    use_distributed_optimizer: bool = True
    # Recompute settings are now part of the main config
    recompute_granularity: str = "selective"
    recompute_method: str = "uniform"
    recompute_num_layers: Optional[int] = 1

    # Selective recompute modules (optional list only used when granularity==selective)
    recompute_modules: Optional[list[str]] = None

    # 新增：Embedding/Loss PP Split 选项
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # Parallelism
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    vpp: Optional[int] = None
    etp: Optional[int] = None

    # Pipeline stage layer counts
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None

    # New field: custom pipeline-model-parallel layout
    pipeline_model_parallel_layout: Optional[str] = None  # Comma-separated ints

    @field_validator("num_gpus")
    def num_gpus_must_be_multiple_of_8(cls, v):
        if v <= 0 or v % 8 != 0:
            raise ValueError("must be a positive multiple of 8")
        return v


def patch_parallel_states(config: MBridgeEstimateConfig):
    from mbridge.core.parallel_states import ParallelStates

    ParallelStates.get_default_parallel_states = lambda: ParallelStates(
        tp_size=config.tp,
        pp_size=config.pp,
        ep_size=config.ep,
        cp_size=config.cp,
        vpp_size=config.vpp,
        etp_size=config.etp,
    )
# ... [其他代码保持不变] ...

async def estimate_with_mbridge(config: MBridgeEstimateConfig):
    # 验证输入
    if config.num_gpus <= 0 or config.num_gpus % 8 != 0:
        return {"error": "Total number of GPUs must be a positive multiple of 8."}

    parallel_product = config.tp * config.pp * config.cp
    if parallel_product == 0:
        return {"error": "Parallelism dimensions (TP, PP, CP) cannot be zero."}

    if config.num_gpus % parallel_product != 0:
        return {
            "error": f"Number of GPUs ({config.num_gpus}) must be divisible by the product of TP*PP*CP ({parallel_product})."
        }

    patch_parallel_states(config)

    hf_model_path = config.hf_model_path


    # if not os.path.isabs(hf_model_path) and not hf_model_path.startswith(
    #         ("http", "./", "../")
    # ):
    #     hf_model_path = os.path.join("/dev/shm", hf_model_path)
    # 对于多模态模型（如 Kimi-K2.5），使用 text_config 子配置
    hf_config_path = os.path.join(hf_model_path, "config.json") if os.path.isdir(hf_model_path) else None
    if hf_config_path and os.path.exists(hf_config_path):
        with open(hf_config_path) as f:
            raw_config = json.load(f)
        if "text_config" in raw_config:
            # 多模态模型：提取 text_config 作为独立配置
            text_cfg = raw_config["text_config"]
            # 映射未注册的 model_type 到已注册的 bridge
            model_type_map = {"kimi_k2": "deepseek_v3"}
            if text_cfg.get("model_type") in model_type_map:
                text_cfg["model_type"] = model_type_map[text_cfg["model_type"]]
            # 移除 auto_map 避免加载自定义代码
            text_cfg.pop("auto_map", None)
            tmp_dir = tempfile.mkdtemp(prefix="mem_est_")
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(text_cfg, f)
            hf_model_path = tmp_dir

    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    tf_config = bridge.config
    hf_config = bridge.hf_config

    # --- 配置统一化 ---
    tf_config.tensor_model_parallel_size = config.tp
    tf_config.pipeline_model_parallel_size = config.pp
    tf_config.expert_model_parallel_size = config.ep
    tf_config.context_parallel_size = config.cp
    tf_config.recompute_granularity = config.recompute_granularity
    tf_config.recompute_method = config.recompute_method
    tf_config.recompute_num_layers = config.recompute_num_layers
    tf_config.recompute_modules = (
        config.recompute_modules if config.recompute_modules is not None else []
    )
    tf_config.account_for_embedding_in_pipeline_split = (
        config.account_for_embedding_in_pipeline_split
    )
    tf_config.account_for_loss_in_pipeline_split = (
        config.account_for_loss_in_pipeline_split
    )
    tf_config.num_layers_per_virtual_pipeline_stage = (
        config.vpp if config.vpp and config.vpp > 1 else None
    )

    if config.num_layers_in_first_pipeline_stage is not None:
        tf_config.num_layers_in_first_pipeline_stage = (
            config.num_layers_in_first_pipeline_stage
        )
    if config.num_layers_in_last_pipeline_stage is not None:
        tf_config.num_layers_in_last_pipeline_stage = (
            config.num_layers_in_last_pipeline_stage
        )

    if config.pipeline_model_parallel_layout:
        # 数值型 layout（如 "10,13,13,13,12"）：提取首尾 stage 层数
        layer_counts = [int(x.strip()) for x in config.pipeline_model_parallel_layout.split(",")]
        assert len(layer_counts) == config.pp, (
            f"pp-layout 长度 ({len(layer_counts)}) 必须等于 PP ({config.pp})"
        )
        tf_config.num_layers_in_first_pipeline_stage = layer_counts[0]
        tf_config.num_layers_in_last_pipeline_stage = layer_counts[-1]

    # 创建 args 对象
    args = argparse.Namespace()
    args.micro_batch_size = config.mbs
    args.seq_length = config.seq_len
    args.use_distributed_optimizer = config.use_distributed_optimizer
    args.data_parallel_size = config.num_gpus // parallel_product
    args.expert_tensor_parallel_size = config.etp if config.etp else 1
    args.transformer_impl = "transformer_engine"
    args.fp8 = False
    args.num_experts = getattr(tf_config, "num_moe_experts", 1)
    args.moe_grouped_gemm = True
    args.qk_layernorm = tf_config.qk_layernorm
    args.multi_latent_attention = "deepseek" in getattr(hf_config, "model_type", "")
    args.padded_vocab_size = getattr(hf_config, "vocab_size")
    args.max_position_embeddings = getattr(hf_config, "max_position_embeddings")
    args.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
    args.world_size = config.num_gpus

    aggregated_reports, raw_chunk_reports = estimate_from_config(tf_config, args)

    processed_reports = []
    for rpt in aggregated_reports:
        p = rpt.copy()
        p.pop("details", None)
        processed_reports.append(p)

    print("processed_report: ", processed_reports)
    print("raw_report:", raw_chunk_reports)

    return {"processed_reports": processed_reports, "raw_chunk_reports": raw_chunk_reports}


if __name__ == "__main__":
    # ✅ 修复：创建配置实例，使用 asyncio 运行异步函数
    config = MBridgeEstimateConfig(
        hf_model_path="/Users/maijia/code/opensource/mbridge/memory_estimator/config/Qwen3-8B",
        num_gpus=8,
        tp=1,
        pp=1
    )
    result = asyncio.run(estimate_with_mbridge(config))
    print(json.dumps(result, indent=2))
