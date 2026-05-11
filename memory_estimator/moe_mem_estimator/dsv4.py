# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""DeepSeek-V4 Memory Estimator

Supports three attention types based on compress_ratio:
- Window-only (ratio=0): Sliding window attention, no compression
- CSA (ratio=4): Compressed Sparse Attention with overlapping compression + indexer
- HCA (ratio>=128): Heavy Compressed Attention, no overlap, no indexer

Also supports:
- Hyper-Connection (HC) residual mixing
- MoE with hash routing (first n_hash_layers) and score-based routing
- Grouped output projection (wo_a + wo_b low-rank)
- Per-head RMSNorm on Q, RMSNorm on compressed KV
- Attention sink
- Multi-Token Prediction (MTP) blocks
"""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import (
    MemEstimator,
    _addindent,
    colored,
    cum_mul,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_expert_tensor_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_global_config,
)
from megatron.core.utils import divide


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DSV4Config:
    """Configuration for DeepSeek-V4 model estimation."""
    # Model dimensions
    hidden_size: int = 7168
    num_hidden_layers: int = 43
    num_attention_heads: int = 128
    head_dim: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    o_groups: int = 16
    o_lora_rank: int = 1024
    window_size: int = 128
    vocab_size: int = 129280
    rms_norm_eps: float = 1e-6

    # Compression (per-layer)
    compress_ratios: tuple = ()

    # Indexer (CSA only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512

    # Hyper-Connection
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    num_hash_layers: int = 3

    # MTP
    num_nextn_predict_layers: int = 1

    # Scoring
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Derived (computed in __post_init__)
    nope_head_dim: int = 0

    # Parallelism (set externally)
    tp_size: int = 1
    ep_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    etp_size: int = 1

    # Training
    micro_batch_size: int = 1
    seq_length: int = 4096
    use_distributed_optimizer: bool = True
    data_parallel_size: int = 1
    world_size: int = 8
    recompute_granularity: str = "selective"
    recompute_modules: Optional[list] = None
    recompute_num_layers: int = 1
    recompute_method: str = "uniform"

    # Pipeline
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None

    # Unfused attention: count intermediate tensors (logits, softmax, gather buffers)
    # that fused kernels (flash attention, fused gate) would avoid materializing
    unfused_attn: bool = False

    def __post_init__(self):
        self.nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if self.recompute_modules is None:
            self.recompute_modules = []

    @classmethod
    def from_hf_config(cls, hf_config: dict, **kwargs):
        """Create config from HuggingFace config.json."""
        return cls(
            hidden_size=hf_config.get("hidden_size", 7168),
            num_hidden_layers=hf_config.get("num_hidden_layers", 43),
            num_attention_heads=hf_config.get("num_attention_heads", 128),
            head_dim=hf_config.get("head_dim", 512),
            q_lora_rank=hf_config.get("q_lora_rank", 1536),
            qk_rope_head_dim=hf_config.get("qk_rope_head_dim", 64),
            o_groups=hf_config.get("o_groups", 16),
            o_lora_rank=hf_config.get("o_lora_rank", 1024),
            window_size=hf_config.get("sliding_window", 128),
            vocab_size=hf_config.get("vocab_size", 129280),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            compress_ratios=tuple(hf_config.get("compress_ratios", [])),
            index_n_heads=hf_config.get("index_n_heads", 64),
            index_head_dim=hf_config.get("index_head_dim", 128),
            index_topk=hf_config.get("index_topk", 512),
            hc_mult=hf_config.get("hc_mult", 4),
            hc_sinkhorn_iters=hf_config.get("hc_sinkhorn_iters", 20),
            hc_eps=hf_config.get("hc_eps", 1e-6),
            n_routed_experts=hf_config.get("n_routed_experts", 256),
            n_shared_experts=hf_config.get("n_shared_experts", 1),
            num_experts_per_tok=hf_config.get("num_experts_per_tok", 6),
            moe_intermediate_size=hf_config.get("moe_intermediate_size", 2048),
            num_hash_layers=hf_config.get("num_hash_layers", 3),
            num_nextn_predict_layers=hf_config.get("num_nextn_predict_layers", 1),
            scoring_func=hf_config.get("scoring_func", "sqrtsoftplus"),
            routed_scaling_factor=hf_config.get("routed_scaling_factor", 1.5),
            swiglu_limit=hf_config.get("swiglu_limit", 10.0),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------

class DSV4Linear(MemEstimator):
    """Linear layer without TP splitting (used for wq_a, wkv, wgate, etc.)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight = (out_features, in_features)

    def num_parameter(self):
        p = cum_mul(self.weight)
        if self.has_bias:
            p += self.out_features
        return p

    def num_activation(self, input_shape: list[int]):
        # Activation = output tensor size (saved for backward)
        return cum_mul(input_shape[:-1]) * self.out_features

    def mock_forward(self, input_shape: list[int]):
        return input_shape[:-1] + [self.out_features]


class DSV4ColumnParallelLinear(MemEstimator):
    """Column-parallel linear: output dim split by TP."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        tp_size = get_tensor_model_parallel_world_size()
        self.out_features_per_partition = divide(out_features, tp_size)
        self.has_bias = bias
        self.weight = (self.out_features_per_partition, in_features)

    def num_parameter(self):
        p = cum_mul(self.weight)
        if self.has_bias:
            p += self.out_features_per_partition
        return p

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:-1]) * self.out_features_per_partition

    def mock_forward(self, input_shape: list[int]):
        return input_shape[:-1] + [self.out_features_per_partition]


class DSV4RowParallelLinear(MemEstimator):
    """Row-parallel linear: input dim split by TP, all-reduce on output."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.in_features_per_partition = divide(in_features, tp_size)
        self.out_features = out_features
        self.has_bias = bias
        self.weight = (out_features, self.in_features_per_partition)

    def num_parameter(self):
        p = cum_mul(self.weight)
        if self.has_bias:
            p += self.out_features
        return p

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:-1]) * self.in_features_per_partition

    def mock_forward(self, input_shape: list[int]):
        return input_shape[:-1] + [self.out_features]


class DSV4RMSNorm(MemEstimator):
    """RMS normalization."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight_size = hidden_size

    def num_parameter(self):
        return self.weight_size

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape[:])

    def mock_forward(self, input_shape: list[int]):
        return input_shape


class DSV4Parameter(MemEstimator):
    """A learnable parameter tensor (e.g. attn_sink, ape)."""

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = tuple(shape)

    def num_parameter(self):
        p = 1
        for s in self.shape:
            p *= s
        return p

    def num_activation(self, input_shape: list[int]):
        return 0

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# KV Compressor (CSA / HCA)
# ---------------------------------------------------------------------------

class DSV4Compressor(MemEstimator):
    """KV Compressor for CSA (ratio=4, overlap=True) and HCA (ratio>=128, overlap=False).

    Parameters:
      - wkv:  hidden_size × (coff × head_dim)   [coff = 2 if overlap else 1]
      - wgate: hidden_size × (coff × head_dim)
      - ape:   ratio × (coff × head_dim)
      - norm:  head_dim (RMSNorm)

    For CSA (ratio=4): overlap=True, coff=2
      Compression: 2m tokens → 1 compressed token (overlapping windows)
    For HCA (ratio>=128): overlap=False, coff=1
      Compression: m tokens → 1 compressed token (non-overlapping)
    """

    def __init__(self, config: DSV4Config, compress_ratio: int, head_dim: int, is_indexer: bool = False):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.is_indexer = is_indexer

        # CSA (ratio=4) uses overlapping compression: coff=2
        # HCA (ratio>=128) uses non-overlapping: coff=1
        self.overlap = (compress_ratio == 4)
        self.coff = 2 if self.overlap else 1

        # wkv and wgate are NOT split by TP (MQA-style, single KV head)
        self.wkv = DSV4Linear(config.hidden_size, self.coff * head_dim)
        self.wgate = DSV4Linear(config.hidden_size, self.coff * head_dim)

        # APE: learnable position bias [ratio, coff * head_dim]
        # Stored in fp32 but counted as params
        self.ape = DSV4Parameter(compress_ratio, self.coff * head_dim)

        # Norm after compression
        self.norm = DSV4RMSNorm(head_dim)

    def num_parameter(self):
        return (self.wkv.num_parameter()
                + self.wgate.num_parameter()
                + self.ape.num_parameter()
                + self.norm.num_parameter())

    def num_activation(self, input_shape: list[int]):
        """Activation for compression forward pass.

        input_shape: [B, S, H] or [S, B, H]
        """
        ret = 0
        # wkv output: B * S * (coff * head_dim)
        ret += self.wkv.num_activation(input_shape)
        # wgate output: B * S * (coff * head_dim)
        ret += self.wgate.num_activation(input_shape)
        # After softmax + weighted sum: B * (S/ratio) * head_dim
        # (compression reduces sequence length by ratio)
        # But we need to save the compressed output for backward
        bs = cum_mul(input_shape[:-1])
        ret += bs * self.head_dim  # compressed KV
        # norm output
        ret += bs * self.head_dim

        # Unfused: gate sigmoid and gated product must be materialized
        if self.config.unfused_attn:
            # sigmoid(wgate_output): same size as wgate output
            ret += bs * self.coff * self.head_dim
            # wkv * sigmoid(wgate): element-wise product, same size
            ret += bs * self.coff * self.head_dim
            # Segment softmax over ratio dimension: B * (S/ratio) * ratio
            ret += bs  # S = (S/ratio) * ratio

        return ret

    def mock_forward(self, input_shape: list[int]):
        # After compression: sequence length reduced by ratio
        # input_shape: [..., S, H] -> [..., S/ratio, H]  (simplified)
        # Actually we keep the same shape since the compressed KV is used
        # alongside the window KV in the attention layer
        return input_shape


# ---------------------------------------------------------------------------
# Sparse Indexer (CSA only, ratio=4)
# ---------------------------------------------------------------------------

class DSV4Indexer(MemEstimator):
    """Lightning Indexer for CSA: selects top-k compressed KV blocks.

    Parameters:
      - wq_b (ColumnParallel): q_lora_rank × (index_n_heads × index_head_dim) / TP
      - weights_proj (ColumnParallel): hidden_size × index_n_heads / TP
      - compressor: DSV4Compressor with index_head_dim
    """

    def __init__(self, config: DSV4Config, compress_ratio: int):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio

        # Indexer query projection (ColumnParallel, split by TP)
        self.wq_b = DSV4ColumnParallelLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
        )

        # Weight projection for multi-head scoring (ColumnParallel)
        self.weights_proj = DSV4ColumnParallelLinear(
            config.hidden_size,
            config.index_n_heads,
        )

        # Indexer has its own compressor with index_head_dim
        self.compressor = DSV4Compressor(
            config, compress_ratio, config.index_head_dim, is_indexer=True
        )

    def num_parameter(self):
        return (self.wq_b.num_parameter()
                + self.weights_proj.num_parameter()
                + self.compressor.num_parameter())

    def num_activation(self, input_shape: list[int]):
        """Activation for indexer forward pass.

        input_shape: [B, S, H]
        """
        ret = 0
        bs = cum_mul(input_shape[:-1])
        tp = get_tensor_model_parallel_world_size()

        # wq_b output: B * S * (index_n_heads * index_head_dim / TP)
        ret += self.wq_b.num_activation(input_shape)
        # weights_proj output: B * S * (index_n_heads / TP)
        ret += self.weights_proj.num_activation(input_shape)
        # Compressor activations
        ret += self.compressor.num_activation(input_shape)

        if self.config.unfused_attn:
            # Indexer attention logits: B * (index_n_heads/TP) * S * compressed_len
            # Count at full S; CP division applied uniformly at report level
            total_seq = input_shape[1] if len(input_shape) > 1 else input_shape[0]
            compressed_len = total_seq // self.compress_ratio
            idx_heads_tp = self.config.index_n_heads // tp
            # bs = B * S (full sequence), after /cp gives B * (S/cp)
            idx_logits = bs * idx_heads_tp * compressed_len
            ret += idx_logits   # attention logits (Q * K^T)
            ret += idx_logits   # softmax output

            # Top-k selection results
            ret += bs * self.config.index_topk       # scores (bf16)
            ret += bs * self.config.index_topk       # indices (counted as bf16 equiv)

        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# V4 Attention (Window / CSA / HCA)
# ---------------------------------------------------------------------------

class DSV4Attention(MemEstimator):
    """DeepSeek-V4 attention layer.

    Three variants based on compress_ratio:
    - ratio=0: Window-only (sliding window, no compression)
    - ratio=4: CSA (compress + sparse index + window + MQA)
    - ratio>=128: HCA (heavy compress + window + MQA, no indexer)

    Parameters:
      Query path:
        - wq_a: hidden_size × q_lora_rank (no TP)
        - q_norm: RMSNorm(q_lora_rank)
        - wq_b: q_lora_rank × (n_heads × head_dim) / TP (ColumnParallel)
        - attn_sink: n_heads / TP (learnable per-head scalar)

      KV path (window, MQA - single KV head, NOT split by TP):
        - wkv: hidden_size × head_dim
        - kv_norm: RMSNorm(head_dim)

      Compressor (optional, for CSA/HCA):
        - DSV4Compressor

      Indexer (optional, for CSA only):
        - DSV4Indexer

      Output path (grouped low-rank):
        - wo_a: (n_heads * head_dim / o_groups) × (o_groups * o_lora_rank) / TP (ColumnParallel)
        - wo_b: (o_groups * o_lora_rank) / TP × hidden_size (RowParallel)
    """

    def __init__(self, config: DSV4Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        tp_size = get_tensor_model_parallel_world_size()
        self.n_heads_per_partition = divide(config.num_attention_heads, tp_size)
        self.o_groups_per_partition = divide(config.o_groups, tp_size)

        # Get compress_ratio for this layer
        if layer_id < len(config.compress_ratios):
            self.compress_ratio = config.compress_ratios[layer_id]
        else:
            self.compress_ratio = 0  # default: window-only

        self.is_csa = (self.compress_ratio == 4)
        self.is_hca = (self.compress_ratio >= 128)
        self.is_window = (self.compress_ratio == 0)

        # --- Query path ---
        self.wq_a = DSV4Linear(config.hidden_size, config.q_lora_rank)
        self.q_norm = DSV4RMSNorm(config.q_lora_rank)
        self.wq_b = DSV4ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * config.head_dim,
        )

        # Per-head RMSNorm on Q (applied after wq_b reshape)
        # This is a parameterless operation in terms of learnable params,
        # but we track it for activation. No separate norm weights since
        # it's per-head rsqrt(x^2.mean()), similar to the reference impl.

        # Attention sink: learnable scalar per head
        self.attn_sink = DSV4Parameter(self.n_heads_per_partition)

        # --- KV path (window, MQA) ---
        # wkv projects to head_dim (single KV head, not split by TP)
        self.wkv = DSV4Linear(config.hidden_size, config.head_dim)
        self.kv_norm = DSV4RMSNorm(config.head_dim)

        # --- Compressor (CSA/HCA) ---
        self.compressor = None
        if not self.is_window:
            self.compressor = DSV4Compressor(
                config, self.compress_ratio, config.head_dim
            )

        # --- Indexer (CSA only) ---
        self.indexer = None
        if self.is_csa:
            self.indexer = DSV4Indexer(config, self.compress_ratio)

        # --- Output path (grouped low-rank) ---
        # wo_a: ColumnParallel, input = n_heads * head_dim / o_groups
        #                    output = o_groups * o_lora_rank (split by TP)
        wo_a_in = config.num_attention_heads * config.head_dim // config.o_groups
        wo_a_out = config.o_groups * config.o_lora_rank
        self.wo_a = DSV4ColumnParallelLinear(wo_a_in, wo_a_out)

        # wo_b: RowParallel, input = o_groups * o_lora_rank (split by TP)
        #                    output = hidden_size
        wo_b_in = config.o_groups * config.o_lora_rank
        self.wo_b = DSV4RowParallelLinear(wo_b_in, config.hidden_size)

        # Recompute
        self.checkpoint_core_attention = (
            config.recompute_granularity == "selective"
            and "core_attn" in config.recompute_modules
        )

    def num_parameter(self):
        ret = 0
        # Query path
        ret += self.wq_a.num_parameter()
        ret += self.q_norm.num_parameter()
        ret += self.wq_b.num_parameter()
        ret += self.attn_sink.num_parameter()
        # KV path
        ret += self.wkv.num_parameter()
        ret += self.kv_norm.num_parameter()
        # Compressor
        if self.compressor is not None:
            ret += self.compressor.num_parameter()
        # Indexer
        if self.indexer is not None:
            ret += self.indexer.num_parameter()
        # Output path
        ret += self.wo_a.num_parameter()
        ret += self.wo_b.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        """Compute activation for the attention layer.

        input_shape: [B, S, H] where B=micro_batch, S=seq_len
        """
        ret = 0
        bs = cum_mul(input_shape[:-1])  # B * S
        tp = get_tensor_model_parallel_world_size()

        # --- Query path ---
        # wq_a: [B, S, H] -> [B, S, q_lora_rank]
        ret += self.wq_a.num_activation(input_shape)
        q_shape = self.wq_a.mock_forward(input_shape)

        # q_norm: [B, S, q_lora_rank]
        ret += self.q_norm.num_activation(q_shape)

        # wq_b: [B, S, q_lora_rank] -> [B, S, n_heads*head_dim/TP]
        ret += self.wq_b.num_activation(q_shape)
        q_out_shape = self.wq_b.mock_forward(q_shape)

        # Per-head RMSNorm on Q (no params, but activation saved for backward)
        ret += cum_mul(q_out_shape)

        # --- KV path (window) ---
        # wkv: [B, S, H] -> [B, S, head_dim] (MQA, no TP split)
        ret += self.wkv.num_activation(input_shape)
        kv_shape = self.wkv.mock_forward(input_shape)

        # kv_norm: [B, S, head_dim]
        ret += self.kv_norm.num_activation(kv_shape)

        # --- Compressor (CSA/HCA) ---
        if self.compressor is not None:
            ret += self.compressor.num_activation(input_shape)

        # --- Indexer (CSA only) ---
        if self.indexer is not None:
            # Indexer takes the compressed q (from wq_a output) and hidden states
            ret += self.indexer.num_activation(input_shape)

        # --- Core attention ---
        # num_kv: number of KV tokens the main attention attends to
        # Window: window_size only
        # CSA: window_size + topk (sparse selection via indexer)
        # HCA: window_size + total_seq/compress_ratio (dense attention on all compressed KV)
        if not self.checkpoint_core_attention:
            total_seq = input_shape[1] if len(input_shape) > 1 else input_shape[0]
            num_kv = self.config.window_size
            if self.is_csa:
                # CSA: only attend to window + top-k selected compressed KV
                num_kv += self.config.index_topk
            elif self.is_hca:
                # HCA: attend to all compressed KV tokens
                num_kv += total_seq // self.compress_ratio

            # Attention output: B * S * n_heads_per_partition * head_dim
            ret += bs * self.n_heads_per_partition * self.config.head_dim

            # Unfused: materialize attention logits, softmax, and gathered KV buffer
            # Count at full S; CP division is applied uniformly at the report level
            if self.config.unfused_attn:
                # Attention logits: B * n_heads_tp * S * num_kv
                # After /cp: B * n_heads_tp * (S/cp) * num_kv
                attn_logits = bs * self.n_heads_per_partition * num_kv
                ret += attn_logits  # Q * K^T logits
                ret += attn_logits  # softmax output

                # Gathered KV buffer for CSA: [B, S, window_size + topk, head_dim]
                # After top-k selection, window KV + selected compressed KV are gathered
                # into a contiguous buffer for the attention kernel.
                # Fused kernels avoid materializing this; unfused must store it.
                if self.is_csa:
                    gathered_kv = bs * num_kv * self.config.head_dim
                    ret += gathered_kv

        # --- Output path ---
        # wo_a: [B, S, n_heads*head_dim/o_groups] -> [B, S, o_groups*o_lora_rank/TP]
        # Input to wo_a is the grouped attention output
        wo_a_input_size = self.config.num_attention_heads * self.config.head_dim // self.config.o_groups
        wo_a_input_shape = input_shape[:-1] + [wo_a_input_size]
        ret += self.wo_a.num_activation(wo_a_input_shape)
        wo_a_out_shape = self.wo_a.mock_forward(wo_a_input_shape)

        # wo_b: [B, S, o_groups*o_lora_rank/TP] -> [B, S, H]
        ret += self.wo_b.num_activation(wo_a_out_shape)

        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# Hyper-Connection
# ---------------------------------------------------------------------------

class DSV4HyperConnection(MemEstimator):
    """Hyper-Connection residual mixing.

    Replaces simple add residual with learned weighted mixing.
    Per sublayer (attn/ffn), adds:
      - hc_fn: [mix_hc, hc_mult * hidden_size]  (main weight)
      - hc_base: [mix_hc]                        (bias)
      - hc_scale: [3]                             (scale)

    Where mix_hc = (2 + hc_mult) * hc_mult
    """

    def __init__(self, config: DSV4Config):
        super().__init__()
        self.config = config
        hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        mix_hc = (2 + hc_mult) * hc_mult

        self.hc_fn = DSV4Parameter(mix_hc, hc_dim)
        self.hc_base = DSV4Parameter(mix_hc)
        self.hc_scale = DSV4Parameter(3)

    def num_parameter(self):
        return (self.hc_fn.num_parameter()
                + self.hc_base.num_parameter()
                + self.hc_scale.num_parameter())

    def num_activation(self, input_shape: list[int]):
        """HC adds activation for the mixing computation and hc_mult copies of residual.

        input_shape: [B, S, H]
        """
        bs = cum_mul(input_shape[:-1])
        hc_mult = self.config.hc_mult

        # hc_fn linear: B * S * mix_hc (output of linear projection)
        mix_hc = (2 + hc_mult) * hc_mult
        ret = bs * mix_hc

        # hc_mult copies of hidden state for residual
        ret += bs * hc_mult * self.config.hidden_size

        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# MoE Gate
# ---------------------------------------------------------------------------

class DSV4Gate(MemEstimator):
    """MoE gate with optional hash routing.

    Score-based: weight [n_routed_experts, hidden_size] + bias [n_routed_experts]
    Hash-based:  weight [n_routed_experts, hidden_size] + tid2eid [vocab_size, num_experts_per_tok]
    """

    def __init__(self, config: DSV4Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.is_hash = (layer_id < config.num_hash_layers)

        # Gate weight (not split by TP or EP - all ranks need full gate for routing)
        self.weight = DSV4Parameter(config.n_routed_experts, config.hidden_size)

        if self.is_hash:
            # Hash routing: tid2eid table (not trainable, but occupies memory)
            self.tid2eid = DSV4Parameter(config.vocab_size, config.num_experts_per_tok)
            self.bias = None
        else:
            # Score-based routing: bias
            self.bias = DSV4Parameter(config.n_routed_experts)
            self.tid2eid = None

    def num_parameter(self):
        ret = self.weight.num_parameter()
        if self.bias is not None:
            ret += self.bias.num_parameter()
        # tid2eid is not trainable but occupies memory
        if self.tid2eid is not None:
            ret += self.tid2eid.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        # Gate scores: B * S * n_routed_experts
        bs = cum_mul(input_shape[:-1])
        ret = bs * self.config.n_routed_experts
        # Top-k indices and weights: B * S * num_experts_per_tok
        ret += bs * self.config.num_experts_per_tok * 2  # indices + weights
        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# Expert FFN
# ---------------------------------------------------------------------------

class DSV4ExpertFFN(MemEstimator):
    """Single expert SwiGLU FFN.

    w1: hidden_size × moe_intermediate_size (gate)
    w3: hidden_size × moe_intermediate_size (up)
    w2: moe_intermediate_size × hidden_size (down)
    """

    def __init__(self, config: DSV4Config, is_shared: bool = False):
        super().__init__()
        self.config = config
        self.is_shared = is_shared
        self.intermediate_size = config.moe_intermediate_size

        if is_shared:
            # Shared expert: split by TP
            tp_size = get_tensor_model_parallel_world_size()
            self.w1_size = (config.hidden_size, divide(config.moe_intermediate_size * 2, tp_size))
            self.w2_size = (divide(config.moe_intermediate_size, tp_size), config.hidden_size)
        else:
            # Routed expert: split by ETP
            try:
                etp_size = get_expert_tensor_parallel_world_size()
            except Exception:
                etp_size = config.etp_size
            self.w1_size = (config.hidden_size, divide(config.moe_intermediate_size * 2, etp_size))
            self.w2_size = (divide(config.moe_intermediate_size, etp_size), config.hidden_size)

        self.activation_recompute = (
            config.recompute_granularity == "selective"
            and "moe_act" in config.recompute_modules
        )

    def num_parameter(self):
        # w1 (gate) + w3 (up) are fused: hidden_size × (intermediate * 2)
        # w2 (down): intermediate × hidden_size
        w1_params = self.w1_size[0] * self.w1_size[1]
        w2_params = self.w2_size[0] * self.w2_size[1]
        return w1_params + w2_params

    def num_activation(self, input_shape: list[int], tokens_per_expert=None):
        if self.activation_recompute:
            return 0
        bs = cum_mul(input_shape[:-1])
        if not self.is_shared and tokens_per_expert is not None:
            bs = tokens_per_expert
        # fc1 output (gate + up): B * S * (intermediate * 2 / TP_or_ETP)
        ret = bs * self.w1_size[1]
        # SwiGLU intermediate: B * S * (intermediate / TP_or_ETP)
        ret += bs * self.w2_size[0]
        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------

class DSV4MoELayer(MemEstimator):
    """MoE layer with routed + shared experts."""

    def __init__(self, config: DSV4Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        ep_size = get_expert_model_parallel_world_size()
        self.num_local_experts = divide(config.n_routed_experts, ep_size)

        self.gate = DSV4Gate(config, layer_id)
        self.routed_experts = DSV4ExpertFFN(config, is_shared=False)
        self.use_shared_expert = (config.n_shared_experts > 0)
        if self.use_shared_expert:
            self.shared_expert = DSV4ExpertFFN(config, is_shared=True)

        self.moe_layer_recompute = (
            config.recompute_granularity == "selective"
            and "moe" in config.recompute_modules
        )

    def num_parameter(self):
        ret = self.gate.num_parameter()
        ret += self.routed_experts.num_parameter() * self.num_local_experts
        if self.use_shared_expert:
            ret += self.shared_expert.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        if self.moe_layer_recompute:
            return 0

        tp_size = get_tensor_model_parallel_world_size()
        etp_size = 1
        try:
            etp_size = get_expert_tensor_parallel_world_size()
        except Exception:
            etp_size = self.config.etp_size

        bs = cum_mul(input_shape[:-1])
        topk = self.config.num_experts_per_tok

        # Gate activation
        ret = self.gate.num_activation(input_shape)

        # Token dispatch: B * S * topk * hidden_size
        ret += bs * topk * self.config.hidden_size

        # Routed experts: tokens are distributed across experts
        # Approximate: each expert gets B * S * topk / num_local_experts tokens
        tokens_per_expert = bs * topk // max(self.num_local_experts, 1)
        ret += self.routed_experts.num_activation(input_shape, tokens_per_expert) * self.num_local_experts

        # Shared expert: all tokens
        if self.use_shared_expert:
            ret += self.shared_expert.num_activation(input_shape)

        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# Transformer Block (with Hyper-Connection)
# ---------------------------------------------------------------------------

class DSV4Block(MemEstimator):
    """DeepSeek-V4 Transformer block.

    Structure:
      x_in [B, S, hc_mult, H]
      -> hc_pre: mix hc_mult copies -> [B, S, H]
      -> attn_norm: RMSNorm
      -> attention: window/CSA/HCA
      -> hc_post: expand -> [B, S, hc_mult, H]
      -> hc_pre: mix -> [B, S, H]
      -> ffn_norm: RMSNorm
      -> MoE
      -> hc_post: expand -> [B, S, hc_mult, H]
    """

    def __init__(self, config: DSV4Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        # Hyper-Connections
        self.hc_attn = DSV4HyperConnection(config)
        self.hc_ffn = DSV4HyperConnection(config)

        # Norms
        self.attn_norm = DSV4RMSNorm(config.hidden_size)
        self.ffn_norm = DSV4RMSNorm(config.hidden_size)

        # Attention
        self.attention = DSV4Attention(config, layer_id)

        # MoE
        self.moe = DSV4MoELayer(config, layer_id)

        # Recompute flags
        self.recompute_attn_norm = (
            config.recompute_granularity == "selective"
            and "layernorm" in config.recompute_modules
        )
        self.recompute_ffn_norm = (
            config.recompute_granularity == "selective"
            and "layernorm" in config.recompute_modules
        )
        self.recompute_moe = (
            config.recompute_granularity == "selective"
            and "mlp" in config.recompute_modules
        )

    def num_parameter(self):
        ret = 0
        ret += self.hc_attn.num_parameter()
        ret += self.hc_ffn.num_parameter()
        ret += self.attn_norm.num_parameter()
        ret += self.ffn_norm.num_parameter()
        ret += self.attention.num_parameter()
        ret += self.moe.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        ret = 0

        # Attention sub-block
        ret += self.attention.num_activation(input_shape)

        # Attention norm (saved for backward unless recomputed)
        if not self.recompute_attn_norm:
            ret += self.attn_norm.num_activation(input_shape)

        # HC activation for attention
        ret += self.hc_attn.num_activation(input_shape)

        # MoE sub-block
        if not self.recompute_moe:
            ret += self.moe.num_activation(input_shape)

        # FFN norm
        if not self.recompute_ffn_norm:
            ret += self.ffn_norm.num_activation(input_shape)

        # HC activation for FFN
        ret += self.hc_ffn.num_activation(input_shape)

        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class DSV4Embedding(MemEstimator):
    """Token embedding with vocab parallelism."""

    def __init__(self, config: DSV4Config):
        super().__init__()
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()
        self.vocab_size_per_partition = divide(config.vocab_size, tp_size)
        self.hidden_size = config.hidden_size

    def num_parameter(self):
        return self.vocab_size_per_partition * self.hidden_size

    def num_activation(self, input_shape: list[int]):
        return cum_mul(input_shape) * self.hidden_size

    def mock_forward(self, input_shape: list[int]):
        return input_shape + [self.hidden_size]


# ---------------------------------------------------------------------------
# Output Head
# ---------------------------------------------------------------------------

class DSV4OutputHead(MemEstimator):
    """Output projection (lm_head) with vocab parallelism.

    Parameters:
      - weight: [vocab_size / TP, hidden_size] (stored in fp32)
      - HC head: hc_head_fn, hc_head_base, hc_head_scale
      - norm: RMSNorm(hidden_size)
    """

    def __init__(self, config: DSV4Config):
        super().__init__()
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()
        self.vocab_size_per_partition = divide(config.vocab_size, tp_size)
        self.hidden_size = config.hidden_size

        # HC head parameters
        hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = DSV4Parameter(hc_mult, hc_dim)
        self.hc_head_base = DSV4Parameter(hc_mult)
        self.hc_head_scale = DSV4Parameter(1)
        self.norm = DSV4RMSNorm(config.hidden_size)

    def num_parameter(self):
        # lm_head weight (fp32)
        ret = self.vocab_size_per_partition * self.hidden_size
        # HC head
        ret += self.hc_head_fn.num_parameter()
        ret += self.hc_head_base.num_parameter()
        ret += self.hc_head_scale.num_parameter()
        # Norm
        ret += self.norm.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        # Output projection: B * S * vocab_size_per_partition (only last token for LM)
        # For training, we save the full output for backward
        bs = cum_mul(input_shape[:-1])
        ret = bs * self.vocab_size_per_partition
        # HC head activation
        hc_mult = self.config.hc_mult
        mix_hc = hc_mult  # head uses sigmoid, not sinkhorn
        ret += bs * mix_hc * self.config.hidden_size
        # Norm
        ret += self.norm.num_activation(input_shape)
        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape[:-1] + [self.vocab_size_per_partition]


# ---------------------------------------------------------------------------
# MTP Block
# ---------------------------------------------------------------------------

class DSV4MTPBlock(MemEstimator):
    """Multi-Token Prediction block.

    Adds projection layers on top of a regular Block:
      - e_proj: hidden_size × hidden_size (embedding projection)
      - h_proj: hidden_size × hidden_size (hidden projection)
      - enorm: RMSNorm(hidden_size)
      - hnorm: RMSNorm(hidden_size)
    Plus the regular Block (attention + MoE + HC).
    """

    def __init__(self, config: DSV4Config, layer_id: int):
        super().__init__()
        self.config = config

        # MTP-specific projections
        self.e_proj = DSV4Linear(config.hidden_size, config.hidden_size)
        self.h_proj = DSV4Linear(config.hidden_size, config.hidden_size)
        self.enorm = DSV4RMSNorm(config.hidden_size)
        self.hnorm = DSV4RMSNorm(config.hidden_size)

        # Regular block
        self.block = DSV4Block(config, layer_id)

    def num_parameter(self):
        ret = 0
        ret += self.e_proj.num_parameter()
        ret += self.h_proj.num_parameter()
        ret += self.enorm.num_parameter()
        ret += self.hnorm.num_parameter()
        ret += self.block.num_parameter()
        return ret

    def num_activation(self, input_shape: list[int]):
        ret = 0
        ret += self.e_proj.num_activation(input_shape)
        ret += self.h_proj.num_activation(input_shape)
        ret += self.enorm.num_activation(input_shape)
        ret += self.hnorm.num_activation(input_shape)
        ret += self.block.num_activation(input_shape)
        return ret

    def mock_forward(self, input_shape: list[int]):
        return input_shape


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class DSV4Model(MemEstimator):
    """Full DeepSeek-V4 model estimator.

    Structure:
      - Embedding
      - N × DSV4Block
      - Final RMSNorm
      - Output head (with HC)
      - MTP blocks (optional)
    """

    def __init__(self, config: DSV4Config, pre_process: bool = True, post_process: bool = True):
        super().__init__()
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process

        # Determine layer range for this PP rank
        pp_rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()

        num_layers = config.num_hidden_layers
        if pp_size > 1:
            if config.num_layers_in_first_pipeline_stage is not None:
                first_layers = config.num_layers_in_first_pipeline_stage
                last_layers = config.num_layers_in_last_pipeline_stage or 0
                middle_layers = num_layers - first_layers - last_layers
                middle_stages = pp_size - (1 if first_layers else 0) - (1 if last_layers else 0)
                if pp_rank == 0 and first_layers:
                    self.num_layers = first_layers
                    self.layer_offset = 0
                elif pp_rank == pp_size - 1 and last_layers:
                    self.layer_offset = first_layers + middle_layers
                    self.num_layers = last_layers
                else:
                    offset = first_layers
                    rank_offset = pp_rank - (1 if first_layers else 0)
                    self.num_layers = middle_layers // middle_stages
                    self.layer_offset = offset + rank_offset * self.num_layers
            else:
                self.num_layers = num_layers // pp_size
                self.layer_offset = pp_rank * self.num_layers
        else:
            self.num_layers = num_layers
            self.layer_offset = 0

        # Embedding (only on first PP rank)
        self.embedding = None
        if pre_process:
            self.embedding = DSV4Embedding(config)

        # Transformer blocks
        self.blocks = []
        for i in range(self.num_layers):
            layer_id = self.layer_offset + i
            self.blocks.append(DSV4Block(config, layer_id))

        # Final norm and output head (only on last PP rank)
        self.final_norm = None
        self.output_head = None
        self.mtp_blocks = []
        if post_process:
            self.final_norm = DSV4RMSNorm(config.hidden_size)
            self.output_head = DSV4OutputHead(config)
            # MTP blocks
            for i in range(config.num_nextn_predict_layers):
                layer_id = num_layers + i
                mtp_layer_id = layer_id if layer_id < len(config.compress_ratios) else num_layers - 1
                self.mtp_blocks.append(DSV4MTPBlock(config, mtp_layer_id))

        # Track per-layer activation for recompute
        self._num_act_pre = 0  # activation before first layer
        self._num_act_post = 0  # activation after last layer (loss)
        self._num_act_per_layer = 0  # activation per layer (for full recompute)
        self._num_act_between_layers = 0  # activation between layers

    def num_parameter(self):
        ret = 0
        if self.embedding is not None:
            ret += self.embedding.num_parameter()
        for block in self.blocks:
            ret += block.num_parameter()
        if self.final_norm is not None:
            ret += self.final_norm.num_parameter()
        if self.output_head is not None:
            ret += self.output_head.num_parameter()
        for mtp in self.mtp_blocks:
            ret += mtp.num_parameter()
        return ret

    def num_parameter_sparse(self):
        """Count MoE expert parameters (excluding shared expert)."""
        ret = 0
        for block in self.blocks:
            moe = block.moe
            ret += moe.routed_experts.num_parameter() * moe.num_local_experts
        for mtp in self.mtp_blocks:
            moe = mtp.block.moe
            ret += moe.routed_experts.num_parameter() * moe.num_local_experts
        return ret

    def num_activation(self, input_shape: list[int]):
        ret = 0
        current_shape = input_shape[:]

        # Embedding
        if self.embedding is not None:
            ret += self.embedding.num_activation(current_shape)
            current_shape = self.embedding.mock_forward(current_shape)
        else:
            # Non-first PP rank: input_shape is already the embedding output shape
            # [mbs, seq_len, hidden_size], no reshape needed
            pass

        self._num_act_pre = ret

        # Transformer blocks
        layer_acts = []
        for block in self.blocks:
            layer_act = block.num_activation(current_shape)
            layer_acts.append(layer_act)
            ret += layer_act

        if layer_acts:
            self._num_act_per_layer = max(layer_acts)
            # Between-layer activation: only the hidden state tensor that must be
            # saved for recomputation during backward (not the full layer activation)
            from moe_mem_estimator.base import cum_mul
            self._num_act_between_layers = cum_mul(current_shape)

        # Final norm + output head
        if self.final_norm is not None:
            post_act = self.final_norm.num_activation(current_shape)
            if self.output_head is not None:
                post_act += self.output_head.num_activation(current_shape)
            ret += post_act
            self._num_act_post = post_act
        else:
            self._num_act_post = 0

        # MTP blocks
        for mtp in self.mtp_blocks:
            ret += mtp.num_activation(current_shape)

        return ret

    def mock_forward(self, input_shape: list[int]):
        if self.embedding is not None:
            input_shape = self.embedding.mock_forward(input_shape)
        return input_shape

    def __repr__(self):
        lines = []
        lines.append(f"DSV4Model(")
        if self.embedding is not None:
            lines.append(f"  (embedding): {self.embedding!r}")
        for i, block in enumerate(self.blocks):
            if i < 3 or i >= len(self.blocks) - 2:
                lines.append(f"  (block.{i}): {block!r}")
            elif i == 3:
                lines.append(f"  ... ({len(self.blocks) - 4} more blocks)")
        if self.final_norm is not None:
            lines.append(f"  (final_norm): {self.final_norm!r}")
        if self.output_head is not None:
            lines.append(f"  (output_head): {self.output_head!r}")
        for i, mtp in enumerate(self.mtp_blocks):
            lines.append(f"  (mtp.{i}): {mtp!r}")

        total_params = self.num_parameter()
        total_act = self.num_activation([self.config.micro_batch_size, self.config.seq_length])
        lines.append(f"  /* n_params={total_params / 1024 / 1024:.2f}M"
                     f"  n_act={total_act / 1024 / 1024:.2f}M */")
        lines.append(")")
        return "\n".join(lines)