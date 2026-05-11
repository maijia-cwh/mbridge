# 显存估算逻辑总结

## 1. 整体流程

```
start.sh / main.py
  └─ estimate_with_mbridge(config)
       ├─ patch_parallel_states(config)         # 模拟分布式并行状态
       ├─ AutoBridge.from_pretrained(model)      # 加载 HF 配置 → Megatron TransformerConfig
       ├─ 设置并行/重计算参数到 tf_config
       └─ estimate_from_config(tf_config, args)  # 核心估算入口
            ├─ 遍历每个 PP rank
            │   └─ report_memory_usage_one_pp_rank()
            │        ├─ 构建模型 (GPTModel)
            │        ├─ 统计参数量 → 静态显存
            │        ├─ 统计激活量 → 动态显存
            │        └─ 输出该 rank 的显存报告
            └─ 汇总所有 PP rank 报告
```

## 2. 模型结构与参数统计

每个组件继承 `MemEstimator`，实现三个方法：

| 方法 | 作用 |
|------|------|
| `num_parameter()` | 该模块的可训练参数元素数 |
| `num_activation(input_shape)` | 前向传播中需要保存的激活元素数 |
| `mock_forward(input_shape)` | 模拟前向传播，返回输出 shape |

### 模型组装层级

```
GPTModel
  ├─ LanguageModelEmbedding          (仅 pre_process=True 的 PP rank)
  │   └─ VocabParallelEmbedding      词表按 TP 切分
  ├─ TransformerBlock
  │   └─ ModuleList[TransformerLayer × num_layers]
  │        ├─ input_layernorm
  │        ├─ SelfAttention / MLASelfAttention
  │        │   ├─ linear_qkv (或 MLA 的 q_down/q_up/kv_down/kv_up)
  │        │   ├─ core_attention (TEDotProductAttention)
  │        │   └─ linear_proj
  │        ├─ pre_mlp_layernorm
  │        ├─ MLP (dense 层) 或 MoELayer (MoE 层)
  │        │   ├─ TopKRouter
  │        │   ├─ GroupedMLP / TEGroupedMLP  (routed experts)
  │        │   └─ SharedExpertMLP             (shared experts)
  │        └─ bias_dropout_add
  └─ OutputLayer (ColumnParallelLinear)  (仅 post_process=True 的 PP rank)
```

## 3. 显存组成

总显存 = **静态显存** + **动态显存**

### 3.1 静态显存（权重 + 梯度 + 优化器状态）

每个参数元素占用的字节数取决于是否使用 Distributed Optimizer：

| 组件 | 不使用 DistOpt | 使用 DistOpt |
|------|--------------|-------------|
| FP16 权重 | 2 B | 2 B |
| FP16 梯度 | 2 B | 2 B |
| FP32 主权重 | 4 B | 4/DP B |
| FP32 动量 (Adam m) | 4 B | 4/DP B |
| FP32 方差 (Adam v) | 4 B | 4/DP B |
| **合计** | **18 B** | **6 + 12/DP B** |

其中 `DP = world_size / (TP × PP × CP)`。

**MoE 模型的特殊处理**：Expert 参数的 DP 组更小（因为 EP 将 expert 分配到不同 rank），公式变为：

```
bytes_per_param_dense = 6 + 12 / DP
bytes_per_param_moe   = 6 + 12 / (world_size / PP / EP / ETP)

静态显存 = (dense_params × bytes_dense + sparse_params × bytes_moe) / 2^30 GB
```

### 3.2 动态显存（激活）

激活量 = 前向传播中需保存用于反向的中间结果数量，乘以 2 字节 (FP16)。

各组件激活量计算：

| 组件 | 激活量（元素数） |
|------|----------------|
| **ColumnParallelLinear** | B × S × output_size_per_partition |
| **RowParallelLinear** | B × S × input_size_per_partition |
| **MLP (fc1+act+fc2)** | fc1 输出 + activation 中间值 |
| **SelfAttention** | QKV 投影 + attention score + 输出投影 |
| **MLASelfAttention** | 低秩压缩后的 Q/KV 激活（比标准 attention 小约 12 倍） |
| **MoELayer** | router 决策 + token dispatch (×topk) + expert 前向 + shared expert |
| **LayerNorm** | hidden_size（用于反向） |

**关键缩放因子**：

- **Sequence Parallel**: 当 SP 开启且 TP > 1 时，layernorm 激活除以 TP
- **Context Parallel**: 所有激活除以 CP
- **Pipeline Parallel**: 每个 PP rank 同时持有多个 microbatch 的激活

### 3.3 Pipeline 并行下的 microbatch 数量

每个 PP rank 同时在飞（in-flight）的 microbatch 数影响激活峰值：

| 场景 | in-flight microbatch 数 |
|------|----------------------|
| 无 VP | `pp_size - pp_rank` |
| VP chunk 0（首） | `pp_size + max((pp_size - pp_rank) × 2 - 1 - pp_size, 0)` |
| VP chunk 中间 | `pp_size` |
| VP chunk 末尾 | `min((pp_size - pp_rank) × 2 + 1, pp_size)` |

最终激活量：
```
num_activation = (per_layer_act × num_layers - loss_act) × num_microbatch + loss_act
```

## 4. 重计算（Recomputation）策略

重计算通过丢弃前向激活、在反向时重新计算来降低显存：

| 策略 | 说明 | 激活节省 |
|------|------|---------|
| `none` | 不重计算，保存所有激活 | 0% |
| `selective` | 按模块选择性重计算（layernorm, mlp, moe_act 等） | 中等 |
| `full` + `uniform` | 每 N 层重计算一次 | 高 |
| `full` + `block` | 整块重计算，仅保留 block 边界激活 | 最高 |

**Full + Block 方法的峰值分析**：

```python
common_act = embedding_act + between_layer_act × num_layers × num_microbatch

peak_forward = common_act + loss_act                          # loss 计算时
peak_backward = common_act + per_layer_act × recompute_layers # 反向重计算时

num_activation = max(peak_forward, peak_backward)
```

## 5. 并行维度对显存的影响

```
world_size = TP × PP × DP × CP    (dense)
world_size = TP × PP × EP × ETP   (expert，DP 隐含)
```

| 并行维度 | 切分对象 | 对参数量的影响 | 对激活的影响 |
|---------|---------|-------------|------------|
| **TP** | hidden_size（注意力头/FFN 列） | 参数 / TP | 激活 / TP（SP 开启时） |
| **PP** | Transformer 层 | 每 rank 仅持有部分层 | 受 microbatch 调度影响 |
| **EP** | MoE Expert 数量 | expert 参数 / EP | expert 激活不变（token 路由） |
| **ETP** | Expert 内部 FFN hidden | expert 参数 / ETP | expert 激活 / ETP |
| **CP** | 序列长度 | 无影响 | 激活 / CP |
| **DP** | 数据批次 | 无影响（DistOpt 下优化器 / DP） | 无影响 |
| **VP** | 层的虚拟分块 | 无影响 | 改变 microbatch 调度 |

## 6. MoE 层显存详解

### 6.1 参数量

```
routed_experts (GroupedMLP):
  fc1: num_local_experts × hidden_size × (moe_ffn_hidden_size × 2 / ETP)  # ×2 for GLU
  fc2: num_local_experts × (moe_ffn_hidden_size / ETP) × hidden_size
  
  其中 num_local_experts = num_moe_experts / EP

shared_experts (SharedExpertMLP):
  fc1: hidden_size × moe_shared_expert_intermediate_size × 2 / TP  # GLU
  fc2: moe_shared_expert_intermediate_size / TP × hidden_size

router (TopKRouter): 0（无可训练参数）
```

### 6.2 激活量

```
MoELayer 激活 =
    router: B × S × hidden_size × 2               # routing 决策
  + dispatcher: B × S × hidden_size × topk         # token 分发
  + experts: GroupedMLP 激活（按 token 均分到 expert）
  + shared_experts: SharedExpertMLP 激活（全部 token）
```

## 7. MLA（Multi-Latent Attention）显存优势

MLA 通过低秩压缩减少 KV cache 和激活：

```
标准 Attention:
  Q/K/V 各 [S, B, num_heads, head_dim]  →  激活 ∝ 3 × num_heads × head_dim

MLA:
  Q: hidden → q_lora_rank → num_heads × qk_head_dim
  KV: hidden → kv_lora_rank → num_heads × (qk_head_dim + v_head_dim)
  
  激活 ∝ q_lora_rank + kv_lora_rank  (远小于 3 × num_heads × head_dim)
```

典型参数：`q_lora_rank=1536, kv_lora_rank=512` vs 全展开 `128 × 128 = 16384`。

## 8. 输出报告字段说明

| 字段 | 含义 |
|------|------|
| `pp_rank` | Pipeline 并行 rank 编号 |
| `vp_chunk` | Virtual Pipeline 分块编号 |
| `num_parameter` | 该 rank 参数量 |
| `num_parameter_sparse` | 其中 MoE expert 参数量 |
| `weight_grad_memory_gb` | 权重+梯度显存 (GB) |
| `weight_grad_optim_memory_gb` | 权重+梯度+优化器显存 (GB) |
| `activation_memory_gb` | 激活显存 (GB) |
| `total_memory_gb` | 总显存 (GB) |
| `num_bytes_per_parameter` | 每参数字节数（含优化器） |
| `details` | 逐层参数/激活明细 |

## 9. 示例：DeepSeek-V3 (256 experts, EP=128, ETP=1, TP=8, PP=16)

```
每 EP rank 本地 expert 数 = 256 / 128 = 2
每 expert fc1 参数 = hidden_size × moe_ffn_hidden_size × 2 / ETP
                   = 7168 × 2048 × 2 / 1 = 29,360,128
每 expert fc2 参数 = moe_ffn_hidden_size / ETP × hidden_size
                   = 2048 / 1 × 7168 = 14,680,064

单 MoE 层 expert 参数 = 2 × (29,360,128 + 14,680,064) = 88,080,384 ≈ 84M

MoE DP size = world_size / PP / EP / ETP = 2048 / 16 / 128 / 1 = 1
bytes_per_param_moe = 6 + 12/1 = 18
```
