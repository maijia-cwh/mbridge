

# 参考脚本

```shell


  ┌──────────┬──────────────────────────────────────┬───────────────────────────────────────┐
  │   类别    │                 参数                 │                 说明                  │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤                                                                                                                                                                                                                   
  │ 并行策略  │ --tp, --pp, --ep, --cp, --etp, --vpp │ 全部并行维度                          │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤                                                                                                                                                                                                                   
  │ PP 分层   │ --first-pp-layers, --last-pp-layers  │ 首尾 stage 自定义层数                 │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤
  │          │ --pp-layout                          │ 完全自定义各 stage 层数，如 '4,8,8,4' │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤
  │          │ --account-embedding, --account-loss  │ 将 Embedding/Loss 计入 PP 切分        │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤
  │ 重计算    │ --recompute                          │ 模式：selective/full/none             │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤
  │          │ --recompute-method                   │ uniform/block                         │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤
  │          │ --recompute-num-layers               │ 重计算层数                            │
  ├──────────┼──────────────────────────────────────┼───────────────────────────────────────┤
  │          │ --recompute-modules                  │ 选择性重计算模块列表                  │
  └──────────┴──────────────────────────────────────┴───────────────────────────────────────┘

  # 使用示例：

  # MoE 模型，非均匀 PP 分层，首尾 stage 少放层
  bash memory_estimator/start.sh cli \
    --model Qwen/Qwen3-235B-A22B \
    --num-gpus 64 --tp 4 --pp 4 --ep 8 \
    --first-pp-layers 2 --last-pp-layers 2 \
    --account-embedding --account-loss

  # 完全自定义 PP layout + full recompute
  bash memory_estimator/start.sh cli \
    --model deepseek-ai/DeepSeek-V3 \
    --num-gpus 128 --tp 8 --pp 8 --ep 64 --etp 4 \
    --pp-layout '6,8,8,8,8,8,8,6' \
    --recompute full --recompute-method uniform --recompute-num-layers 4   
    
  # 完全自定义 PP layout + full recompute + cp
  bash memory_estimator/start.sh cli \
    --model memory_estimator/config/kimi-K2.5 \
    --num-gpus 640 --tp 8 --pp 5 --ep 128 --etp 1 --cp 16 \
    --pp-layout '10,13,13,13,12' \      
    --recompute full --recompute-method uniform --recompute-num-layers 1
    
```