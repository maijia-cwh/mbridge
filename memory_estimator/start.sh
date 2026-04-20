#!/bin/bash
# Memory Estimator 启动脚本
# 支持两种模式: webui (Web界面) 和 cli (命令行单次估算)

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Python 解释器 (需要 3.10+)
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.10}"

# Megatron-LM 路径 (请根据实际情况修改)
MEGATRON_PATH="${MEGATRON_PATH:-/Users/maijia/code/opensource/Megatron-LM}"

# 设置 PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}:${MEGATRON_PATH}:${PYTHONPATH}"

# ---------- 依赖检查 ----------
check_deps() {
    local mode="${1:-cli}"
    echo "检查依赖..."
    local imports="torch, megatron, mbridge"
    if [ "${mode}" = "webui" ]; then
        imports="fastapi, uvicorn, ${imports}"
    fi
    ${PYTHON} -c "import ${imports}" 2>/dev/null || {
        echo "缺少依赖，请先安装:"
        echo "  pip install torch transformers sentencepiece tokenizers mbridge"
        [ "${mode}" = "webui" ] && echo "  pip install fastapi 'uvicorn[standard]'"
        echo "  并确保 Megatron-LM 已克隆到: ${MEGATRON_PATH}"
        exit 1
    }
    echo "依赖检查通过"
}

# ---------- WebUI 模式 ----------
run_webui() {
    local host="${HOST:-0.0.0.0}"
    local port="${PORT:-8800}"
    echo "启动 WebUI: http://${host}:${port}"
    cd "${SCRIPT_DIR}"
    ${PYTHON} -m uvicorn webui.main:app --host "${host}" --port "${port}" --reload
}

# ---------- CLI 模式 ----------
run_cli() {
    # 默认值
    local model="${SCRIPT_DIR}/config/Qwen3-8B"
    local num_gpus=8
    local tp=1
    local pp=1
    local ep=1
    local cp=1
    local etp="None"
    local vpp="None"
    local seq_len=4096
    local mbs=1
    local use_distributed_optimizer="True"
    local recompute_granularity="selective"
    local recompute_method="uniform"
    local recompute_num_layers="1"
    local recompute_modules=""
    local first_pp_layers="None"
    local last_pp_layers="None"
    local pp_layout="None"
    local account_embedding="False"
    local account_loss="False"

    # 解析命名参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)            model="$2"; shift 2 ;;
            --num-gpus)         num_gpus="$2"; shift 2 ;;
            --tp)               tp="$2"; shift 2 ;;
            --pp)               pp="$2"; shift 2 ;;
            --ep)               ep="$2"; shift 2 ;;
            --cp)               cp="$2"; shift 2 ;;
            --etp)              etp="$2"; shift 2 ;;
            --vpp)              vpp="$2"; shift 2 ;;
            --seq-len)          seq_len="$2"; shift 2 ;;
            --mbs)              mbs="$2"; shift 2 ;;
            --no-dist-opt)      use_distributed_optimizer="False"; shift ;;
            --recompute)        recompute_granularity="$2"; shift 2 ;;
            --recompute-method) recompute_method="$2"; shift 2 ;;
            --recompute-num-layers) recompute_num_layers="$2"; shift 2 ;;
            --recompute-modules) recompute_modules="$2"; shift 2 ;;
            --first-pp-layers)  first_pp_layers="$2"; shift 2 ;;
            --last-pp-layers)   last_pp_layers="$2"; shift 2 ;;
            --pp-layout)        pp_layout="$2"; shift 2 ;;
            --account-embedding) account_embedding="True"; shift ;;
            --account-loss)     account_loss="True"; shift ;;
            *)                  echo "未知参数: $1"; exit 1 ;;
        esac
    done

    # 如果是本地路径，转为绝对路径
    if [ -d "${model}" ] || [ -f "${model}/config.json" ]; then
        model=$(cd "${model}" && pwd)
    fi

    echo "CLI 估算模式"
    echo "  模型: ${model}"
    echo "  GPU数: ${num_gpus}, TP: ${tp}, PP: ${pp}, EP: ${ep}, CP: ${cp}, ETP: ${etp}, VPP: ${vpp}"
    echo "  SEQ_LEN: ${seq_len}, MBS: ${mbs}, DistOpt: ${use_distributed_optimizer}"
    echo "  Recompute: ${recompute_granularity}, Method: ${recompute_method}, Layers: ${recompute_num_layers}"
    echo "  FirstPPLayers: ${first_pp_layers}, LastPPLayers: ${last_pp_layers}, PPLayout: ${pp_layout}"
    echo "---"

    # 构建需要引号包裹的字符串/列表参数
    local modules_arg="None"
    if [ -n "${recompute_modules}" ]; then
        modules_arg="[$(echo "${recompute_modules}" | sed "s/,/', '/g" | sed "s/^/'/" | sed "s/$/'/" )]"
    fi
    local pp_layout_arg="None"
    if [ "${pp_layout}" != "None" ]; then
        pp_layout_arg="'${pp_layout}'"
    fi

    cd "${SCRIPT_DIR}"
    ${PYTHON} -c "
import asyncio, json, sys
sys.path.insert(0, '.')
from main import MBridgeEstimateConfig, estimate_with_mbridge

config = MBridgeEstimateConfig(
    hf_model_path='${model}',
    num_gpus=${num_gpus},
    tp=${tp},
    pp=${pp},
    ep=${ep},
    cp=${cp},
    etp=${etp},
    vpp=${vpp},
    seq_len=${seq_len},
    mbs=${mbs},
    use_distributed_optimizer=${use_distributed_optimizer},
    recompute_granularity='${recompute_granularity}',
    recompute_method='${recompute_method}',
    recompute_num_layers=${recompute_num_layers},
    recompute_modules=${modules_arg},
    num_layers_in_first_pipeline_stage=${first_pp_layers},
    num_layers_in_last_pipeline_stage=${last_pp_layers},
    pipeline_model_parallel_layout=${pp_layout_arg},
    account_for_embedding_in_pipeline_split=${account_embedding},
    account_for_loss_in_pipeline_split=${account_loss},
)
result = asyncio.run(estimate_with_mbridge(config))
print(json.dumps(result, indent=2, ensure_ascii=False))
"
}

# ---------- 入口 ----------
usage() {
    echo "用法: $0 [模式] [参数...]"
    echo ""
    echo "模式:"
    echo "  webui          启动 Web 界面 (默认端口 8800)"
    echo "  cli [选项]     命令行单次估算"
    echo "  check          仅检查依赖"
    echo ""
    echo "CLI 参数:"
    echo ""
    echo " 并行策略:"
    echo "  --model <path>          模型路径或HF ID (默认: ./config/Qwen3-8B)"
    echo "  --num-gpus <N>          GPU数量 (默认: 8, 须为8的倍数)"
    echo "  --tp <N>                Tensor Parallel (默认: 1)"
    echo "  --pp <N>                Pipeline Parallel (默认: 1)"
    echo "  --ep <N>                Expert Parallel (默认: 1)"
    echo "  --cp <N>                Context Parallel (默认: 1)"
    echo "  --etp <N>               Expert Tensor Parallel (可选)"
    echo "  --vpp <N>               Virtual Pipeline Parallel stages (可选)"
    echo ""
    echo " Pipeline 分层:"
    echo "  --first-pp-layers <N>   第一个PP stage的层数 (可选)"
    echo "  --last-pp-layers <N>    最后一个PP stage的层数 (可选)"
    echo "  --pp-layout <layout>    自定义PP layout, 逗号分隔 (如: '4,8,8,4')"
    echo "  --account-embedding     将Embedding计入PP切分"
    echo "  --account-loss          将Loss计入PP切分"
    echo ""
    echo " 训练配置:"
    echo "  --seq-len <N>           序列长度 (默认: 4096)"
    echo "  --mbs <N>              Micro Batch Size (默认: 1)"
    echo "  --no-dist-opt           禁用 Distributed Optimizer"
    echo ""
    echo " 重计算 (Recomputation):"
    echo "  --recompute <mode>      重计算模式: selective|full|none (默认: selective)"
    echo "  --recompute-method <m>  重计算方法: uniform|block (默认: uniform)"
    echo "  --recompute-num-layers <N>  重计算层数 (默认: 1)"
    echo "  --recompute-modules <m> 选择性重计算模块, 逗号分隔 (如: 'mlp,attention')"
    echo ""
    echo "示例:"
    echo "  $0 webui"
    echo "  $0 cli --model ./config/Qwen3-8B --num-gpus 8 --tp 1 --pp 1"
    echo "  $0 cli --model Qwen/Qwen3-235B-A22B --num-gpus 64 --tp 4 --pp 4 --ep 8 --cp 2"
    echo "  $0 cli --model deepseek-ai/DeepSeek-V3 --num-gpus 128 --tp 8 --pp 16 --ep 64 --etp 4"
    echo "  $0 cli --model Qwen/Qwen3-235B-A22B --pp 4 --first-pp-layers 2 --last-pp-layers 2"
    echo "  $0 cli --model Qwen/Qwen3-235B-A22B --pp 4 --pp-layout '14,18,18,14' --account-embedding"
    echo ""
    echo "环境变量:"
    echo "  MEGATRON_PATH  Megatron-LM 目录 (默认: /Users/maijia/code/opensource/Megatron-LM)"
    echo "  HOST           WebUI 监听地址 (默认: 0.0.0.0)"
    echo "  PORT           WebUI 端口 (默认: 8800)"
}

MODE="${1:-webui}"
shift 2>/dev/null || true

case "${MODE}" in
    webui)
        check_deps webui
        run_webui
        ;;
    cli)
        check_deps cli
        run_cli "$@"
        ;;
    check)
        check_deps webui
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "未知模式: ${MODE}"
        usage
        exit 1
        ;;
esac
