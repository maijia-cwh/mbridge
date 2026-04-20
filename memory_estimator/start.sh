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
    local model="${1:-${SCRIPT_DIR}/config/Qwen3-8B}"
    # 如果是本地路径，转为绝对路径
    if [ -d "${model}" ] || [ -f "${model}/config.json" ]; then
        model=$(cd "${model}" && pwd)
    fi
    local num_gpus="${2:-8}"
    local tp="${3:-1}"
    local pp="${4:-1}"
    local seq_len="${5:-4096}"
    local mbs="${6:-1}"

    echo "CLI 估算模式"
    echo "  模型: ${model}"
    echo "  GPU数: ${num_gpus}, TP: ${tp}, PP: ${pp}, SEQ_LEN: ${seq_len}, MBS: ${mbs}"
    echo "---"

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
    seq_len=${seq_len},
    mbs=${mbs},
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
    echo "  $0 cli <model_path> <num_gpus> <tp> <pp> <seq_len> <mbs>"
    echo ""
    echo "示例:"
    echo "  $0 webui"
    echo "  $0 cli ./config/Qwen3-8B 8 1 1 4096 1"
    echo "  $0 cli Qwen/Qwen3-8B 16 2 2 8192 2"
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
