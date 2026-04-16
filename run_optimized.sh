#!/bin/bash
# IndexTTS2 优化启动脚本 - 针对 22GB 显存 (RTX 2080 Ti)
#
# 优化说明:
# 1. --fp16: 使用半精度(FP16)推理，显存减半，速度更快
# 2. 不启用 --deepspeed: DeepSpeed 会占用更多显存
# 3. 不启用 --cuda_kernel: BigVGAN CUDA内核会占用额外显存
# 4. 梯度清零和内存优化已在代码中自动处理

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置模型目录 - 使用共享目录
export HF_HOME="/mnt/e/ai-models"
export HF_ENDPOINT="https://hf-mirror.com"
export MODELSCOPE_CACHE="/mnt/e/ai-models/ms_cache"
export TRANSFORMERS_CACHE="/mnt/e/ai-models/transformers_cache"
export HF_DATASETS_CACHE="/mnt/e/ai-models/datasets_cache"

# 设置 Python 路径
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "======================================"
echo "IndexTTS2 启动中 (22GB VRAM 优化模式)"
echo "======================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "模型目录: $SCRIPT_DIR/checkpoints -> $(readlink -f $SCRIPT_DIR/checkpoints)"
echo "HF_HOME: $HF_HOME"
echo "模式: FP16 半精度推理"
echo "======================================"

# 使用 FP16 优化，禁用 DeepSpeed 和 CUDA kernel
exec uv run webui.py \
    --fp16 \
    --port 7860 \
    --host 0.0.0.0 \
    --model_dir ./checkpoints
