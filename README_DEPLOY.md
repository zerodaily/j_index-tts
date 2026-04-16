# IndexTTS2 本地部署说明 (22GB 显存优化版)

## 部署状态: ✅ 成功

### 显存使用情况
- 加载后显存占用: **~8.5 GB** (FP16 模式)
- 可用显存: **~14 GB**
- 优化效果: 22GB 显存完全够用

### 模型文件位置
- 主模型目录: `/mnt/e/ai-models/IndexTTS2` (符号链接到 `checkpoints/`)
- 本地模型缓存: `checkpoints/w2v_bert/` - facebook/w2v-bert-2.0
- 本地模型缓存: `checkpoints/MaskGCT_model/` - MaskGCT 模型
- 本地模型缓存: `checkpoints/bigvgan_v2/` - BigVGAN 声码器
- 本地模型缓存: `checkpoints/campplus_cn_common.bin` - CAMPPlus 模型

## 启动方式

### 优化启动脚本 (推荐)
```bash
cd /home/zerodaily/projects/index-tts
./run_optimized.sh
```

### 手动启动
```bash
cd /home/zerodaily/projects/index-tts
export HF_HOME="/mnt/e/ai-models"
export HF_ENDPOINT="https://hf-mirror.com"
PYTHONPATH="$PWD:$PYTHONPATH" uv run webui.py --fp16 --port 7860
```

### 访问地址
- 本地访问: http://127.0.0.1:7860
- 局域网访问: http://<你的IP>:7860

## 22GB 显存优化说明

### 启用的优化
1. **FP16 半精度推理** (`--fp16`)
   - 显存占用减半
   - 速度提升约 30-50%
   - 质量损失可忽略

2. **禁用 DeepSpeed** - DeepSpeed 会额外占用 2-4GB 显存

3. **禁用 CUDA Kernel** (`--cuda_kernel=False`) - BigVGAN CUDA 内核会占用额外显存

### 不推荐的选项
- `--deepspeed`: 会导致显存不足
- `--cuda_kernel`: 会导致显存不足

## 修改的文件

为了支持离线加载和 22GB 显存优化，修改了以下文件:

1. `indextts/infer_v2.py`
   - 添加本地路径加载支持 (w2v_bert, semantic_codec, campplus, BigVGAN)
   - 修改为优先使用本地模型文件

2. `indextts/utils/maskgct_utils.py`
   - 修改 `build_semantic_model()` 支持本地 w2v-bert-2.0 模型

3. `run_optimized.sh` - 优化启动脚本

## 已知问题

### 示例音频文件未下载
`examples/` 目录下的音频文件是 Git LFS 指针，需要运行:
```bash
git lfs pull
```
如果您没有安装 git-lfs，可以:
1. 安装 git-lfs: `sudo apt-get install git-lfs && git lfs install`
2. 或者在 WebUI 中直接上传您自己的参考音频

## 网络问题解决

如果遇到网络超时错误:
1. 设置 HF_ENDPOINT 环境变量使用镜像
2. 预先下载模型到 `/mnt/e/ai-models/` 目录
3. 模型会优先从本地路径加载

## 性能测试

| 模式 | 显存占用 | 推荐场景 |
|------|----------|----------|
| FP16 (当前) | ~8.5 GB | 22GB 显存推荐 |
| FP32 | ~15 GB | 24GB+ 显存 |
| FP16 + DeepSpeed | ~12 GB | 需要更多优化时 |

## 常用命令

```bash
# 查看显存使用
nvidia-smi

# 查看 WebUI 日志
tail -f webui.log

# 停止 WebUI
pkill -f "webui.py"

# 重新启动
./run_optimized.sh
```
