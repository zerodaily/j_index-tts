# j-index-tts 相比原版 index-tts 的修改

## 概述

本项目基于 [index-tts](https://github.com/index-tts/index-tts) 进行修改，主要优化了模型的本地加载机制，减少对网络下载的依赖。

## 主要变更

### 1. 本地模型加载支持

修改了 `indextts/infer_v2.py`，优先使用本地路径加载模型：

- **SeamlessM4TFeatureExtractor**: 支持从本地 `w2v_bert` 目录加载
- **Semantic Codec**: 支持从本地 `MaskGCT_model/semantic_codec/model.safetensors` 加载
- **Campplus Model**: 支持从本地 `campplus_cn_common.bin` 加载
- **BigVGAN Vocoder**: 支持从本地 `bigvgan_v2` 目录加载

如果本地路径不存在，回退到从 HuggingFace Hub 下载。

#### 代码修改详情 (infer_v2.py)

```python
# 修改前
self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# 修改后
w2v_bert_path = os.path.join(self.model_dir, "w2v_bert")
if os.path.exists(w2v_bert_path):
    self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_path)
else:
    self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
```

```python
# 修改前
semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")

# 修改后
local_semantic_codec = os.path.join(self.model_dir, "MaskGCT_model", "semantic_codec", "model.safetensors")
if os.path.exists(local_semantic_codec):
    semantic_code_ckpt = local_semantic_codec
else:
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
```

```python
# 修改前
campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")

# 修改后
local_campplus = os.path.join(self.model_dir, "campplus_cn_common.bin")
if os.path.exists(local_campplus):
    campplus_ckpt_path = local_campplus
else:
    campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
```

```python
# 修改前
self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)

# 修改后
local_bigvgan = os.path.join(self.model_dir, "bigvgan_v2")
if os.path.exists(local_bigvgan):
    self.bigvgan = bigvgan.BigVGAN.from_pretrained(local_bigvgan, use_cuda_kernel=self.use_cuda_kernel)
else:
    self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
```

### 2. 语义模型本地加载

修改了 `indextts/utils/maskgct_utils.py` 的 `build_semantic_model` 函数：

- 新增 `local_w2v_path` 参数，支持从本地加载 Wav2Vec2BertModel
- 自动查找本地模型路径，路径不存在时回退到 HuggingFace

#### 代码修改详情 (maskgct_utils.py)

```python
# 修改前
def build_semantic_model(path_='./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt'):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

# 修改后
def build_semantic_model(path_='./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt', local_w2v_path=None):
    if local_w2v_path is None:
        if path_.startswith('./'):
            local_w2v_path = "./w2v_bert"
        else:
            parent_dir = os.path.dirname(os.path.dirname(path_))
            local_w2v_path = os.path.join(parent_dir, "w2v_bert")

    if os.path.exists(local_w2v_path):
        semantic_model = Wav2Vec2BertModel.from_pretrained(local_w2v_path)
    else:
        semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
```

## 文件变更

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `indextts/infer_v2.py` | 修改 | 添加本地模型加载逻辑 |
| `indextts/utils/maskgct_utils.py` | 修改 | 支持本地 w2v-bert 模型 |
| `.gitignore` | 新增 | 忽略 `.log` 文件 |
| `README_DEPLOY.md` | 新增 | 部署文档 |
| `run_optimized.sh` | 新增 | 优化启动脚本 |
| `test_load.py` | 新增 | 模型加载测试脚本 |
| `checkpoints/` | 新增 | 本地模型目录 |
| `checkpoints/config.yaml` | 删除 | 不再需要 |
| `checkpoints/pinyin.vocab` | 删除 | 不再需要 |

## 使用方式

### 本地模型目录结构

```
checkpoints/
├── w2v_bert/                    # SeamlessM4TFeatureExtractor 本地模型
├── MaskGCT_model/
│   └── semantic_codec/
│       └── model.safetensors    # Semantic Codec 本地模型
├── campplus_cn_common.bin       # Campplus 本地模型
└── bigvgan_v2/                  # BigVGAN 本地模型
```

### 启动命令

```bash
bash run_optimized.sh
```

## 优势

1. **离线可用**: 模型本地缓存后，无需网络即可运行
2. **启动加速**: 避免重复下载大模型文件
3. **更稳定**: 不受 HuggingFace Hub 服务质量影响
