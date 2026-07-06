# ABSA · 方面级情感分析 Web 系统

> 北京科技大学 · 计算机与人工智能实践 · 第一周项目
>
> Aspect-Based Sentiment Analysis — 基于 PyTorch 的情感分析 Web 应用

---

## 📋 项目概述

本项目基于 [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch)，实现了一个完整的**方面级情感分析（ABSA）**Web 系统。用户输入句子和方面词（aspect），系统判断对该方面的情感是 **正面 / 负面 / 中性**。

### 核心功能

- 🔍 **单模型分析**：选择 LSTM 或 BERT_SPC 进行情感推理
- ⚡ **双模型对比**：LSTM 和 BERT_SPC 同时推理，结果并排对比
- 📊 **概率可视化**：三色概率条展示 Negative / Neutral / Positive 分布
- 🎯 **快捷例句**：4 个预设测试用例一键填入
- ⌨️ **键盘快捷键**：`Ctrl+Enter` 分析 · `Ctrl+Shift+Enter` 对比 · `Esc` 清空
- 📱 **响应式设计**：自适应桌面和移动端

---

## 🖥 在线演示

启动后在浏览器打开：**http://127.0.0.1:5000**

![Web Interface](screenshots/web-demo.png)

### 界面截图

| 单模型分析 | 双模型对比 |
|:---:|:---:|
| ![Single](screenshots/single.png) | ![Compare](screenshots/compare.png) |

---

## ⚙️ 环境配置

### 依赖

| 组件 | 版本 |
|------|------|
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| transformers | 5.13.0 |
| Flask | 3.1.3 |
| scikit-learn | 1.9.0 |

### 安装步骤

```bash
# 1. 安装 PyTorch (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. 安装依赖
pip install transformers flask flask-cors scikit-learn numpy

# 3. 下载 GloVe 词向量 (仅 LSTM 模型需要)
# 从 https://nlp.stanford.edu/data/glove.42B.300d.zip 下载
# 解压到项目根目录

# 4. (国内用户) 设置 HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com
```

---

## 🚀 快速启动

```bash
cd ABSA-PyTorch
set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com
python app.py
```

首次启动会自动加载 LSTM 和 BERT_SPC 模型（约需 10-15 秒），之后在浏览器打开 `http://127.0.0.1:5000`。

### 训练自己的模型

```bash
# 训练 LSTM
python train.py --model_name lstm --dataset laptop

# 训练 BERT_SPC
python train.py --model_name bert_spc --dataset laptop
```

模型权重保存在 `state_dict/` 目录。

---

## 📊 模型对比

| 指标 | LSTM + GloVe | BERT_SPC | BERT 提升 |
|------|:-----------:|:--------:|:---------:|
| **Accuracy** | 68.50% | **78.68%** | **+10.18** |
| **F1 (macro)** | 63.36% | **74.74%** | **+11.38** |
| 参数量 | ~0.9M | ~110M | ×122 |
| 训练时间 | ~3 min | ~10 min | ×3.3 |
| 推理速度 | 快 (~5ms) | 慢 (~50ms) | ×10 |

### 关键发现

BERT_SPC 能正确区分**同一句子中不同方面词**的情感：

```
句子: "The screen is great but the battery is poor."

方面词   | LSTM        | BERT_SPC    | 正确
--------|-------------|-------------|------
screen  | Positive ✅ | Positive ✅  | ✅
battery | Positive ❌ | Negative ✅  | BERT 胜
```

> LSTM 使用静态 GloVe 词向量，无法区分同一句子中的不同方面；BERT 通过 `[CLS] 句子 [SEP] 方面词 [SEP]` 的输入格式和 Segment Embedding，能感知当前分析的方面词。

---

## 🏗 系统架构

```
┌──────────────────────┐     HTTP POST      ┌───────────────────────┐
│  前端 (HTML/CSS/JS)   │ ──────────────────► │  Flask 后端 (app.py)   │
│                      │                    │                       │
│  · 句子输入           │ ◄────────────────── │  · /api/predict        │
│  · 方面词输入         │     JSON 响应       │  · /api/models         │
│  · 模型选择           │                    │  · 动态加载双模型       │
│  · 概率可视化         │                    │  · RESTful API         │
└──────────────────────┘                    └───────┬───────────────┘
                                                    │
                                     ┌──────────────┴──────────────┐
                                     │                             │
                               LSTM 模型                      BERT_SPC 模型
                          (GloVe 300d 词向量)            (bert-base-uncased)
                          (~0.9M 参数, 轻量)             (~110M 参数, 高准确率)
```

### API 接口

**`POST /api/predict`**

```json
// 请求
{
  "sentence": "The screen is great but the battery is poor.",
  "aspect": "screen",
  "model_type": "bert_spc"
}

// 响应
{
  "result": {
    "sentiment": "Positive",
    "probabilities": {
      "Negative": 0.1562,
      "Neutral": 0.0405,
      "Positive": 0.8033
    }
  }
}
```

**`GET /api/models`** — 返回可用模型列表

---

## 📁 项目结构

```
ABSA-PyTorch/
├── app.py                              # Flask 后端（Web 系统入口）
├── templates/
│   └── index.html                      # 前端页面（纯 HTML/CSS/JS）
├── train.py                            # 模型训练脚本
├── infer_example.py                    # 推理脚本模板
├── data_utils.py                       # 数据处理 & Tokenizer
├── state_dict/                         # 训练好的模型权重
│   ├── lstm_laptop_val_acc_0.685      # LSTM 最佳模型 (68.50%)
│   └── bert_spc_laptop_val_acc_0.7868 # BERT 最佳模型 (78.68%)
├── models/                             # 模型定义
│   ├── lstm.py                         # LSTM (25 行)
│   ├── bert_spc.py                     # BERT_SPC (已修复)
│   └── lcf_bert.py                     # LCF_BERT (已修复导入)
├── layers/
│   └── dynamic_rnn.py                  # 动态 LSTM 层 (已修复)
├── datasets/semeval14/                 # SemEval 2014 数据集
├── laptop_tokenizer.dat               # Tokenizer（自动生成）
├── 300_laptop_embedding_matrix.dat    # GloVe 词向量矩阵（自动生成）
├── README.md                           # 本文件
└── 实验记录.md                          # 详细实验记录
```

---

## 🐛 已知问题与解决

| # | 问题 | 解决方案 |
|---|------|----------|
| 1 | Python 3.14 无 tokenizers 预编译包 | 升级 transformers 5.13.0 |
| 2 | `pack_padded_sequence` 要求 CPU tensor | `dynamic_rnn.py` + `.cpu()` |
| 3 | transformers 5.x 导入路径变更 | 改为 `transformers.models.bert.modeling_bert` |
| 4 | BERT 输出迭代返回 key 而非 value | 改用 `outputs.pooler_output` |
| 5 | HuggingFace 国内连接超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| 6 | 浏览器缓存旧版页面 | `Cache-Control: no-cache` meta 标签 |
| 7 | Jinja2 模板缓存不更新 | `TEMPLATES_AUTO_RELOAD = True` |
| 8 | 端口被多进程占用 | 用 PowerShell 彻底清理 Python 进程 |

---

## 📚 参考资料

- [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) — 原始开源项目
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) (Pennington et al., 2014)
- [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/) — Laptop/Restaurant 数据集

---

## 📝 实验记录

详细训练日志、测试结果和问题排查见 [实验记录.md](./实验记录.md)。

---

## 📄 许可

本项目基于 MIT 许可的 [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) 开发。

---

> **作者**: Elysia11110925  
> **日期**: 2026年7月  
> **课程**: 北京科技大学 · 计算机与人工智能实践 · 暑期小学期
