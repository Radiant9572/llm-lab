# llm-lab

大模型学习实验仓库。

目标：用 26 周时间，从零实现到能跑通完整的后训练流水线，并留下可复现的实验记录。

**进度**：W1 完成中 → 26 周计划见 [`PLAN.md`](PLAN.md)

## 环境

| 项目 | 值 |
|---|---|
| Python 环境 | conda `llm`（`E:\Anaconda\envs\llm`，Python 3.11） |
| 核心依赖 | torch 2.14.0+cpu、transformers 5.17.0、datasets 5.0.1、accelerate 1.15.0 |
| 本机算力 | 无 NVIDIA 显卡（CPU 模式），训练用云端 GPU |

```bash
conda activate llm
```

### HuggingFace 镜像与缓存（重要）

**huggingface.co 在本机网络下不可达**（实测连接超时），必须走 hf-mirror 镜像，
否则 `from_pretrained` 会一直卡住直到超时。

| 变量 | 值 | 为什么 |
|---|---|---|
| `HF_ENDPOINT` | `https://hf-mirror.com` | 官方站不可达 |
| `HF_HOME` | `E:\hf_cache` | 默认缓存在 C 盘，而 C 盘只剩 24GB；一个 Qwen 7B 就 15GB |

两个变量已写入用户环境变量（永久生效）。脚本里另有保险：`common/hf_env.py`
在 `import transformers` **之前**调用 `setup()`，即使换到没配过环境变量的机器也能工作。

```python
import sys; sys.path.insert(0, "common")
from hf_env import setup
setup()                        # ← 必须在 import transformers 之前

import transformers            # 之后才能导入
```

云 GPU 机器上不用改代码：`setup()` 检测到不是 Windows 就跳过改缓存路径。

## 目录结构

```
llm-lab/
├── README.md              本文件：项目总览 + 进度
├── PLAN.md                26 周计划（带进度状态）
├── common/                公共工具
│   ├── env_check.py       环境自检（换机器/换环境后先跑这个）
│   └── hf_env.py          HuggingFace 镜像与缓存设置
├── docs/                  长期文档
│   ├── github_workflow.md 本仓库的记录规范（每天怎么提交）
│   └── day-YYYY-MM-DD.md  每日执行清单
├── journal/               学习日志，一天一个文件
├── papers/                论文精读笔记（每周 1 篇）
├── leetcode/              Hot 100 计划与题解
│   └── _TEMPLATE.py       题解模板（每道题 cp 一份）
├── week01/                手写 Transformer 基础
│   ├── attention.py       缩放点积注意力 + 因果掩码
│   ├── attention_blank.py 默写版（无提示）
│   ├── shape_drill.py     张量形状专项 10 题
│   ├── hf_quickstart.py   HuggingFace 官方 quickstart + 两个观察实验
│   ├── ATTENTION_CONCEPTS.md
│   └── PREREQUISITES.md
└── weekNN/                后续每周
```

## 运行方式

```bash
# 环境自检
python common/env_check.py

# HuggingFace 配置检查（打印镜像端点与实际缓存路径）
python common/hf_env.py

cd week01
python attention.py            # 注意力练习（含 9 项测试）
python shape_drill.py          # 形状专项 10 题
python hf_quickstart.py        # HuggingFace quickstart（首次下载约 600MB）
python attention_blank.py      # 默写版，不看提示重写
```

## 每周进度

| 周 | 日期 | 内容 | 状态 |
|---|---|---|---|
| W1 | 09.14 – 09.20 | 环境与基础；手写缩放点积注意力 | ✅ 提前完成 |
| W2 | 09.21 – 09.27 | 手写完整 GPT | ⬜ |
| W3 | 09.28 – 10.04 | 训练 10M 参数小模型 | ⬜ |
| W4 | 10.05 – 10.11 | 结构改造与消融实验 | ⬜ |
| W5–W9 | 10.12 – 11.15 | SFT 全链路 + 数据消融 | ⬜ |
| W10–W14 | 11.16 – 12.20 | 偏好对齐（RM / DPO）+ 评测 | ⬜ |
| W15–W19 | 12.21 – 01.24 | GRPO 数学推理 RL ★ | ⬜ |
| W20–W26 | 01.25 – 03.14 | 评测、作品集、简历、投递 | ⬜ |

## 实验记录约定

每个实验都必须回答四个问题，缺一不可：

1. **问题**：我想验证什么假设
2. **做法**：控制了哪些变量，改了什么
3. **结果**：量化指标，带基线对比
4. **归因**：结论是什么，边界在哪

没有量化结果的实验等于没做。

## 怎么记录

见 [`docs/github_workflow.md`](docs/github_workflow.md)。一句话版本：

```bash
cd /e/llm-lab
git add -A && git commit -m "feat(week01): 实现因果掩码" && git push
```

外加在 `journal/` 下写当天日志（≤ 10 行）。

**commit message 要带数字和结论**——面试官真的会翻。
