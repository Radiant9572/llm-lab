# 论文精读

**每周 1 篇，26 周 26 篇。** 这是替代"论文发表"的东西——你没有论文，但你可以证明你读得懂、讲得清、能复现。

对应 `PLAN.md` 里「贯穿全程的日常」中的「论文精读」一项。

---

## 规则

| 项 | 要求 |
|---|---|
| 频率 | 每周 1 篇，周日之前完成 |
| 单篇时间 | **1.5–2 小时**，不要超过。读不完就是方法不对，不是不够努力 |
| 产出 | 一份 200–300 字的摘要，存成 `NN-短标题.md` |
| 硬要求 | **必须写「我的问题」这一栏** —— 写成"没什么问题"等于没读 |

**不许逐字读，也不许翻译。** 一篇顶会论文的有效信息集中在：摘要、引言最后一段、方法部分的图和公式、实验的主表、结论。剩下的（相关工作、详细推导、附录）是查阅用的，不是通读用的。

---

## 摘要模板

```markdown
# NN · 论文标题 (年份)

**一句话**：（它提出的方法是什么 —— 一句话，不要超过 40 字）

**解决什么问题**：（在这篇之前，大家怎么做？卡在哪？）

**核心方法**：（3–5 句说清关键机制。允许出现公式，但每个符号要能解释）

**为什么有效**：（作者给的解释 + 你自己的判断。两者要分开标）

**实验里最关键的证据**：（哪张表/哪个消融最能支持结论？为什么？）

**我的问题**：（读完还有哪里不懂 / 哪里觉得可疑 —— 这栏最重要）

**和我的项目的关联**：（W1–W26 的哪一个实验会用到它）
```

---

## 索引

| # | 周 | 论文 | 文件 | 状态 |
|---|---|---|---|---|
| 01 | W1 | Attention Is All You Need (2017) | [`01-attention-is-all-you-need.md`](01-attention-is-all-you-need.md) | ⬜ |
| 02 | W2 | 待定 | | ⬜ |
| 03 | W3 | 待定 | | ⬜ |

> 每周日选下一篇，选完把标题填进表里。**选的时候优先挑"当周代码要用到"的**，
> 这样读论文和写代码互相印证，效率最高。

---

## 候选池（按当周主题挑）

不全，也不按重要性排序。**标记 ★ 的是和你的求职方向直接相关、建议不要跳过的。**

**架构与基础**
- Attention Is All You Need (2017) ★
- Language Models are Unsupervised Multitask Learners / GPT-2 (2019)
- LLaMA: Open and Efficient Foundation Language Models (2023)
- RoFormer / RoPE 旋转位置编码 (2021) ★

**规模与数据（→ 数据方向）**
- Scaling Laws for Neural Language Models (2020) ★
- Training Compute-Optimal LLMs / Chinchilla (2022) ★
- DoReMi: Optimizing Data Mixtures (2023) ★
- Textbooks Are All You Need / phi 系列 (2023) ★
- Self-Instruct: Aligning LM with Self-Generated Instructions (2022) ★
- LIMA: Less Is More for Alignment (2023) ★

**后训练与对齐（→ 后训练方向）**
- Training language models to follow instructions / InstructGPT (2022) ★
- Proximal Policy Optimization / PPO (2017) ★
- Direct Preference Optimization / DPO (2023) ★
- Constitutional AI (2022)
- DeepSeekMath（GRPO 提出处）(2024) ★
- DeepSeek-R1: Incentivizing Reasoning Capability (2025) ★
- Tulu 3: Pushing Frontiers in Open Language Model Post-Training (2024)

**推理与评测（→ 评测 / LLM4Math 方向）**
- Chain-of-Thought Prompting (2022) ★
- Self-Consistency Improves Chain of Thought Reasoning (2022)
- GSM8K / Training Verifiers to Solve Math Word Problems (2021) ★
- Judging LLM-as-a-Judge / MT-Bench (2023) ★

**工程与效率（→ Infra 加分项，选读）**
- LoRA: Low-Rank Adaptation (2021) ★
- QLoRA: Efficient Finetuning of Quantized LLMs (2023) ★
- FlashAttention (2022)
- Efficient Memory Management for LLM Serving / PagedAttention & vLLM (2023)

---

*配套：`PLAN.md`（26 周计划）、`docs/github_workflow.md`（怎么提交）*
