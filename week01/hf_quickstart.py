"""
W1 · 验收项：跑通 HuggingFace 官方 quickstart
========================================================================

官方 quickstart（https://huggingface.co/docs/transformers/quicktour）只有三段：
    ① 用 pipeline 一行完成推理
    ② 拆开 pipeline，看里面到底发生了什么
    ③ 微调一个模型

这个文件把 ① ② 做完，并加了两个小实验。③（微调）留到 W5 的 SFT 阶段，
因为那本质上就是"训练循环 + Trainer"，属于 P2 的内容。

运行：
    cd E:\\llm-lab\\week01
    python hf_quickstart.py

首次运行会下载两个模型（约 600MB），走 hf-mirror 镜像，缓存在 E:\\hf_cache。
本次实测耗时 12 分钟（网络慢）。下载完再跑就很快了。

本次实测踩到的坑（已在 common/hf_env.py 里处理）：
  · huggingface.co 连接超时 —— 必须走镜像 HF_ENDPOINT=https://hf-mirror.com
  · Xet 存储后端（cas-bridge.xethub.hf.co）**不经过镜像**，下大文件时会
    "The read operation timed out"。它会自动续传，不致命，但会让下载时间翻倍。
    设 HF_HUB_DISABLE_XET=1 即可让全部流量走镜像。
========================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --- 必须在 import transformers 之前设置镜像和缓存路径 -----------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from hf_env import report as hf_report  # noqa: E402

hf_report()
print()

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

CLS_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
GEN_MODEL = "distilgpt2"

SENTENCES = [
    "This movie is absolutely wonderful and I loved every minute of it.",
    "The plot was dull and the acting felt completely lifeless.",
    "I'm not sure whether this was good or bad, honestly.",
]


def sep(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def new_part(text: str, prompt: str, width: int = 80) -> str:
    """取出生成出来的新内容，并把换行压成空格。

    不处理的话，生成结果里的换行会把终端刷出一大片空行 —— 看不到真实输出。
    """
    tail = text[len(prompt):] if text.startswith(prompt) else text
    return " ".join(tail.split())[:width] or "(空)"


# ==========================================================================
# 1 · pipeline：一行完成推理
# ==========================================================================
def part1_pipeline():
    sep("1 · pipeline —— 一行完成推理")

    clf = pipeline("sentiment-analysis", model=CLS_MODEL)
    print(f"模型：{CLS_MODEL}\n")

    t0 = time.time()
    results = clf(SENTENCES)
    dt = time.time() - t0

    for s, r in zip(SENTENCES, results):
        print(f"  [{r['label']:8s} {r['score']:.4f}]  {s[:60]}")
    print(f"\n3 条推理耗时 {dt:.2f}s（CPU）")

    print("\n观察：pipeline 帮你做了 tokenize → 前向 → 取 argmax → 转标签")
    print("      这四步。问题是你不知道中间发生了什么 —— 所以要拆开看看。")


# ==========================================================================
# 2 · 拆开 pipeline
# ==========================================================================
def part2_manual():
    sep("2 · 拆开 pipeline —— 手动走一遍")

    tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL)
    model.eval()

    text = SENTENCES[0]
    print(f"输入：{text}\n")

    # ---- 2.1 tokenize --------------------------------------------------
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print("【tokenizer 输出】")
    print(f"  keys           : {list(encoded.keys())}")
    print(f"  input_ids.shape: {tuple(input_ids.shape)}")
    print(f"  input_ids      : {input_ids.tolist()}")
    print(f"  attention_mask : {attention_mask.tolist()}")
    print(f"  词表大小       : {tokenizer.vocab_size}")

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"\n  前 12 个 token  : {tokens[:12]}")
    print(f"  token 0 是什么？ : {tokenizer.decode(input_ids[0, 0])!r}  ← 分类任务要的 [CLS] 位")

    # ---- 2.2 前向 ------------------------------------------------------
    print("\n【前向传播】")
    with torch.no_grad():
        outputs = model(**encoded)

    print(f"  outputs 的字段 : {[k for k in outputs.keys()]}")
    print(f"  logits.shape   : {tuple(outputs.logits.shape)}   ← (batch, 类别数)")
    print(f"  logits         : {outputs.logits.tolist()[0]}")

    # ---- 2.3 后处理 ----------------------------------------------------
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = int(probs.argmax())
    label = model.config.id2label[pred_id]

    print("\n【后处理】")
    print(f"  id2label       : {model.config.id2label}")
    print(f"  softmax 概率   : {[round(p, 4) for p in probs.tolist()]}")
    print(f"  argmax → 标签  : {pred_id} → {label}")
    print(f"  置信度         : {probs[pred_id]:.4f}")

    print("\n关键理解：**logits 是没归一化的分数，标签是 argmax 得到的。**")
    print("          softmax 只影响可读性，不影响 argmax 的结果。")

    return tokenizer, model


# ==========================================================================
# 3 · 两个观察实验
# ==========================================================================
def part3_experiments(tokenizer, model):
    sep("3 · 两个观察实验")

    # ---- 实验 A：单条 vs 批量，结果会变吗？ -----------------------------
    print("【实验 A】同样的句子，单条推理 vs 批量推理，结果一样吗？")
    print("          —— 考的是 padding 和 attention_mask 的作用\n")

    singles = []
    for s in SENTENCES:
        enc = tokenizer(s, return_tensors="pt")
        with torch.no_grad():
            singles.append(model(**enc).logits[0])
    single_logits = torch.stack(singles)

    batch_enc = tokenizer(SENTENCES, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        batch_logits = model(**batch_enc).logits

    print(f"  批量输入的 shape        : {tuple(batch_enc['input_ids'].shape)}")
    print(f"  attention_mask（注意 0）: {batch_enc['attention_mask'].tolist()}")
    print(f"  单条 vs 批量 最大绝对差 : {(single_logits - batch_logits).abs().max().item():.2e}")

    print("\n  结论：差异在浮点误差量级（~1e-6），可以认为**一致**。")
    print("        能一致的原因是 tokenizer 给了 attention_mask，")
    print("        模型知道哪些位置是补出来的 padding，不去关注它们。")

    # 故意不传 attention_mask，看会怎样
    with torch.no_grad():
        no_mask = model(input_ids=batch_enc["input_ids"]).logits
    diff_no_mask = (single_logits - no_mask).abs().max().item()
    print(f"\n  如果**不传 attention_mask**，最大绝对差 : {diff_no_mask:.2e}")
    if diff_no_mask > 1e-3:
        print("  → 差异明显变大。这就是 attention_mask 存在的原因：")
        print("    没有它，padding 位置的 [PAD] token 会被当成正常内容参与计算。")
    else:
        print("  → 差异仍很小（这个模型对 padding 不敏感），但**不要依赖这一点**：")
        print("    换成长文本、大 batch，或者自回归模型，影响会立刻显现。")

    # ---- 实验 B：encode → decode 是可逆的吗？ ---------------------------
    print("\n【实验 B】encode 再 decode，能拿回原文吗？\n")
    tests = [
        "Hello, World!",
        "I can't believe it's not butter!",
        "GPT-4 costs $20/month.",
    ]
    for t in tests:
        ids = tokenizer.encode(t)
        back = tokenizer.decode(ids)
        flag = "一致" if back == t else "不等于原文"
        print(f"  {t!r}")
        print(f"    → {len(ids):2d} 个 token → {back!r}   [{flag}]")

    print("\n  结论：**tokenize 不是可逆操作。** 看丢了什么：")
    print("        · 大写变小写（uncased 词表）· 空格被重新排版 · 加了 [CLS]/[SEP]")
    print("        所以做数据过滤时，不能拿 decode 出来的文本当原始文本用。")

    # ---- 实验 C：中文在英文词表下会被怎么切？ ---------------------------
    print("\n【实验 C】中文在英文词表下会发生什么？\n")
    unk_id = tokenizer.unk_token_id
    samples = [
        "北京大学数学科学学院",
        "注意力机制的权重之和恒为 1。",
        "模型不收敛，loss 变成 NaN，排查了三天。",
        "The model failed to converge and the loss went to NaN.",
    ]
    total_chars = total_unk = 0
    for t in samples:
        ids = tokenizer.encode(t)
        n_unk = sum(1 for i in ids if i == unk_id)
        total_chars += len(t)
        total_unk += n_unk
        print(f"  {t!r}")
        print(f"    字符 {len(t):3d}  →  token {len(ids):3d}  →  [UNK] {n_unk:2d}")
        print(f"    {tokenizer.convert_ids_to_tokens(ids)}")
        print()

    print("  结论（按上面实测数据判断，不要凭印象）：")
    if total_unk:
        print(f"    {total_chars} 个字符里有 {total_unk} 个变成了 [UNK] —— **信息直接丢了**。")
        print("    这不是'效果差一点'，是根本做不了。")
    else:
        print("    这个词表恰好收录了这些汉字，所以这一批没丢信息（1 字 ≈ 1 token）。")
        print("    **但这是巧合，不是通用结论。**")
    print("\n    真正可推广的观察是：**同样的语义，中文的 token 数通常远多于英文。**")
    print("    这直接决定训练成本、上下文长度上限和推理速度 —— 所以做数据配比时，")
    print("    '中文占比'不能只看条数，要看 token 数。（W8 的数据消融会用到这一点）")
    print("\n    行动项：中文任务必须选带中文词表的模型（Qwen / GLM / Baichuan 等）。")
    print("            这是 W5 选基座时的第一道筛子。")


# ==========================================================================
# 4 · 生成任务
# ==========================================================================
def part4_generate():
    sep("4 · 生成任务（自回归）")

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)
    model.eval()

    prompt = "The best way to learn deep learning is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"模型：{GEN_MODEL}")
    print(f"提示：{prompt!r}")
    print(f"token 数：{inputs['input_ids'].shape[1]}\n")

    # ---- 4.1 贪心解码 --------------------------------------------------
    with torch.no_grad():
        greedy = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    greedy_text = tokenizer.decode(greedy[0], skip_special_tokens=True)
    print("【贪心解码 do_sample=False】—— 每次结果都一样")
    print(f"  {new_part(greedy_text, prompt)}")

    # 再跑一次，验证"每次结果都一样"
    with torch.no_grad():
        again = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    same = tokenizer.decode(again[0], skip_special_tokens=True) == greedy_text
    print(f"  再跑一次，结果完全相同：{same}   ← 贪心是确定性的")

    # ---- 4.2 采样解码 --------------------------------------------------
    print("\n【采样解码 do_sample=True】—— 同一个提示连跑三次")
    torch.manual_seed(0)
    outputs = []
    for i in range(3):
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=30,
                do_sample=True, temperature=0.9, top_k=50, top_p=0.95,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(text)
        print(f"  第 {i + 1} 次: {new_part(text, prompt)}")
    print(f"\n  三次结果互不相同：{len(set(outputs)) == 3}   ← 采样是随机的")

    print("\n关键理解：")
    print("  · 模型每一步输出的其实是**下一个 token 的概率分布**，不是确定的词")
    print("  · 贪心 = 每步都取概率最大的（会重复、会呆板）")
    print("  · temperature 调低 → 更保守；调高 → 更随机（过高会胡言乱语）")
    print("  · top_k / top_p 决定从多少个候选里采样")
    print("\n  这些采样逻辑你会在 W3 自己手写一遍 —— 那时候它就是几行代码，")
    print("  而不是一个 black box 参数。")


# ==========================================================================
# 5 · 自测问题
# ==========================================================================
def quiz():
    sep("5 · 自测问题（答不上来的回看对应小节）")

    questions = [
        ("tokenizer 输出的 input_ids 和 attention_mask 分别是什么？（§2.1）", None),
        ("attention_mask 里的 0 代表什么？为什么不能省掉它？（§3 实验 A，实测差 7.4e-02）", None),
        ("pipeline 里发生的四件事是什么？（§1）", "tokenize → 前向 → argmax → 映射标签"),
        ("logits 和概率是什么关系？argmax 需要先做 softmax 吗？（§2.3）", None),
        ("为什么 tokenizer.decode(tokenizer.encode(x)) 可能 != x？（§3 实验 B）", None),
        ("实验 C 里，中文的 token 数是变多还是变少？这对训练成本意味着什么？（§3 实验 C）", None),
        ("如果换成一个生僻字的词表，实验 C 的 [UNK] 计数会怎样？为什么这是'信息丢失'而不是'效果变差'？（§3 实验 C）", None),
        ("贪心解码为什么是确定性的？采样解码为什么不是？（§4）", None),
        ("temperature 调高会发生什么？调低呢？（§4.2）", None),
        ("huggingface.co 连不上时会发生什么？为什么光设 HF_ENDPOINT 还不够？（文件开头）", None),
        ("如果要把这批代码搬到云 GPU 上，哪一行必须改？为什么？（文件开头）", None),
    ]
    for i, (q, hint) in enumerate(questions, 1):
        print(f"  {i}. {q}")
        if hint:
            print(f"     ↳ {hint}")
    print()
    print("  第 11 题的答案：`sys.path` 那一行和 hf_env 的缓存路径判断 ——")
    print("  Linux 上没有 E 盘，hf_env.setup() 会自动跳过改缓存位置。")
    print("  这就是为什么要把环境设置抽成一个模块，而不是硬编码在脚本里。")
    print()
    print("  第 10 题：Xet 后端的 blob 地址（cas-bridge.xethub.hf.co）**不经过镜像**，")
    print("  设了 HF_ENDPOINT 也拦不住它 —— 还要加 HF_HUB_DISABLE_XET=1。")


def main():
    part1_pipeline()
    tokenizer, model = part2_manual()
    part3_experiments(tokenizer, model)
    part4_generate()
    quiz()

    sep("完成")
    print("HuggingFace quickstart 跑通了。这是 W1 验收项的最后一项。")
    print()
    print("下一步（明天的 W2）：不看这个文件，从零实现一个迷你 GPT。")
    print("你会发现 §2.2 那一步 —— model(**encoded) —— 就是你要写的东西。")


if __name__ == "__main__":
    main()
