# 01 · Attention Is All You Need —— 阅读导览

> 配套文件：`01-attention-is-all-you-need.md`（读完写摘要）
> 原文 PDF：`C:\Users\蒙蒙\Desktop\note\NIPS-2017-attention-is-all-you-need-Paper.pdf`
> 中英对照：`D:\pdf2zh\output\attention-is-all-you-need.zh.dual.pdf`
> 纯中文版：同目录 `.zh.mono.pdf`　术语表：`.zh.glossary.csv`

---

## 0 · 结论：这篇是你最容易读的第一篇英文论文

不是安慰你，有四条具体理由：

| 理由 | 说明 |
|---|---|
| **内容你已经会了** | 你手写过 softmax、缩放点积、因果掩码。你不是在读新知识，是在看"同一件事的英文怎么说" |
| **只有 1 个公式** | 就是 `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`，你实现过 |
| **只有 4 个小节要精读** | §3.1 §3.2 §3.2.1 §3.5，总共约 5 页 |
| **结构是模板化的** | 摘要 → 引言 → 背景 → 模型 → 实验 → 结论。所有 Transformer 论文都长这样 |

**你卡住的不是内容，是"内容的英文说法"。** 这两件事差别很大——前者要几周，后者只要一次。

---

## 1 · 中英对照怎么用（方法比工具重要）

⚠️ **最大的陷阱：顺着中文读下去。**

如果你从左边中文版开始读，你会一路读完中文，英文一眼都不看。90 分钟后你"读完了论文"，但英文阅读能力增长为零。26 周后你还是读不了。

**正确的用法：中文是校验，不是正文。**

```
每一段：
  1. 先读英文（哪怕磕磕绊绊）
  2. 在心里用自己的话归纳一句："这段在说 X"
  3. 再看中文，确认猜得对不对
  4. 猜错了 → 回去看是哪几个词没认出来，圈出来
```

**唯一允许的例外**：连续两段完全读不下去（比如 §2 背景那种满是引用的段落）→ 允许先中文后英文。**但那是例外，不是常态。**

**判断标准**：读完这篇后，随便指一段英文，你能说出它在讲什么。做不到就是方法用错了。

---

## 2 · 逐节导航

| 节 | 标题 | 读法 | 时间 | 读完要能回答 |
|---|---|---|---|---|
| 摘要 | Abstract | 慢读，只 5 句 | 5 min | 他们的方法一句话是什么 |
| §1 | Introduction | 读最后两段就够 | 5 min | 为什么 RNN 不行 |
| §2 | Background | **快扫**，只找"self-attention 跟以前有什么不同" | 5 min | 别人做过什么 |
| **§3.1** | Encoder and Decoder Stacks | **精读**，对着 Figure 1 | 15 min | 编码器和解码器各由什么堆成 |
| **§3.2** | Attention | **精读，全文最重要** | 25 min | 为什么要除 √d_k（两行推导） |
| **§3.2.1** | Multi-Head Attention | **精读** | 10 min | 多头跟单头差在哪 |
| §3.2.2 | Applications of Attention | 略读 | 5 min | 三种用法分别是什么 |
| §3.3 / §3.4 | FFN / Embeddings | 读 §3.4 最后一段 | 5 min | 位置编码为什么必要 |
| **§3.5** | Positional Encoding | **精读** | 10 min | 公式里 sin/cos 为什么要成对 |
| §4 | Why Self-Attention | 只看 Table 1 + 结论段 | 10 min | 复杂度 O(n²d) 怎么来的 |
| §5–§7 | Training / Results / Conclusion | 快扫 | 10 min | 它当时打败了谁 |

**总计约 105 分钟。** 读的时候开计时器——超时了就往下走，不要卡在某一节。

---

## 3 · 六个关键句（读到那里时对照看）

这六句分别对应你已经写过的代码。**不是帮你翻译全文，是给你六个锚点。**

**① §3.2，缩放的理由**

> "We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."

→ "我们**怀疑**当 d_k 很大时，点积的值会变大，把 softmax 推到梯度极小的区域。"
**注意这个词：suspect。** 论文这里只是猜测，没有实验。而你这周做了一张表（d_k = 1024 时 Jacobian 掉到 0.0001）——**你用一晚上给出了它没有的实证**。记住这个感觉。

**② §3.2，两行证明**

> "assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q·k = Σq_i·k_i, has mean 0 and variance d_k."

→ 假设 q 和 k 的分量是均值 0、方差 1 的独立随机变量，那么点积的方差是 d_k。
**这是纯概率论**，两行就证完了 —— 你的主场。整个"除以 √d_k"的理由就是这句话。

**③ §3.2.1，多头是什么**

> "Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections..."

→ 不是用 d_model 维做一次注意力，而是用 h 组不同的线性投影各做一次。
**翻译成代码：把 `scaled_dot_product_attention` 复制 h 份，每份用自己的一套 W_Q/W_K/W_V。** 你下周要写的 Multi-Head，就是这一句话。

**④ §3.2.3，Q/K/V 从哪来**

> "In the encoder self-attention layers, all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder."

→ 编码器的自注意力里，K、V、Q 都来自同一个地方：上一层的输出。
**这就是你写的"同一个 X 乘三个不同矩阵得到 Q、K、V"。**

**⑤ §3.4，为什么需要位置编码**

> "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence."

→ 因为没有循环也没有卷积，必须**人为注入**位置信息。
**这是注意力的软肋**：它对词序完全不敏感，"我打你"和"你打我"在它眼里一样。位置编码就是补丁。

**⑥ §4，为什么能替代 RNN**

> "Self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d..."

→ 当序列长度 n 小于表示维度 d 时，自注意力比循环层更快。
**它没有说"我们更好"，它给了一个条件。** 这就是好的论文写法——结论带边界。你写实验报告要学这个。

---

## 4 · 学术表达速查（论文里反复出现的句式）

| 英文 | 中文 | 出现场景 |
|---|---|---|
| We propose / We introduce | 我们提出 | 摘要、引言，宣告贡献 |
| as shown in Figure N / Table N | 如图 N 所示 | 几乎每节都有，**读到时必须回去看图** |
| we found it beneficial to | 我们发现……更有效 | 软化的实验结论，等于"我们试出来这样更好" |
| we suspect that | 我们怀疑 | **没有实验支撑的猜测** —— 和上一行要分清 |
| respectively | 分别 | 前后两个列表一一对应，读乱了对不上 |
| in this work / in this paper | 本文 | 划范围的 |
| state of the art (SOTA) | 当时最好 | 表格里的对比基线 |
| we leave ... to future work | 留待未来工作 | 论文承认的局限 |
| ablation | 消融实验 | **你的 W8 要做的就是这个词** |
| hyperparameters | 超参数 | 附录里那张表 |

**两条实用技巧**：

1. **遇到长句先找主谓宾，把定语从句括起来先不看。** 论文的句子长是因为塞了定语，主干通常很短。
2. **看见 "respectively" 就回头数一遍顺序。** 这个词出现的地方就是最容易读错的地方。

---

## 5 · 术语锚点表（术语 ↔ 你的代码）

**这张表是你现在最缺的东西。** 左边是论文英文，中间是你已经写过的代码，右边是中文。

| 论文里的英文 | 你写过的代码 | 中文 |
|---|---|---|
| attention | `scaled_dot_product_attention` | 注意力 |
| query / key / value | `Q` / `K` / `V` | 查询 / 键 / 值 |
| scaled dot-product attention | `(Q @ K.mT) / math.sqrt(d_k)` | 缩放点积注意力 |
| softmax | `stable_softmax` | 归一化指数函数 |
| mask / masking | `masked_fill(~mask, -inf)` | 掩码 |
| causal / autoregressive | `make_causal_mask` | 因果的 / 自回归的 |
| dot product | `Q @ K.transpose(-1, -2)` | 点积 |
| projection | `X @ W_Q` 这类矩阵乘 | 投影 |
| linear / linear layer | `nn.Linear` | 线性层 |
| embedding | 第 0 层查表 | 嵌入 |
| position encoding | 待 W2 实现 | 位置编码 |
| residual connection | 待 W2 实现 | 残差连接 |
| layer normalization | 待 W2 实现 | 层归一化 |
| feed-forward network (FFN) | 待 W2 实现 | 前馈网络 |
| head | 待 W2 实现 | 头 |
| hyperparameter | — | 超参数 |
| ablation study | — | 消融研究 |
| baseline | — | 基线 |
| tokens | `input_ids` 里的元素 | 词元 |

**读的时候把这张表放在旁边。** 看到右边的中文术语，立刻回想到左列的代码——这样你读的不是英文，是"你已经写过的三行代码的英文名"。

---

## 6 · 读完必须回答的三个问题

写在 `01-attention-is-all-you-need.md` 里，**每个答案要有原文依据**（哪一节、哪句话）。

1. **这篇论文要解决什么问题？在它之前大家怎么做？**
   （提示：去看 §1 的最后两段）

2. **它的核心改动是什么？用一句话说。**
   （不能用"提出了 Transformer"这种回答，那等于没说）

3. **为什么这个改动有效？作者给了什么理由，这个理由有没有实验支撑？**
   （提示：§3.2 的缩放推导 vs §4 的复杂度分析，是两种不同类型的论证）

然后加上第 4 栏 **「我的问题」** —— 读完哪里还不懂、哪里觉得可疑。**这一栏写"没有"就是没读。**

---

## 7 · 时间预算

**总共 105 分钟。** 建议这样拆：

```
今晚 / 明早    90 min   按 §2 导航读，边读边写摘要草稿
休息 10 min
接着           15 min   填 papers/01-attention-is-all-you-need.md
```

**如果读不完就停**：把 §3.2 和 §3.2.1 读完，其余快扫。**这两节是核心，其他都可以补。**

---

## 8 · 一个必须说清楚的事

**英文阅读不是可选项，是这项工作的入场券。** 往后看：

| 场景 | 语言 |
|---|---|
| 26 周里的 26 篇论文 | 英文 |
| 你要用的框架文档（veRL、TRL、LLaMA-Factory） | 英文 |
| GitHub 上的 issue 和 PR 讨论 | 英文 |
| 面试被问"读过什么论文" | 你得能讲出它的贡献 |
| 王老师组里具身智能 / 世界模型的前沿 | 全在 arXiv 上 |

**所以：今天用对照版把门槛降下来是对的，但方法必须是"英文在前、中文校验"。**

26 篇论文，就是你练这件事的地方。**第 1 篇最难，第 10 篇你会觉得轻松，第 26 篇你大概只需要中文做关键词确认。**

今天的目标不是"读懂论文"，是**建立"我能读下来"的信心**。这个信心比任何一个术语都值钱。

---

*读完后把摘要发我，我帮你看有没有漏掉关键点。*
