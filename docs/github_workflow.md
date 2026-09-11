# GitHub 学习记录工作流

> 一次性配置约 40 分钟；之后每天 3 分钟。
> 建立于 2026-09-11

---

## 0 · 为什么要做这件事

不是"存档"。面试官点开你的 GitHub，只看三件事：

| 他看到什么 | 他得出的结论 |
|---|---|
| 提交记录连续 20 周没断 | **这人能持续投入**（这是招聘方最缺、也最难验证的品质） |
| README 上一行命令能跑出结果 | 项目是真的，不是抄的 |
| commit message 写的是"为什么"和数字 | 他理解自己在做什么，不是照教程敲 |

反过来：**一个三个月只有 3 次 commit、每次叫 `update` 的仓库，比没有仓库更糟。** 它同时证明了"你确实做过"和"你并不认真"。

对一个零实习、无论文的候选人来说，这个仓库不是加分项，是**替代品**——它替代的就是"实习经历"和"论文"本该证明的东西。

---

## 1 · 一次性配置（约 40 分钟）

### 1.0 先装 Git for Windows（本机还没有）

**2026-09-11 实测：这台机器上没有安装 Git。** 系统 PATH 里没有 `git`，`C:\Program Files\Git` 也不存在。
（WorkBuddy 内部带了一个 PortableGit，但它在 `.workbuddy\binaries\` 私有目录里，不在 PATH 上，
且随 WorkBuddy 升级可能换路径 —— 不要依赖它。）

**没有它，后面所有步骤都跑不了**：`git` 命令会报"不是内部或外部命令"，
VS Code 左侧的源代码管理面板也会是灰的。

1. 装包已下载到：**`E:\downloads\Git-2.55.0.5-64-bit.exe`**（62 MB，走 npmmirror 镜像）
2. 双击运行。安装向导里**其他全默认，只有一个地方要确认**：

   > 到 **"Adjusting your PATH environment"** 这一步时，
   > 保持选中 **`Git from the command line and also from 3rd-party software`**（默认就是这个）。
   > 这一项决定 `git` 能不能在 VS Code 终端里直接用。

3. 装完**关掉并重开 VS Code**（新终端才会读到新的 PATH）
4. 验证：

```bash
git --version
```

应该输出 `git version 2.55.0.windows.5`。

> 如果下载的安装包丢了，从 npmmirror 重新下（国内速度最快）：
> `https://registry.npmmirror.com/-/binary/git-for-windows/v2.55.0.windows.5/Git-2.55.0.5-64-bit.exe`

### 1.1 注册账号

https://github.com/signup

- 用户名建议 `radiant` + 数字/拼音，**别用中二网名**——它会出现在简历链接里
- 邮箱用你长期不换的那个（后面 git 身份要填同一个）
- 注册完去 Settings → Emails，把 **Private email** 记下来，形如 `12345678+用户名@users.noreply.github.com`。用它可以隐藏真实邮箱但照样计入贡献图

### 1.2 配置 git 身份

**这一步错了，贡献图上不会显示你的提交。** 姓名和邮箱必须和 GitHub 账号一致。

```bash
git config --global user.name "Radiant"
git config --global user.email "你的GitHub注册邮箱"

# 中文文件名在 git status 里正常显示，而不是显示成 \346\226\207\344\273\266
git config --global core.quotepath false

# 配合仓库里的 .gitattributes，统一用 LF，避免整文件 diff 噪音
git config --global core.autocrlf false

git config --global init.defaultBranch main
```

验证：

```bash
git config --global --list
```

### 1.3 本地初始化

仓库已经在 `E:\llm-lab`（`git init` 已完成，文件已暂存）。执行：

```bash
cd /e/llm-lab
git commit -m "chore: 初始化仓库，W1 手写缩放点积注意力"
```

### 1.4 建远程仓库

GitHub 网页右上角 `+` → **New repository**

| 选项 | 填什么 | 为什么 |
|---|---|---|
| Repository name | `llm-lab` | 和本地目录一致 |
| Visibility | **Public** | 私有仓库面试官看不到，等于没做 |
| Add a README file | **不勾** | 本地已有，勾了推送会冲突 |
| Add .gitignore | **None** | 本地已有 |
| Choose a license | 可选 MIT | 加上显得专业，且不妨碍别人看 |

### 1.5 关联并推送

```bash
cd /e/llm-lab
git remote add origin https://github.com/<你的用户名>/llm-lab.git
git push -u origin main
```

第一次 push 会弹出浏览器让你登录 GitHub（Windows 的 Git 凭据管理器），**登录一次之后就不用再管了**。

> **不要**把密码或 token 写进任何文件、任何命令。凭据管理器会替你记住。

### 1.6 验证

刷新仓库网页 → 应该能看到 `README.md`、`week01/`、`journal/` 等目录。点开 `week01/attention.py` 确认能看到代码。

**顺便看一眼你的用户名旁边有没有绿色小方块。** 从今天起，这个方块群就是你 26 周的脸面。

---

## 2 · 每天怎么记（3 分钟）

做完当天的任务，执行固定三连：

```bash
cd /e/llm-lab
git add -A
git commit -m "feat(week01): 默写注意力实现，9/9 通过"
git push
```

再加一个新文件：`journal/2026-09-11.md`（模板见 `journal/README.md`）。

**三个层次分工，别混：**

| 层次 | 文件 | 写什么 | 频率 | 篇幅 |
|---|---|---|---|---|
| 改动 | commit message | 这次改了什么、为什么 | 每次有意义的改动 | 一行 |
| 日志 | `journal/YYYY-MM-DD.md` | 今天干了什么、卡在哪、明天做什么 | 每天 | ≤ 10 行 |
| 实验 | `weekNN/EXPERIMENTS.md` | 问题 → 做法 → 量化结果 → 归因 | 每个实验 | 写全 |

**日报写短，实验报告写全。** 日报写长了是在消耗做实验的时间，收益为零。

---

## 3 · commit message 怎么写

格式：`类型(范围): 一句话说做了什么`

| 类型 | 用于 | 例子 |
|---|---|---|
| `feat` | 新代码、新实现 | `feat(week01): 实现因果掩码，支持广播` |
| `exp` | 实验（**必须带数字**） | `exp(week08): 数据量 1k/3k/10k 消融，3k 后收益消失` |
| `fix` | 修 bug | `fix(week03): mask 整行为 -inf 导致 loss 变 nan` |
| `docs` | 文档、笔记、论文摘要 | `docs(paper): 精读 Attention Is All You Need` |
| `chore` | 环境、依赖、配置 | `chore: 切换到 USTC 镜像源` |

**好的：**
```
feat(week01): 手写缩放点积注意力 + 因果掩码，9/9 测试通过
exp(week08): 合成数据占比 0/30/70%，70% 时评测集过拟合
fix(week03): 修复 d_k 缩放遗漏导致的 logits 饱和
docs: 补全日报模板与实验记录规范
```

**差的（永远不要写）：**
```
update
修改
提交
今天学了很多东西
完成作业
```

**为什么值得较真**：面试官真的会翻 commit 记录。一串带数字和结论的 commit，是你"做实验的人"这个身份的直接证据。而它的成本接近于零——你本来就要提交。

---

## 4 · 目录约定

```
llm-lab/
├── README.md              项目总览 + 进度表（面试官第一眼看这里）
├── PLAN.md                26 周计划（含进度状态）
├── .gitignore             大文件/密钥排除
├── .gitattributes         行尾统一
├── common/                公共工具（env_check.py 等）
├── journal/               每天一条 → 学习日志
│   ├── README.md          模板与写法
│   └── YYYY-MM-DD.md
├── papers/                论文精读笔记（每周 1 篇，26 周 26 篇）
├── weekNN/                每周的代码与实验
│   ├── EXPERIMENTS.md     实验记录（问题→做法→结果→归因）
│   └── xxx.py
├── leetcode/              Hot 100 计划与题解
└── docs/                  长期文档（本文件、学习地图等）
```

**命名规则**：英文小写 + 下划线，不要空格、不要中文文件名。

> 中文文件名的坑：Linux 服务器上是 UTF-8，Windows 默认 GBK，`rsync` / `tar` / `git diff` 在两者之间反复出乱码。`leetcode/` 里那两个中文名 .md 是历史遗留，**新文件一律用英文名**。

---

## 5 · 什么不要提交

`.gitignore` 已经排除：`models/` `checkpoints/` `outputs/` `runs/` `data/` `wandb/` `.env` `*.key` `HF_TOKEN`。

两条底线：

1. **模型权重、数据集绝不入库。** 一个 Qwen 7B 是 15GB，GitHub 单文件上限 100MB、仓库建议 1GB 以内。推上去会直接失败，而且**一旦进了历史，删起来极其麻烦**。
2. **任何 key / token / 密码绝不入库。** 包括 HuggingFace token、云平台密钥、API key。**公开仓库的密钥泄露是永久性的**——爬虫几分钟内就会扫到。

万一误提交了（还没 push 时最好救）：

```bash
# 只是取消了跟踪，本地文件还在
git rm -r --cached models/
git commit -m "chore: 移除误加入的模型目录"
```

如果已经 push 了而且含密钥：**立刻去对应平台吊销那个 key**，删历史是第二步。

---

## 6 · 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 贡献图不显示我的提交 | git email 和 GitHub 账号不一致 | 改 1.2 的配置；`git log --format='%ae'` 检查 |
| `git status` 中文文件名显示成转义码 | `core.quotepath` 默认 true | `git config --global core.quotepath false` |
| 每次 checkout 后整个文件都显示改动 | 行尾 CRLF / LF 不一致 | 已有 `.gitattributes` + `core.autocrlf false` |
| push 被拒（`rejected — fetch first`） | 远程有你本地没有的提交（比如建仓库时勾了 README） | `git pull --rebase origin main` 再 push |
| push 要反复输密码 | 没走凭据管理器 | 用 HTTPS + 浏览器登录一次；或用 SSH key |
| `filename too long` | Windows 路径长度限制 | `git config --global core.longpaths true` |

**一条铁律：不要用 `git push --force`。** 你现在是单人仓库还摔不疼，但这个习惯一旦养成，在协作仓库里是灾难级的。

---

## 7 · 周节奏（每周日 15 分钟）

1. **更新 `README.md` 里的进度表**（W1 ✅ / W2 🟡 ...）
2. 扫一遍本周 `journal/`，看有没有实验做完却忘了写记录
3. 看一眼贡献图：**这周有没有断档**
4. 把下周的 `weekNN/` 目录建好（空目录也行，`touch` 一个 README）
5. `git add -A && git commit -m "docs: 更新第 N 周进度" && git push`

---

## 8 · 面试前一周要做的事

- 把仓库 README 从头读一遍，**假装自己是面试官**：能不能 30 秒内看懂这个仓库在做什么、怎么跑起来
- **确认 README 里的命令真的能跑**（在干净环境里验一次）。跑不通的 README 是负分
- 给最核心的 2–3 个项目在 README 里加"结果"段落：**一句话结论 + 一张表/图**
- 简历上的 GitHub 链接点一遍，确认是公开的

---

## 附：为什么 commit 要天天有

一个零实习无论文的候选人，简历上唯一能"自证"的东西就是这个仓库。

招聘方在 30 秒内会形成两个判断：**这人做过吗？** 和 **这人靠谱吗？**

第一个靠 README 和代码，第二个靠 commit 记录。**两个都过了，你才有机会进到"考察能力"那一轮。**

所以：

> **每天 3 分钟，宁可提交得零碎，也不要三天不提交然后一次推一大堆。**

零碎的高频提交本身就是信号——它说明你在持续工作，而不是赶工。

---

*配套：`journal/README.md`（日志模板）、`PLAN.md`（26 周计划）*
