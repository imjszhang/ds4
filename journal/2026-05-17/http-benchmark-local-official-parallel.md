# HTTP 基准（本地 / 官方）与 OpenClaw Agent 路径验证（单路 / 双路）

> 日期：2026-05-17  
> 项目：ds4  
> 类型：功能实现 / 调研分析  
> 来源：Cursor Agent 对话

> **口径**：**LOCAL ↔ OFFICIAL** 一段是 **直连 Chat Completions** 的延迟/吞吐对比；**OpenClaw** 一段是 **已部署 Agent 栈（Gateway、编排、会话、工具上下文等）** 的 **能力与整轮表现** 验证，**不**与上段「裸 API 官方」做同一维度对标。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [分析过程](#2-分析过程)
3. [方案设计](#3-方案设计)
4. [实现要点](#4-实现要点)
5. [验证与测试](#5-验证与测试)
6. [后续演化](#6-后续演化)

---

## 1. 背景与动机

需要对 **OpenAI 兼容 Chat Completions** 在 **本地**（`ds4-server`）与 **DeepSeek 官方 API** 之间做可复现的延迟与吞吐对比；并验证 **多 worker** 场景下需 **并发客户端** 才能压满服务端，而非仅重复串行请求。

顺带解决：密钥已写在仓库 `.env` 中，但原脚本只读进程环境变量，导致未 `export` 时无法跑 `--compare-official`。

**延伸（与 HTTP 对比无关）**：OpenClaw 侧用于在 **同一套用户可见 prompt**（便于复现脚本）下，对 **`openclaw agent`（如 main）** 做 **Agent 能力 / 整轮可用性** 相关观测：能否稳定完成一轮、墙钟与 CLI 元数据、payload 是否正常等。要求 **独立 explicit 会话**，避免占用默认 `main` 对话、也避免历史 transcript 干扰。**目的不是**回答「与 DeepSeek 官方直连 API 谁更快」，而是回答「经 OpenClaw 的 Agent 在这条输入上表现如何」。

## 2. 分析过程

- 现有脚本：`scripts/benchmark_http_server.py`，支持流式 + usage、非流式、`--align` 短输出对齐、`--compare-official`。
- `ds4-server` 支持 `--workers`（1–2）；双 worker 时需 **同时** 在途请求才能体现并行，串行 `--repeat` 仍会排队在单槽上。
- 官方与本地对齐 thinking：默认同发 `thinking: {"type":"disabled"}`（与 `--no-thinking` 一致）；遇 400 可用 `--strip-thinking-official`（脚本已支持，本次未用到）。
- **OpenClaw**（见 openclaw 仓库内 `docs/cli/agent.md`、`docs/tools/agent-send.md`）：脚本化入口为 **`openclaw agent --message … --json`**，默认经 **Gateway**；`--session-id` 与 `--agent` 可固定路由。显式 session id 解析为 **`agent:<agentId>:explicit:<id>`**，与默认 **`agent:main:main`** 隔离，满足「独立 session」需求。

## 3. 方案设计

| 能力 | 做法 | 理由 |
| ---- | ---- | ---- |
| 读 `.env` | 在 `main()` 最早处加载；尝试仓库根与 `cwd` 的 `.env`；**不覆盖**已存在环境变量 | 与常见 dotenv 行为一致；免手写 `export` |
| 并发压测 | 新增 `--parallel N`，每轮用线程池同时发 `N` 个相同 chat 请求 | 与 `--workers 2` 语义匹配；并报告 `batch_wall` 与 `sum_tok/s batch` |
| OpenClaw 段 | `subprocess` 调用 `openclaw agent`；`--session-id ds4-bench-<uuid>`；每轮/每路并发各用新 uuid | **Agent 栈** 验证；与 LOCAL/OFFICIAL **不作同类对比** |
| OpenClaw thinking | `--no-thinking` 时默认传 CLI `--thinking off`；可被 `--openclaw-thinking` 覆盖 | 与 HTTP 侧「关 thinking」倾向一致 |

### 关键决策

| 决策 | 选择 | 理由 |
| ---- | ---- | ---- |
| 是否引入 `python-dotenv` | 否，自写极简解析 | 脚本零额外依赖 |
| 并发模型 | `ThreadPoolExecutor` + 复用现有 `post_stream_sse` / `post_json` | urllib 同步 I/O，线程足够 |
| OpenClaw 指标 | 墙钟 + JSON `meta.durationMs`（若有）+ `payloads` 文本字符 | CLI 未必暴露 tokenizer `completion_tokens`；字符数仅辅助 |

## 4. 实现要点

- **文件**：`scripts/benchmark_http_server.py`（HTTP + 可选 OpenClaw 冒烟）；`scripts/eval_openclaw_agent.py`（**专用** Agent 题库回归）；`scripts/openclaw_agent_common.py`（`.env`、payload 文本拼接、`run_openclaw_agent_once`）。
- **HTTP 侧**：`load_dotenv_candidates`（自 common）；`--parallel N`；`parallel>1` 时的 `batch_wall`、`sum_tok/s batch`；COMPARE 表（无并发时 batch 行为 N/A）。
- **OpenClaw 冒烟**（benchmark）：`run_openclaw_benchmark`（warmup、repeat、`--parallel` 并发多 session、检测 `meta` 中 embedded/fallback）；可选 **`--openclaw-model`** 传入 CLI **`--model`**；输出块标明 **Agent 路径，非 API 对标**。
- **OpenClaw 评测**（eval）：读 JSON 数组题库；**单轮**题为每题每 attempt 独立 `--session-id`；**多轮**题为同一次 attempt 内 **`turns`** 数组中的各轮**共用**同一 `session-id`。题目二选一：顶层非空 **`prompt`**，或非空 **`turns`**（每项含 `prompt`），二者不可并存。判据 **`expect`** 默认作用于**最后一轮**的 Agent 输出文本；**`max_wall_s`** 在多轮时表示**各轮墙钟之和**。可选 **`all_turns_exit_zero`**（省略时默认 **true**）：任一轮非 0 exit 即本题失败（与末轮的 `exit_zero` 判据配合使用）。`expect` 另含 `output_contains`、`output_regex`、`min_output_chars`、`no_embedded_fallback` 等；suite 有失败则进程 exit 1；可选 `--report-json`。**`--model <provider/id>`** 等价 **`openclaw agent --model`**，覆盖 `~/.openclaw/openclaw.json` 中 `agents.defaults.model.primary`；与本机配置中 **`qwen/qwen3.6-plus`**（`agents.defaults.models` 键名、Model Studio 提供方）对齐时可作 A/B。
- **数据类**：`SuiteAgg` …（benchmark 专用）。

### 项目结构（与本篇相关）

```
ds4/
├── scripts/
│   ├── benchmark_http_server.py
│   ├── eval_openclaw_agent.py
│   ├── openclaw_agent_common.py
│   ├── openclaw_agent_tasks.example.json
│   └── openclaw_agent_tasks.kl46.json
└── journal/
    └── 2026-05-17/
        └── http-benchmark-local-official-parallel.md
```

### 关键模块

| 文件 | 职责 |
| ---- | ---- |
| `scripts/benchmark_http_server.py` | 直连 Chat：本地/官方计时与对比；可选 `--openclaw`：**Agent 栈**冒烟，非官方 API 对标 |
| `scripts/eval_openclaw_agent.py` | JSON 题库 + 断言 + 通过率 / 墙钟分位数 + `--report-json` |
| `scripts/openclaw_agent_common.py` | `openclaw agent --json` 与 `.env` 加载，benchmark 与 eval 共用 |
| `scripts/openclaw_agent_tasks.example.json` | 示例题库，可复制后自行增删 |
| `scripts/openclaw_agent_tasks.kl46.json` | KL46 分层样例（L1 能跑 / L2 能用 / L3 边界），含多轮 `turns` 与可选 `tier` 字段（脚本忽略 `tier`，仅作文档） |

### OpenClaw 相关 CLI 参数（脚本内）

| 参数 | 默认 | 说明 |
| ---- | ---- | ---- |
| `--openclaw` | 关 | 开启 Agent 路径段（LOCAL 之后、OFFICIAL 之前）；语义为 **测 Agent**，非扩展「官方 API 对比」 |
| `--openclaw-bin` | `$OPENCLAW_BIN` 或 `openclaw` | 可执行路径 |
| `--openclaw-agent` | `main` | `--agent` |
| `--openclaw-local` | 关 | 等价 CLI `--local` |
| `--openclaw-thinking` | （空） | 显式覆盖；否则在 `--no-thinking` 下为 `off` |
| `--openclaw-session-prefix` | `ds4-bench` | `{prefix}-<uuid>` |
| `--openclaw-model` | （空） | 传给 **`openclaw agent --model`**；空则用配置默认（如 `local-llm/deepseek-v4-flash`） |
| `--openclaw-warmup` | `1` | 预热次数 |

### Agent 能力回归（独立脚本，与 HTTP 不对标）

```bash
cd /path/to/ds4
python3 scripts/eval_openclaw_agent.py \
  --tasks scripts/openclaw_agent_tasks.example.json \
  --report-json /tmp/openclaw-eval-report.json
```

分层叙事样例（本地龙虾「能跑 → 能用 → 边界」）：

```bash
python3 scripts/eval_openclaw_agent.py \
  --tasks scripts/openclaw_agent_tasks.kl46.json \
  --report-json /tmp/openclaw-eval-kl46-report.json
```

与配置中 **Qwen3.6 Plus** 对照（`~/.openclaw/openclaw.json`：`models.providers.modelstudio`、`agents.defaults.models` 键 **`qwen/qwen3.6-plus`**）：

```bash
python3 scripts/eval_openclaw_agent.py \
  --tasks scripts/openclaw_agent_tasks.kl46.json \
  --model qwen/qwen3.6-plus \
  --session-prefix ds4-eval-qwen36 \
  --report-json /tmp/openclaw-eval-kl46-qwen36-report.json
```

题库为 **JSON 数组**；字段见 §4。`--no-dotenv` 可跳过加载 `.env`。

**环境**：本地默认 `--base-url http://127.0.0.1:8005/v1`；官方 `https://api.deepseek.com/v1`；密钥来自 `.env` 中 `DEEPSEEK_API_KEY`（由脚本加载）。OpenClaw 需本机可执行 `openclaw` 且 Gateway 可用（除非 `--openclaw-local`）。

**仅 HTTP + 官方**：

```bash
cd /path/to/ds4
python3 scripts/benchmark_http_server.py \
  --compare-official \
  --no-thinking \
  --align short-tokens \
  --repeat 3 \
  --parallel 1
```

**增加 OpenClaw（main，独立 explicit session，用于 Agent 能力路径）**：

```bash
python3 scripts/benchmark_http_server.py \
  --compare-official \
  --no-thinking \
  --align short-tokens \
  --repeat 3 \
  --parallel 1 \
  --openclaw
```

## 5. 验证与测试

### 测试设备与环境（2026-05-17 实测）

下文 **§5.0–§5.4** 及其中 OpenClaw / KL46 的数值均在这台**客户端所在主机**上采集；用于说明**可复现时的硬件与系统边界**（与机房共置、其他机型对比时应另记设备篇）。HTTP 官方侧为**本机经公网**访问 `api.deepseek.com`，RTT 与路由受出口影响；本地 `ds4-server` 为 **127.0.0.1** 回环（与正文 `--base-url http://127.0.0.1:8005/v1` 一致）。

| 项目 | 说明 |
| ---- | ---- |
| 机型 | Apple **Mac Studio** |
| 型号标识 | **Mac15,14** |
| Apple 配置 / 型号编号 | **Z1CE001AMCH/A** |
| 芯片 | **Apple M3 Ultra** |
| CPU | **32** 核（**24** 性能核 + **8** 能效核）；同步 **32** 逻辑处理器 |
| 内存 | **512 GB** |
| 操作系统 | **macOS 26.2**（内部版本 **25C56**），内核 **Darwin 25.2.0** |
| 系统固件 | Boot ROM / 系统固件 **13822.61.10** |
| 主存储 | 内置 **APPLE SSD AP16384Z**，**APFS**，系统数据卷总容量约 **16 TB**（介质：**SSD**，协议：**Apple Fabric**，内建） |
| Python（脚本解释器） | **3.14.4**，路径 **`/opt/homebrew/bin/python3`**（**Homebrew arm64** 安装） |

**说明**：未在文中记录主板序列号、硬件 UUID 等标识，以免无关扩散；若需在内部工单中精确对机，可在本机用「系统信息」或 `system_profiler SPHardwareDataType` 自行核对。

### 5.0 本轮数据是怎么采的（2026-05-17 完整复测）

同机连续执行三条命令（中间未改 `ds4-server` / OpenClaw 配置；**官方 HTTP 受公网 RTT 影响**，**本地 HTTP 受当时负载影响**，故与过往日记数字并存时以本节为准）。对齐参数：`align=short-tokens`、`max_tokens=32`、`model=deepseek-v4-flash`、用户消息 **20 字符**（脚本 `[数据来源] prompt 字符数=20`）、thinking **disabled**。

| 顺序 | 命令要点 | 报告用途 |
| ---- | -------- | -------- |
| 1 | `benchmark_http_server.py --compare-official --no-thinking --align short-tokens --repeat 3 --parallel 1 --openclaw` | §5.1 单路 HTTP + §5.1b OpenClaw（单路 CLI） |
| 2 | 同上，`--parallel 2` | §5.2 双路 HTTP + §5.1c OpenClaw（双路 CLI 并发） |
| 3 | `eval_openclaw_agent.py --tasks scripts/openclaw_agent_tasks.kl46.json --report-json /tmp/openclaw-eval-kl46-report.json` | §5.4 KL46 |

OpenClaw 段：`openclaw bin=openclaw`（或 `$OPENCLAW_BIN`）、`agent=main`、会话 `ds4-bench-<uuid>`（benchmark）或 `ds4-eval-…`（KL46）、`timeout_s=600`（benchmark）/ `900`（KL46）。**decode 分列**：本批 TTFT ≈ wall，脚本打印 `tok/s ~(decode)` 与 `ms/out_tok ~(decode)` 为 **N/A**。

### 5.0b OpenClaw 测试「题面是什么、在测什么」（中文）

OpenClaw 相关分两类：**HTTP 基准里挂的冒烟**（与 KL46 题库 **不是** 同一套题），以及 **KL46 专项题库**（`openclaw_agent_tasks.kl46.json`）。下面按 **发给模型/Agent 的用户消息** 与 **脚本如何用输出做门禁** 说明；便于读者区分「测整轮链路的短问」和「测叙事能力的 11 题」。

#### （一）benchmark 中的 OpenClaw 冒烟（§5.1b / §5.1c）

| 项目 | 说明 |
| ---- | ---- |
| **题从哪来** | `benchmark_http_server.py` 在 `--align short-tokens` 且未指定 `--prompt` / `--prompt-file` 时，把用户消息设为固定短问（与 LOCAL/OFFICIAL HTTP **同一** `user_message`，便于对齐「同一句话」走 HTTP 与走 Agent 的差异）。 |
| **具体题目** | **「用不超过20个汉字解释：栈和队列的区别。」**（**20 个汉字**，与 §5.0「20 字符」即此题长度一致。） |
| **在测什么** | 在 **Thinking off**、`max_tokens=32` 前提下，经 **Gateway 的完整 Agent 一轮**：CLI 能否 **exit 0**、JSON 是否可解析、`payloads` 是否有可读文本；并采样 **进程墙钟**、**`meta.durationMs`**、**回复字符数**；`parallel=2` 时再观察 **双路独立 session** 并发时的墙钟与 batch。 |
| **不叫「能力题」的原因** | 冒烟 **不设** `expect` 子串断言；不检查答案正确性，只看 **链路可用 + 时延量级**。 |

#### （二）KL46 题库（§5.4，`eval_openclaw_agent.py`）

整体在测：**经 OpenClaw 跑通多类用户任务时**，能否 **稳定 exit 0**，且 **末轮（或多轮最后一轮）输出** 满足 **长度 / 子串 / 正则** 等门槛。分层含义：**L1** 偏「能跑、基础语感与延迟叙述」；**L2** 偏「日常向：规划、工具表述、长文摘录、多轮记忆」；**L3** 偏「代码片段、小推理、常识、并发常识」。**同一套题**可用 **`--model`** 换后端（如本地 Flash vs 云端 Qwen）做 **A/B**，对比墙钟与叙事差异。

| `id` | 层 | 验证目的（白话） | 发给 Agent 的题意（摘要） | 自动判定（`expect` 要点） |
| ---- | -- | ---------------- | ------------------------- | ------------------------- |
| `L1-lobster-local-first-contact` | L1 | Agent 能否根据**当前会话可见信息**说明是否在本地模型上跑，避免胡编系统细节 | 两三句话：是否可能在「本地」推理、**依据是什么** | `exit 0`；输出不少于 **20** 字 |
| `L1-zh-after-heavy-quant` | L1 | 大模型被强量化后 **中文能力**是否还能组织简短中文回答 | 284B 级模型激进量化后，中文能力**常见退化**列两条，每条 ≤30 字，中文答 | `exit 0`；≥15 字；正则 **至少一个汉字** |
| `L1-latency-user-perception` | L1 | 能否用通俗中文拆解 **端到端延迟**（不配具体毫秒） | ≤100 字：一次问答延迟由哪些**主要环节**构成（网关、排队、首 token、生成等），**不要编造精确毫秒** | `exit 0`；≥30 字 |
| `L2-plan-complex-article` | L2 | **复杂指令拆解**成可执行子步骤 | 写技术文章 + 配图 + 排版 → **至少 5 步**编号列表，每步一行、步骤名短 | `exit 0`；输出含 **「1」…「5」**；≥40 字 |
| `L2-tool-weather-path` | L2 | **工具/联网**相关意图：会否查天气、缺什么前提、离线怎么说 | 若查「今天上海气温」：**会否用联网或工具**、需要你提供什么前提、**无法联网**时如何答复 | `exit 0`；须同时出现 **「天气」**与 **「联网」** |
| `L2-longcontext-ds4-motivations` | L2 | **长文进 prompt**：只依据摘录回答，不脑补 | 给出 DS4 README「Motivations」**英文摘录**；中文问：第 **3** 与 **6** 条各讲什么；概括里分别体现 **「百万」**与 **「KV」** | `exit 0`；含 **「百万」**与 **「KV」** |
| `L2-multi-turn-remember-KL46` | L2 | **同一 session 多轮**：是否记住前文代号 | 第 1 轮：记住代号 **KL46**，只答「收到」；第 2 轮：**只输出**代号本身无标点空格 | 两轮均 `exit 0`；末轮输出含 **「KL46」** |
| `L3-code-python-snippet` | L3 | **短代码生成**是否含定义与目标输出词 | 最小可运行 Python3：`hello()` 打印 **Hello**；可放在 markdown 代码块 | `exit 0`；含 **「def」**与 **「Hello」** |
| `L3-reasoning-chicken-rabbit` | L3 | **小学约束推理**能否落成指定格式（断言偏格式，不代替人工验算） | 鸡兔同笼 8 头 22 腿；**一句**「鸡几只兔几只」，阿拉伯数字，例：鸡5兔3 | `exit 0`；含 **鸡、兔、5、3**（**不**保证算术必对，仅检查关键词与数字子串） |
| `L3-knowledge-capital` | L3 | **极短常识**输出是否干净 | 法国首都 → **只输出**标准中文三字城市名 | `exit 0`；含 **「巴黎」** |
| `L3-concurrency-local-queue` | L3 | **本地单实例多会话**时调度/排队的常识描述，避免空话过短 | ≤120 字：单机单卡或单推理进程、多并发会话时请求**如何排队/调度**，勿绝对化承诺 | `exit 0`；≥25 字 |

**说明**：KL46 的 **墙钟表**仍见 §5.4；本节只补齐 **题与目的**，避免表格里只有 `id` 却读不懂「到底考了什么」。

### 5.1 单路直连 HTTP：LOCAL vs OFFICIAL（`--parallel 1`）

源自 §5.0 命令 **1**；本节 **不包含** OpenClaw，仅 **Chat Completions**。

| 指标 | LOCAL | OFFICIAL |
| ---- | ----- | -------- |
| GET `/v1/models` ms（单次） | **11.1** | **97.8** |
| 流式 `wall_total_s`（`repeat=3` 均值） | **1.056** | **0.759** |
| 流式 `TTFT_s`（均值） | **1.056** | **0.759** |
| 流式 `prompt_tokens`（均值） | **16.0** | **16.0** |
| 流式 `completion_tokens`（均值） | **11.0** | **11.0** |
| 流式 `tok/s e2e`（均值） | **10.42** | **14.82** |
| 流式 `ms/output_tok e2e` | **96.01** | **68.99** |
| 流式 `finish_reason` | **stop**（3/3） | **stop**（3/3） |
| 非流式 `wall_total_s`（均值） | **1.055** | **0.818** |
| 非流式 `ms/output_tok e2e` | **95.88** | **74.39** |

**解读（HTTP）**：本批 LOCAL 单路 wall 与 tok/s 劣于 OFFICIAL，属 **负载 + RTT + 批内波动** 组合结果；`ms/output_tok` 仍可做 **每输出 token 摊到的墙钟** 对照。`completion_tokens` 与 prompt 对齐一致。

### 5.1b OpenClaw Agent（同 §5.0 命令 **1**，CLI 单路）

与 §5.1 **同一次** `benchmark` 进程内，在 LOCAL HTTP 之后插入；**独立 explicit session**；用户消息同上（**20 字符**）；`repeat=3`。**题面与冒烟目的**见 **§5.0b（一）**。

| 指标 | 值 |
| ---- | -- |
| 进程墙钟 `wall_total_s`（**每样本**均值，`repeat=3`） | **6.875** |
| JSON `meta.durationMs`（均值，若存在） | **4319.0** |
| 回复文本字符（payloads 累加，均值） | **13.7** |
| 样本数 × 退出码 | **3** \| 非零 **0** |

**解读（Agent）**：整轮 **Gateway + 会话 + 模型**；`meta.durationMs` 与进程 wall 的差来自 **CLI/RPC**。**勿**与 §5.1 OFFICIAL HTTP 做「谁更快」式对标。

### 5.1c OpenClaw Agent（同 §5.0 命令 **2**，CLI `parallel=2`）

同次 benchmark 内，`--repeat 3`、`--parallel 2`：每轮 **2** 路 `openclaw agent` 并发，共 **6** 条样本。题面仍为 §5.0b（一）同一短问。

| 指标 | 值 |
| ---- | -- |
| `wall_total_s`（每样本均值，6 次） | **9.830** |
| `meta.durationMs`（均值） | **6519.3** |
| 回复文本字符（均值） | **15.3** |
| 进程退出码 | **6** 次均为 **0** |
| `batch_wall_s`（max，每轮两路取最长） | **11.634** |
| `sum_字符/s`（batch，payload 字符和 ÷ batch_wall） | **2.63** |

**解读**：双路并发时 Agent 墙钟与 batch_wall 上升，用于观察 **多 session 争用**；仍 **非** HTTP 竞品指标。

### 5.2 双路并发 HTTP（`--parallel 2`）

源自 §5.0 命令 **2** 中打印的 **LOCAL** 与 **OFFICIAL** 直连 HTTP 汇总（同次进程内另有 OpenClaw 块，见 §5.1c）。`repeat=3` → 每侧 **6** 次流式/非流式 HTTP 样本。

| 指标 | LOCAL | OFFICIAL |
| ---- | ----- | -------- |
| GET `/v1/models` ms | **12.3** | **94.9** |
| 流式 `wall_total_s`（单笔均值） | **1.592** | **0.705** |
| 流式 `TTFT_s`（均值） | **1.591** | **0.704** |
| 流式 `completion_tokens`（均值） | **11.0** | **11.0** |
| 流式 `tok/s e2e`（均值） | **7.78** | **15.90** |
| 流式 `ms/output_tok e2e` | **144.70** | **64.06** |
| 流式 `batch_wall_s`（max） | **2.124** | **0.758** |
| 流式 `sum_tok/s batch` | **10.36** | **29.74** |
| 流式 `finish_reason` | **stop**（6/6） | **stop**（6/6） |
| 非流式 `wall_total_s`（单笔均值） | **1.595** | **0.827** |
| 非流式 `ms/output_tok e2e` | **144.98** | **75.19** |
| 非流式 `batch_wall_s`（max） | **2.126** | **0.895** |
| 非流式 `sum_tok/s batch` | **10.35** | **24.73** |

**解读**：并发下本地 **单笔 wall**、**ms/output_tok** 相对单路 §5.1 变差；官方 **batch** 更短、`sum_tok/s batch` 更高。若宣称双 worker，应确认服务端日志与 `--workers 2`。

### 5.3 历史单路 HTTP 对照（不同批次、无 OpenClaw）

较早一次采样（**非** §5.0 同批）：LOCAL 流式 wall **0.557**、ms/output_tok **50.61**；OFFICIAL wall **0.693**、ms/output_tok **63.03**；官方 GET `/models` **244.6 ms**。与 §5.1 本批（**1.056 / 0.759** 等）**不可混为同一实验**；仅说明 **复测间波动**。

日记曾记另一并发批（旧 §5.2）：LOCAL 流式单笔均值 **0.950**、batch_wall **1.139**、sum_tok/s **19.34**；OFFICIAL **0.628 / 0.649 / 34.11**；非流式 sum_tok/s batch LOCAL **19.31**、OFFICIAL **24.81**。与 §5.2 本批亦 **不同场**。

### 5.4 OpenClaw Agent 题库回归（KL46）— 双模型对照

每道题的**中文题意、验证点、断言规则**见 **§5.0b（二）**；本节保留 **数据与对照**。

`scripts/eval_openclaw_agent.py` + **`scripts/openclaw_agent_tasks.kl46.json`**；`agent=main`，`--timeout 900`，逐题串行，`repeat=1`。机器报告：默认模型 **`/tmp/openclaw-eval-kl46-report.json`**；Qwen  **`/tmp/openclaw-eval-kl46-qwen36-report.json`**（路径均可由 `--report-json` 自定）。

| 配置 | `--model` | `--session-prefix` | OpenClaw 配置出处（本机） |
| ---- | --------- | ------------------ | ------------------------ |
| **A. 默认主模型** | （省略） | `ds4-eval` | `agents.defaults.model.primary` → **`local-llm/deepseek-v4-flash`**（`models.providers.local-llm`） |
| **B. Qwen3.6 Plus** | **`qwen/qwen3.6-plus`** | `ds4-eval-qwen36` | `agents.defaults.models` 键名；`models.providers.modelstudio` 中 **`qwen3.6-plus`**（`~/.openclaw/openclaw.json`） |

两套均为 **`summary.all_passed`: true**、`11/11` runs、脚本 exit **0**；`fallback_warning` 均为 **false**。

| 汇总项 | A（Flash 本地） | B（Qwen 云端） |
| ---- | -----: | -----: |
| 各题 `wall_s` 之和 | **≈145.70** | **≈102.79** |
| 单题 wall 最小 / 最大 | **6.182 / 25.331** | **5.037 / 19.684** |
| 多轮题 `L2-multi-turn-remember-KL46` 合计 | **10.939**（轮次 **6.809 + 4.130**） | **19.684**（**14.827 + 4.857**） |

**解读**：墙钟为 **进程级整轮**（Gateway + 会话 + 生成），**不与** §5.1 HTTP 对标。同一 **`expect`** 下两模型均通过；**叙事上** L1 首题答案会随 `--model` 变化（Flash 侧重「local-llm」、Qwen 侧重「当前为 qwen/qwen3.6-plus 云端」）。本批 Qwen 在多数题目上 **更短**，但 **多轮首答** 更慢，可与网关路由、云端排队、首轮输出长度相关。

#### 逐题 `wall_s`（秒）对照

| `id` | 分层 | A Flash | B Qwen3.6+ | Δ（B−A） |
| ---- | ---- | ------: | ---------: | -------: |
| `L1-lobster-local-first-contact` | L1 | 21.703 | 15.315 | −6.388 |
| `L1-zh-after-heavy-quant` | L1 | 8.800 | 7.675 | −1.125 |
| `L1-latency-user-perception` | L1 | 12.775 | 8.329 | −4.446 |
| `L2-plan-complex-article` | L2 | 25.331 | 7.642 | −17.689 |
| `L2-tool-weather-path` | L2 | 21.281 | 10.527 | −10.754 |
| `L2-longcontext-ds4-motivations` | L2 | 10.883 | 7.040 | −3.843 |
| `L2-multi-turn-remember-KL46` | L2 | 10.939 | 19.684 | +8.745 |
| `L3-code-python-snippet` | L3 | 8.497 | 6.669 | −1.828 |
| `L3-reasoning-chicken-rabbit` | L3 | 7.018 | 6.652 | −0.366 |
| `L3-knowledge-capital` | L3 | 6.182 | 5.037 | −1.145 |
| `L3-concurrency-local-queue` | L3 | 12.492 | 8.225 | −4.267 |

复现：

```bash
cd /path/to/ds4
# A：默认主模型（local-llm/deepseek-v4-flash）
python3 scripts/eval_openclaw_agent.py \
  --tasks scripts/openclaw_agent_tasks.kl46.json \
  --session-prefix ds4-eval \
  --report-json /tmp/openclaw-eval-kl46-report.json
# B：Qwen3.6 Plus（与 openclaw.json 中 qwen/qwen3.6-plus 一致）
python3 scripts/eval_openclaw_agent.py \
  --tasks scripts/openclaw_agent_tasks.kl46.json \
  --model qwen/qwen3.6-plus \
  --session-prefix ds4-eval-qwen36 \
  --report-json /tmp/openclaw-eval-kl46-qwen36-report.json
```

## 6. 后续演化

- 可增加 **`--parallel` 与官方限流** 的说明或可选「官方侧 `--parallel 1`」对照，避免误触配额。
- 若 TTFT ≈ wall，decode 分列仍为 N/A；可考虑更细的首 token / 后续分段时间戳（需服务端或 chunk 级解析）。
- **OpenClaw**：并行 `--parallel 2` + 双 session 可观察多路 Agent 行为。**Agent 能力回归**用 `eval_openclaw_agent.py`（可选 **`--model`** 与默认 `local-llm/deepseek-v4-flash` / `qwen/qwen3.6-plus` 做 KL46 对照；报告 `summary.model` 可记录覆盖值）+ `openclaw_agent_tasks.*.json`；冒烟段可用 **`--openclaw-model`**。
- 模板中路径写为其他仓库的 `docs/prism/journal/`，ds4 内约定为 `journal/YYYY-MM-DD/*.md`，与本条一致即可。

---

<!--
本仓库日记路径：journal/YYYY-MM-DD/<slug>.md（slug 英文小写 + 连字符）。
-->
