# DwarfStar 4（ds4）：本机Metal部署、国内镜像下载与HTTP性能对照

> 日期：2026-05-15  
> 落笔时间（本地）：2026-05-15 11:08:12 +0800  
> 项目：ds4（DwarfStar 4 / fork）  
> 类型：调研分析 · 升级迁移 · 功能实现 · 验证与测试  
> 来源：Cursor Agent 对话整理  
> 环境摘要：Apple Silicon（Darwin arm64）、约 512 GB RAM、Metal 后端；模型为 **deepseek-v4-flash** 的 GGUF **q4-imatrix**（约 153 GiB）

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

在 **512 GB RAM 的 Mac Studio** 上使用本仓库的 **Metal** 路径跑 **DeepSeek V4 Flash** 推理，并起一个 **兼容 OpenAI 的本地 HTTP 服务**（`ds4-server`），供日常使用或与其它工具对接。

同时希望对 **DeepSeek 官方 API** 与 **本地 `ds4-server`** 做可比性更强的 **端到端 HTTP 延时与吞吐取样**，厘清此前测试里发现的 **thinking 字段处理不一致** 等问题。

---

## 2. 分析过程

### 构建与运行时

- 仓库为 **原生 C**，在 macOS 上默认 **`make`** 产出带 **Metal** 的 `ds4`、`ds4-server`、`ds4-bench`；不提供 Docker 作为主要部署方式。
- **模型权重**：仅能用此项目配套的 GGUF；512 GB 机器按文档优先 **q4-imatrix**。
- **`download_model.sh`** 默认从 **`https://huggingface.co/.../resolve/main/…`** 拉取；当前网络环境下 **直连官网易超时**，**国内镜像域名**可走通（脚本侧已支持 **`DS4_HF_BASE`** 覆盖域名）。

### 性能测试分层

| 层级 | 工具 | 测的是什么 |
| ---- | ---- | ---------- |
| 引擎吞吐（不经 HTTP） | `./ds4-bench` + `speed-bench/` | KV/图上的 prefill/generation frontier |
| HTTP 体感 | `scripts/benchmark_http_server.py`（自研） | `GET /models`、首包（TTFT）、流式/非流式、`usage`、`finish_reason` |
| HTTP 对齐难点 | （实践总结） | 双方 **stop/max_tokens/thinking** 不一致会导致 **completion_tokens** 不可比 |

### Thinking 对齐问题（已厘清）

曾对官方请求 **`strip` 掉顶层 `thinking` 字段**（为规避旧版兼容性顾虑），再在本地传 **`thinking: {"type":"disabled"}`**，会导致：

- **本地**：显式关掉 thinking；  
- **官方**：等同「未声明」→ 往往仍走 **默认 thinking**（与其它侧不对齐）。

**修正方向**：DeepSeek **同样支持** `thinking: {"type":"disabled"}`，因此 **默认应向官方下发与本地一致的 JSON**；仅在遇 **`400`** 等特殊错误时，可选用 **`--strip-thinking-official`** 作为逃生开关。

---

## 3. 方案设计

### 下载

- **模型**：**q4-imatrix** → 约 **153 GiB**，落盘 `./gguf/`，并让 **`./ds4flash.gguf`** 指向该文件。  
- **镜像**：设置 **`DS4_HF_BASE=https://hf-mirror.com`**（无尾斜杠），其余路径与原版脚本一致：`$BASE/antirez/deepseek-v4-gguf/resolve/main/<文件名>`。

### 本机部署

- **二进制**：在项目根 **`make`**。  
- **服务脚本**：新增 **`scripts/ds4-server-mac-studio-512g.sh`**——  
  - 端口 **`8005`**；  
  - **`--ctx 1048576`**（上限档，按需可用环境变量 `DS4_CTX` 下调）；  
  - **`--warm-weights`**；  
  - 磁盘 KV：`~/Library/Caches/ds4-server-kv`，配额约 **96 GiB**（`98304` MiB）；  
  - 线程数：**`min(hw.logicalcpu, 24)`**；  
  - 可调环境变量：**`DS4_MODEL`** 在此处表示 **GGUF 路径**（默认 `ds4flash.gguf`），勿与下文 API **`model`** 混淆。

### 配置与密钥

- 仓库根 **`/.env`**（**已 `.gitignore`**）：  
  - **`DEEPSEEK_API_KEY`**：官方对照用（勿提交）；  
  - **`DS4_API_MODEL=deepseek-v4-flash`**：HTTP 脚本里 **`model`** 默认值来源，与 OpenAI 兼容字段一致。

### HTTP 基准与官方对照

- 脚本：**`scripts/benchmark_http_server.py`**（仅用标准库 **`urllib`**）。  
- **`--compare-official`**：先后跑 LOCAL 与官方，末尾打印 **`COMPARE`** 摘要表。  
- **对齐**：  
  - **`--align short-tokens` / `--align digit-answer`**、`temperature=0`、`top_p=1`、惩罚项置 **0**；  
  - 指标除 **tok/s** 外补充 **`ms/output_tok`**（墙钟 ÷ `completion_tokens`），减少「一边生得更长」带来的误读。

### 关键决策

| 决策 | 选择 | 理由 |
| ---- | ---- | ---- |
| 量化档位 | q4-imatrix | 文档建议 ≥256 GB RAM 机器使用 |
| 下载入口 | **`DS4_HF_BASE`** 指向国内镜像 | 本机实测直连 `huggingface.co` 易超时 |
| API 密钥 | `.env` + gitignore | 避免误入版本库 |
| `DS4_MODEL` vs `DS4_API_MODEL` | 分离 | 前者在启动脚本里表示 **权重路径**，后者表示 Chat **`model`** 名字 |
| 官方 thinking | 默认 **不下掉** JSON 字段 | 与 **`--no-thinking`** 对齐，否则可比性失真 |

---

## 4. 实现要点

### 变更与新增路径（节选）

```
ds4/
├── download_model.sh          # DS4_HF_BASE 镜像基址支持
├── .env                       # （本地）密钥与 DS4_API_MODEL，勿提交
├── .gitignore                 # 含 .env
├── scripts/
│   ├── ds4-server-mac-studio-512g.sh
│   └── benchmark_http_server.py
└── gguf/
    └── *.gguf                  # q4-imatrix 本体（不在版本库）
```

### 常用命令（可操作）

在项目根：

```bash
# 构建（Metal）
make -j

# 国内镜像下载 q4-imatrix（约 153 GiB）
DS4_HF_BASE=https://hf-mirror.com ./download_model.sh q4-imatrix

# 启动本机API（端口 8005）
./scripts/ds4-server-mac-studio-512g.sh

# 载入 .env（勿把密钥贴进聊天记录）
set -a && source .env && set +a

# HTTP 取样：关闭 thinking，并与官方对照 + 对齐预设 short-tokens
python3 scripts/benchmark_http_server.py \
  --compare-official \
  --base-url http://127.0.0.1:8005/v1 \
  --no-thinking \
  --align short-tokens \
  --warmup 1 --repeat 3
```

---

## 5. 验证与测试

### 5.1 模型与默认链路

- 下载完成后存在最终 **`.gguf`** 文件，`./ds4flash.gguf` 软链就绪。  
- **`GET http://127.0.0.1:8005/v1/models`** 返回中包含 **`deepseek-v4-flash`**，上下文长度信息与脚本设定一致。

### 5.2 烟囱测试（curl）

曾对 **`POST /v1/chat/completions`** 做：

- **非流式**：简单算术/短答，内容与 `finish_reason` 合理；  
- **流式 + `thinking.disabled`**：**SSE `data:` → `[DONE]`** 路径正常。

### 5.3 官方 vs 本地（thinking 对齐后的一轮 **`short-tokens`**）

下列条件便于人类阅读：**同一短问、`max_tokens=32`、三方均为 `thinking: disabled`、`repeat=3`、流式与非流式都测**。

| 指标（均值） | 本地 ds4-server | DeepSeek 官方 |
| ------------- | ---------------- | ------------- |
| `GET /models` | ~**11 ms** | ~**358 ms** |
| **`completion_tokens`（流式）** | **11** | **11** |
| **`finish_reason`（流式）** | **stop** | **stop** |
| **TTFT_s（流式）** | **~0.55** | **~0.95** |
| **`tok/s`（流式 e2e）** | **~20** | **~11.7** |
| **`ms/output_tok`（流式 e2e）** | **~50 ms** | **~87 ms** |
| **`ms/output_tok`（非流式 e2e）** | **~50 ms** | **~81 ms** |

**阅读习惯上的结论**：  

- **`/models` 与控制面**：本地显著更快（数量级差异主要来自 **网络与时延**，不是推理核心）。  
- **在长度与停机原因对齐后**：本轮 **首轮感知（TTFT）**与 **端到端每输出 token 成本**在本机链路下 **优于**官方 HTTP 这一轮读数——仍应理解这是 **端到端样本**，非厂商 SLA，也未覆盖长上下文与高负载排队。

历史上还存在 **thinking _strip 默认值**导致的「本地关 thinking、云端仍默认开」的假差异；脚本已调整为 **两岸同一 JSON**。若极少数环境报 **`400`**，再尝试 **`--strip-thinking-official`** 并标明「对齐性会降低」。

### 5.4 脚本缺陷修复（记入以免复现）

- `run_named_suite` 参数已由 `official_compat_payload` 改为 **`strip_thinking_for_official`** 后，**本地分支**有一处旧关键字未删尽会导致 **`TypeError`**，已改过；若你从旧脚本复制粘贴，注意不要混用两套参数名。

---

## 6. 后续演化

可改进但不限于：

| 方向 | 说明 |
| ---- | ---- |
| 载荷进一步对齐 | `stop`、`max_completion_tokens`、`response_format` 等与业务强绑定时再收口 |
| 引擎层对照 | 定期跑一次 **`ds4-bench`** CSV，与 HTTP 体感分开存档 |
| 监控 | 服务端 **`--trace`** 对疑难会话留存；磁盘 KV 目录可周期性清理 |
| 安全 | **`.env` 若曾泄露**：轮换控制台里的 **API Key** |
| Thinking 逃逸门 | **`--strip-thinking-official`** 仅作兼容；默认保持「同源 JSON」以减少误解读 |

---

*以上由对话与工作区操作整理而成，命令与数值以仓库内脚本及当次实测为准；生产环境请以你本机与安全策略为准再行核对。*
