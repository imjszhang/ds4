# ds4-server 双 Worker HTTP 并发与运维备忘

> 日期：2026-05-15  
> 文档撰写时间（终端）：2026-05-15 12:34:57 CST  
> 项目：ds4  
> 类型：架构设计 / 功能实现 / 问题排查  
> 来源：Cursor Agent 对话

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

**问题 1：** `ds4_server.c` 是否「单线程」，能否改成「双线程」以提升并行？

**澄清：** 实际需求是 **多 worker**：允许 **两名用户同时访问 API**，推理路径不再串行排队在一套会话上。

**问题 2：** 进程退出后 Activity Monitor 仍显示较高内存占用，是否泄漏或未释放？

**问题 3：** 希望在 Mac Studio 场景下 **固定以 2 worker 启动**，并与脚本一致。

---

## 2. 分析过程

### 实际架构认知（纠正「整条链路单线程」）

- HTTP 侧：**每个连接**有独立的 client 线程读请求。
- 推理侧：原先 **单一 Metal worker** 从队列取 job，独占 **`ds4_session`**（KV、checkpoint、live/tool 状态）。
- 因此瓶颈在 **「单会话 + 单 worker」**，而非「进程只有一个 pthread」。

### 双 worker 的核心约束

- **不可**让两个 worker **共用同一个 `ds4_session`**：会破坏 KV frontier、tool memory、Responses/Anthropic live continuation 等语义。
- **可行 MVP：** 每个 worker 一套隔离状态：**独立队列、`ds4_session`、tool memory、live 状态**。
- **磁盘 KV：** 多 worker 同时写同一目录风险高 → **按 worker 分子目录**（如 `worker-0`、`worker-1`）。
- **Metal/CUDA：** `ds4_metal.m`（及 CUDA 路径）存在 **全局 command buffer / pending buffers / transient buffers** 等可变状态；仅改 server 层仍可能在 GPU  glue 层竞争 → 需要 **GPU 执行路径互斥或重构为 per-context**。

### 进程退出后「内存仍在」

- 用 `pgrep`/`ps` 确认 **无 `ds4-server` 进程** 后，`vm_stat` 常见表现为大量 **File-backed / Inactive** 页面。
- **结论：** 更像 **mmap 模型权重后的文件缓存 + 系统缓存**，不是典型的进程私有内存泄漏；压力上来或被回收策略驱逐后会下降。

---

## 3. 方案设计

### 总体形态

```
accept → client threads → 调度器（最少负载）→ worker slot 队列 → worker → ds4_session_i
```

### 关键决策

| 决策 | 选择 | 理由 |
| ---- | ---- | ---- |
| 会话模型 | 每 worker 独立 `ds4_session` | 保持原有「单会话顺序推进」语义，避免共享可变 frontier |
| 调度策略 MVP | `queued + running` 最少者优先 | 实现简单；sticky routing 留给后续 |
| KV 目录 | `--workers>1` 时用 `<base>/worker-{i}` | 避免索引与文件写入并发损坏 |
| Trace | 全局单文件 + 锁，`worker=id` 写入 | 多槽共享 trace 时不交错损坏 |
| GPU 并发（第一阶段） | Metal：`begin/end_commands` + `synchronize` 套全局锁；CUDA：`begin/end/synchronize` 同步路径加锁 | 先保证正确性；真正并行 GPU 需拆分全局 GPU 状态 |

---

## 4. 实现要点

### 涉及文件（本次代码变更）

| 文件 | 职责 |
| ---- | ---- |
| [`ds4_server.c`](/Users/jszhang/github/fork/ds4/ds4_server.c) | `server_group` + 多 slot；`--workers`（1–2）；调度；KV 子目录；shutdown 逐 worker 持久化；trace 聚合 |
| [`ds4_gpu.h`](/Users/jszhang/github/fork/ds4/ds4_gpu.h) | `ds4_gpu_execution_lock/unlock` 声明 |
| [`ds4_metal.m`](/Users/jszhang/github/fork/ds4/ds4_metal.m) | Metal 命令批次全局互斥；同线程 `synchronize` 时避免二次加锁 |
| [`ds4_cuda.cu`](/Users/jszhang/github/fork/ds4/ds4_cuda.cu) | CUDA `begin/end/synchronize` 与锁配对 |
| [`scripts/ds4-server-mac-studio-512g.sh`](/Users/jszhang/github/fork/ds4/scripts/ds4-server-mac-studio-512g.sh) | `DS4_WORKERS` 默认 `2`，传入 `--workers` |

### CLI

```bash
./ds4-server --workers 2 ...
```

帮助：`./ds4-server --help` 可见 `--workers N`（默认 1，最大 2）。

### Mac Studio 脚本示例

```bash
# 脚本已默认 DS4_WORKERS=2；也可显式指定
DS4_WORKERS=2 ./scripts/ds4-server-mac-studio-512g.sh
```

---

## 5. 验证与测试

| 项目 | 结果 |
| ---- | ---- |
| `make ds4-server` | 通过 |
| `make cpu`（`DS4_NO_GPU`） | 通过 |
| `make test` | 通过 |
| `bash -n scripts/ds4-server-mac-studio-512g.sh` | 通过 |
| 运行时日志 | `started 2 inference workers`；KV 分别为 `.../worker-0`、`.../worker-1` |
| HTTP 探活 | `curl -fsS http://127.0.0.1:8005/v1/models` → ok |
| 单元测试补充 | worker 负载调度偏好（最少负载） |

---

## 6. 后续演化

1. **Sticky routing：** 按 conversation/thread/user 等稳定 id 哈希到固定 slot，减少跨 worker 时的 live/tool 连续性损失。
2. **GPU 真正并行：** 将 Metal/CUDA 全局 `g_batch_cb`、`g_pending_cbs`、`g_transient_buffers` 等改为 **per-session / per-queue-context**，去掉或减少全局串行锁。
3. **workers 上限：** 若硬件与语义允许，放宽 `--workers` 上限并评估内存（每 session 一套 KV/graph）。
4. **运维文档：** 在 README 或运维小节标明「双 worker ≠ 双倍 GPU 吞吐」在当前阶段的语义（队列与会话并行 vs GPU 批次锁）。

---

<!--
存放：`journal/2026-05-15/ds4-server-multi-worker-http-runtime.md`
-->
