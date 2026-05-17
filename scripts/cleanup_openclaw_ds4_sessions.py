#!/usr/bin/env python3
"""清理 OpenClaw 中由 ds4 基准/评测脚本创建的 explicit 会话。

``benchmark_http_server.py`` 使用 ``--session-id`` 前缀 ``ds4-bench-<uuid>``；
``eval_openclaw_agent.py`` 使用 ``ds4-eval-...``（含 ``ds4-eval-qwen36-`` 等）。

从 ``agents/<agent>/sessions/sessions.json`` 中删除对应键，并删除关联的
``.jsonl`` / ``.trajectory.jsonl`` / ``.trajectory-path.json``；并扫描目录中
残留的 ``ds4-bench-*``、``ds4-eval-*`` 文件名。

默认状态目录：环境变量 ``OPENCLAW_STATE_DIR``（若设置），否则 ``~/.openclaw``。

用法：
  python3 scripts/cleanup_openclaw_ds4_sessions.py --dry-run
  python3 scripts/cleanup_openclaw_ds4_sessions.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 与仓库内 benchmark / eval 的 session 前缀一致
EXPLICIT_PREFIXES = ("ds4-bench-", "ds4-eval-")


def _default_state_dir() -> Path:
    raw = (os.environ.get("OPENCLAW_STATE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".openclaw").resolve()


def _session_key_prefix(agent: str, blob: str) -> str:
    return f"agent:{agent}:explicit:{blob}"


def _is_ds4_test_session_key(agent: str, key: str) -> bool:
    if not isinstance(key, str):
        return False
    for p in EXPLICIT_PREFIXES:
        if key.startswith(_session_key_prefix(agent, p)):
            return True
    return False


def _collect_files_for_row(store: Path, row: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    sf = row.get("sessionFile")
    if isinstance(sf, str) and sf.strip():
        out.append(Path(sf).expanduser().resolve())
    sid = row.get("sessionId")
    if isinstance(sid, str) and sid.strip():
        sid = sid.strip()
        out.append(store / f"{sid}.jsonl")
        out.append(store / f"{sid}.trajectory.jsonl")
        out.append(store / f"{sid}.trajectory-path.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="从 OpenClaw session store 移除 ds4-bench / ds4-eval 测试会话"
    )
    ap.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=f"OpenClaw 状态根目录（默认：$OPENCLAW_STATE_DIR 或 ~/.openclaw）",
    )
    ap.add_argument(
        "--agent",
        default="main",
        help="Agent id，对应 agents/<id>/sessions/（默认 main）",
    )
    ap.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="直接指定含 sessions.json 的目录（若设则忽略 --state-dir/--agent）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的操作，不写回 JSON、不删文件",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份 sessions.json（不建议）",
    )
    args = ap.parse_args()

    if args.sessions_dir is not None:
        store = args.sessions_dir.expanduser().resolve()
    else:
        state = (args.state_dir or _default_state_dir()).expanduser().resolve()
        store = state / "agents" / str(args.agent).strip() / "sessions"

    agent = str(args.agent).strip() or "main"

    path = store / "sessions.json"
    if not path.is_file():
        print(f"[cleanup] 未找到 {path}", file=sys.stderr)
        sys.exit(2)

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print("[cleanup] sessions.json 根须为对象", file=sys.stderr)
        sys.exit(2)

    removed_keys: list[str] = []
    files_to_delete: set[Path] = set()

    for key, row in list(data.items()):
        if not _is_ds4_test_session_key(agent or "main", key):
            continue
        removed_keys.append(key)
        if isinstance(row, dict):
            for p in _collect_files_for_row(store, row):
                files_to_delete.add(p)

    for p in store.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name == "sessions.json" or name.startswith("sessions.json.bak"):
            continue
        if name.startswith(EXPLICIT_PREFIXES):
            files_to_delete.add(p.resolve())

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    backup = store / f"sessions.json.bak-ds4-cleanup-{ts}"

    print(f"[cleanup] store={store}")
    print(f"[cleanup] agent key 前缀: agent:{agent}:explicit:ds4-{{bench,eval}}-…")
    print(f"[cleanup] 将移除 {len(removed_keys)} 条 session 键")
    print(f"[cleanup] 将删除至多 {len(files_to_delete)} 个路径（去重后）")

    if args.dry_run:
        for k in removed_keys[:20]:
            print(f"  - key: {k}")
        if len(removed_keys) > 20:
            print(f"  … 另有 {len(removed_keys) - 20} 条")
        sys.exit(0)

    if not args.no_backup:
        shutil.copy2(path, backup)
        print(f"[cleanup] 已备份 -> {backup}")

    for k in removed_keys:
        del data[k]

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    deleted = 0
    for f in sorted(files_to_delete):
        try:
            if f.is_file():
                f.unlink()
                deleted += 1
        except OSError as e:
            print(f"[cleanup] 删除失败 {f}: {e}", file=sys.stderr)

    print(f"[cleanup] 已写回 sessions.json；已删除 {deleted} 个文件")


if __name__ == "__main__":
    main()
