"""OpenClaw CLI 辅助：.env 加载与 `openclaw agent --json` 单次调用。

供 benchmark_http_server.py 与 eval_openclaw_agent.py 共用。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    """将 .env 写入 os.environ；仅当键未设置时写入（不覆盖已有环境变量）。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = rest.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val


def load_dotenv_candidates() -> None:
    """依次尝试仓库根目录与当前工作目录下的 .env（不覆盖已存在环境变量）。"""
    roots = [REPO_ROOT, Path.cwd()]
    seen: set[Path] = set()
    for root in roots:
        try:
            p = (root / ".env").resolve()
        except OSError:
            continue
        if p in seen:
            continue
        seen.add(p)
        load_env_file(p)


def openclaw_json_meta(blob: dict[str, Any]) -> dict[str, Any]:
    m = blob.get("meta")
    if isinstance(m, dict):
        return m
    res = blob.get("result")
    if isinstance(res, dict):
        m2 = res.get("meta")
        if isinstance(m2, dict):
            return m2
    return {}


def openclaw_payload_text(blob: dict[str, Any]) -> str:
    """拼接 result.payloads[].text，以双换行连接（供断言）。"""
    parts: list[str] = []
    res = blob.get("result")
    if isinstance(res, dict):
        payloads = res.get("payloads")
        if isinstance(payloads, list):
            for p in payloads:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    t = p["text"].strip()
                    if t:
                        parts.append(t)
    return "\n\n".join(parts)


def openclaw_payload_text_len(blob: dict[str, Any]) -> int:
    return len(openclaw_payload_text(blob))


def run_openclaw_agent_once(
    openclaw_bin: str,
    agent_id: str,
    session_id: str,
    message: str,
    timeout_sec: int,
    *,
    local: bool = False,
    thinking: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[float, int, dict[str, Any], str]:
    """执行一次 openclaw agent --json。返回 (wall_s, exit_code, parsed_json, stderr)。

    model：非空时追加 CLI ``--model``（``provider/id`` 或配置中的模型 key），与 openclaw.json 中
    ``agents.defaults.models`` 的条目一致，例如 ``qwen/qwen3.6-plus``。
    """
    cmd: list[str] = [
        openclaw_bin,
        "agent",
        "--agent",
        agent_id,
        "--session-id",
        session_id,
        "--message",
        message,
        "--json",
        "--timeout",
        str(max(0, int(timeout_sec))),
    ]
    if local:
        cmd.append("--local")
    if thinking and thinking.strip():
        cmd.extend(["--thinking", thinking.strip()])
    mo = (model or "").strip()
    if mo:
        cmd.extend(["--model", mo])
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout_sec) + 120.0 if timeout_sec > 0 else 7_200.0,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        return wall, 124, {}, "subprocess.TimeoutExpired"
    wall = time.perf_counter() - t0
    raw = (proc.stdout or "").strip()
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return wall, int(proc.returncode), parsed, (proc.stderr or "").strip()
