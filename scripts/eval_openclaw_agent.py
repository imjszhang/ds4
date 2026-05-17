#!/usr/bin/env python3
"""OpenClaw Agent 能力回归：读 JSON 题库，逐题 `openclaw agent --json`，断言与统计。

支持单轮 `prompt` 与同会话多轮 `turns`（共享 `session-id`，判据默认作用于末轮输出；
`max_wall_s` 对多轮为各轮墙钟之和）。可选 ``--model`` 覆盖当轮推理模型（对齐 ``openclaw agent --model``）。
与直连 Chat HTTP 基准无关。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from openclaw_agent_common import (  # noqa: E402
    load_dotenv_candidates,
    openclaw_json_meta,
    openclaw_payload_text,
    run_openclaw_agent_once,
)


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    m = n // 2
    if n % 2:
        return float(s[m])
    return (float(s[m - 1]) + float(s[m])) / 2.0


def _p90(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = int(math.ceil(0.9 * len(s))) - 1
    return float(s[max(0, min(idx, len(s) - 1))])


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise SystemExit(f"题库须为 JSON 数组: {path}")
    return data  # type: ignore[return-value]


def _validate_task(t: Any, idx: int) -> dict[str, Any]:
    if not isinstance(t, dict):
        raise SystemExit(f"题目 #{idx} 须为对象")
    tid = t.get("id")
    prompt = t.get("prompt")
    expect = t.get("expect")
    if not isinstance(tid, str) or not tid.strip():
        raise SystemExit(f"题目 #{idx} 缺少非空 id")
    if not isinstance(expect, dict):
        raise SystemExit(f"题目 {tid!r} 缺少 expect 对象")

    turns = t.get("turns")
    if turns is not None:
        if not isinstance(turns, list) or len(turns) < 1:
            raise SystemExit(f"题目 {tid!r}: turns 须为非空数组")
        if isinstance(prompt, str) and prompt.strip():
            raise SystemExit(f"题目 {tid!r}: 已提供 turns 时不要同时使用顶层 prompt")
        for j, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise SystemExit(f"题目 {tid!r}: turns[{j}] 须为对象")
            tp = turn.get("prompt")
            if not isinstance(tp, str) or not tp.strip():
                raise SystemExit(f"题目 {tid!r}: turns[{j}] 缺少非空 prompt")
    else:
        if not isinstance(prompt, str) or not prompt.strip():
            raise SystemExit(f"题目 {tid!r} 缺少非空 prompt（且无 turns）")
    return t  # type: ignore[return-value]


def _check_expect(
    expect: dict[str, Any],
    *,
    exit_code: int,
    wall_s: float,
    out_text: str,
    blob: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True

    if expect.get("exit_zero", True) is not False:
        if exit_code != 0:
            ok = False
            reasons.append(f"exit_code={exit_code} (want 0)")

    max_wall = expect.get("max_wall_s")
    if isinstance(max_wall, (int, float)) and max_wall >= 0:
        if wall_s > float(max_wall):
            ok = False
            reasons.append(f"wall_s={wall_s:.3f} > max_wall_s={max_wall}")

    text = out_text.strip()
    subs = expect.get("output_contains")
    if isinstance(subs, list):
        for sub in subs:
            if not isinstance(sub, str):
                ok = False
                reasons.append("output_contains 含非字符串项")
                continue
            if sub and sub not in text:
                ok = False
                reasons.append(f"输出未包含 {sub!r}")

    pat = expect.get("output_regex")
    if isinstance(pat, str) and pat.strip():
        try:
            if not re.search(pat, text, re.DOTALL):
                ok = False
                reasons.append(f"output_regex 不匹配: {pat!r}")
        except re.error as e:
            ok = False
            reasons.append(f"output_regex 无效: {e}")

    min_chars = expect.get("min_output_chars")
    if isinstance(min_chars, int) and min_chars >= 0:
        if len(text) < min_chars:
            ok = False
            reasons.append(f"输出长度 {len(text)} < min_output_chars {min_chars}")

    if expect.get("no_embedded_fallback") is True:
        meta = openclaw_json_meta(blob)
        if meta.get("transport") == "embedded" or meta.get("fallbackFrom") == "gateway":
            ok = False
            reasons.append("embedded/fallback 禁止使用但出现")

    return ok, reasons


def main() -> None:
    p = argparse.ArgumentParser(description="OpenClaw Agent JSON 题库回归")
    p.add_argument("--tasks", type=Path, required=True, help="JSON 题库路径（数组）")
    p.add_argument(
        "--openclaw-bin",
        default=os.environ.get("OPENCLAW_BIN") or "openclaw",
        help="默认同环境变量 OPENCLAW_BIN 或 openclaw",
    )
    p.add_argument("--agent", default="main")
    p.add_argument("--session-prefix", default="ds4-eval")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--local", action="store_true")
    p.add_argument("--thinking", default="", help="全局默认 --thinking（题目未覆盖时）")
    p.add_argument(
        "--model",
        default="",
        help="传给 CLI --model（如 qwen/qwen3.6-plus），覆盖网关/会话默认模型；空则省略",
    )
    p.add_argument("--report-json", type=Path, default=None, help="写入机器可读报告")
    p.add_argument("--no-dotenv", action="store_true", help="不加载 .env")
    args = p.parse_args()

    if not args.no_dotenv:
        load_dotenv_candidates()

    if not args.tasks.is_file():
        raise SystemExit(f"题库不存在: {args.tasks}")

    raw_tasks: list[dict[str, Any]] = []
    for i, item in enumerate(_load_tasks(args.tasks)):
        raw_tasks.append(_validate_task(item, i))

    bin_path = str(args.openclaw_bin).strip()
    agent_id = str(args.agent).strip()
    prefix = str(args.session_prefix).strip() or "ds4-eval"
    global_thinking = str(args.thinking).strip() or None
    timeout_sec = max(0, int(args.timeout))
    model_override = str(args.model).strip() or None

    report_tasks: list[dict[str, Any]] = []
    total_runs = 0
    passed_runs = 0
    suite_failed = False

    mo_note = f" | model={model_override!r}" if model_override else ""
    print(f"[eval] tasks={args.tasks} | agent={agent_id} | prefix={prefix} | timeout={timeout_sec}s{mo_note}")

    for t in raw_tasks:
        tid = str(t["id"])
        prompt = str(t.get("prompt", "") or "")
        expect = t["expect"]
        if not isinstance(expect, dict):
            raise SystemExit(f"{tid}: expect 非法")
        repeat = t.get("repeat", 1)
        if not isinstance(repeat, int) or repeat < 1:
            raise SystemExit(f"{tid}: repeat 须为 >=1 的整数")

        task_thinking = t.get("thinking")
        if isinstance(task_thinking, str) and task_thinking.strip():
            th = task_thinking.strip()
        else:
            th = global_thinking

        walls: list[float] = []
        runs_out: list[dict[str, Any]] = []
        task_pass_count = 0

        for r in range(repeat):
            safe_id = "".join(c if c.isalnum() or c in "-_" else "-" for c in tid)[:48]
            session_id = f"{prefix}-{safe_id}-{uuid.uuid4().hex[:12]}"

            turns_spec = t.get("turns")
            if isinstance(turns_spec, list) and len(turns_spec) > 0:
                wall_total = 0.0
                turn_details: list[dict[str, Any]] = []
                text = ""
                last_blob: dict[str, Any] = {}
                last_err = ""
                last_code = 0
                failed_turn_idx: int | None = None

                for ti, turn in enumerate(turns_spec):
                    assert isinstance(turn, dict)
                    tp = str(turn["prompt"]).strip()
                    wall, code, blob, err = run_openclaw_agent_once(
                        bin_path,
                        agent_id,
                        session_id,
                        tp,
                        timeout_sec,
                        local=bool(args.local),
                        thinking=th,
                        model=model_override,
                    )
                    wall_total += wall
                    last_code = code
                    last_blob = blob
                    last_err = err
                    text = openclaw_payload_text(blob)
                    turn_details.append(
                        {
                            "turn_index": ti + 1,
                            "wall_s": round(wall, 6),
                            "exit_code": code,
                            "output_excerpt": (text[:400] + ("…" if len(text) > 400 else "")),
                        }
                    )
                    if code != 0:
                        failed_turn_idx = ti + 1
                        break

                if failed_turn_idx is not None and expect.get("all_turns_exit_zero", True) is not False:
                    passed = False
                    reasons = [f"第 {failed_turn_idx} 轮 exit_code={last_code}（要求各轮成功）"]
                else:
                    passed, reasons = _check_expect(
                        expect, exit_code=last_code, wall_s=wall_total, out_text=text, blob=last_blob
                    )

                total_runs += 1
                if passed:
                    passed_runs += 1
                    task_pass_count += 1
                else:
                    suite_failed = True

                meta = openclaw_json_meta(last_blob)
                fb = meta.get("transport") == "embedded" or meta.get("fallbackFrom") == "gateway"

                run_row = {
                    "attempt": r + 1,
                    "session_id": session_id,
                    "multi_turn": True,
                    "turns": turn_details,
                    "wall_s": round(wall_total, 6),
                    "exit_code": last_code,
                    "passed": passed,
                    "fail_reasons": reasons,
                    "fallback_warning": fb,
                    "output_excerpt": (text[:500] + ("…" if len(text) > 500 else "")),
                    "stderr_excerpt": (last_err[-800:] if last_err else ""),
                }
                walls.append(wall_total)
                runs_out.append(run_row)

                st = "OK" if passed else "FAIL"
                print(
                    f"  [{st}] {tid} attempt {r + 1}/{repeat} "
                    f"wall={wall_total:.3f}s turns={len(turn_details)} exit={last_code}"
                )
                if not passed:
                    for rr in reasons:
                        print(f"       - {rr}")
                if fb:
                    print("       ! meta 提示 embedded/fallback（默认不判失败）")
                continue

            wall, code, blob, err = run_openclaw_agent_once(
                bin_path,
                agent_id,
                session_id,
                prompt,
                timeout_sec,
                local=bool(args.local),
                thinking=th,
                model=model_override,
            )
            walls.append(wall)
            text = openclaw_payload_text(blob)
            passed, reasons = _check_expect(
                expect, exit_code=code, wall_s=wall, out_text=text, blob=blob
            )
            total_runs += 1
            if passed:
                passed_runs += 1
                task_pass_count += 1
            else:
                suite_failed = True

            meta = openclaw_json_meta(blob)
            fb = meta.get("transport") == "embedded" or meta.get("fallbackFrom") == "gateway"

            run_row = {
                "attempt": r + 1,
                "session_id": session_id,
                "wall_s": round(wall, 6),
                "exit_code": code,
                "passed": passed,
                "fail_reasons": reasons,
                "fallback_warning": fb,
                "output_excerpt": (text[:500] + ("…" if len(text) > 500 else "")),
                "stderr_excerpt": (err[-800:] if err else ""),
            }
            runs_out.append(run_row)

            st = "OK" if passed else "FAIL"
            print(f"  [{st}] {tid} attempt {r + 1}/{repeat} wall={wall:.3f}s exit={code}")
            if not passed:
                for rr in reasons:
                    print(f"       - {rr}")
            if fb:
                print("       ! meta 提示 embedded/fallback（默认不判失败）")

        report_tasks.append(
            {
                "id": tid,
                "repeat": repeat,
                "passed_attempts": task_pass_count,
                "wall_median_s": round(_median(walls), 6),
                "wall_p90_s": round(_p90(walls), 6),
                "runs": runs_out,
            }
        )

    print("-" * 48)
    print(f"合计 runs: {total_runs} | 通过: {passed_runs} | 失败: {total_runs - passed_runs}")
    for block in report_tasks:
        print(
            f"  {block['id']}: pass {block['passed_attempts']}/{block['repeat']} "
            f"| wall p50 {_median([r['wall_s'] for r in block['runs']]):.3f}s "
            f"p90 {_p90([r['wall_s'] for r in block['runs']]):.3f}s"
        )

    if args.report_json:
        summary: dict[str, Any] = {
            "total_runs": total_runs,
            "passed_runs": passed_runs,
            "all_passed": not suite_failed,
        }
        if model_override:
            summary["model"] = model_override
        out = {"tasks": report_tasks, "summary": summary}
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[eval] report written: {args.report_json}")

    if suite_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
