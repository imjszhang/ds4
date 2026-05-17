#!/usr/bin/env python3
"""
对 OpenAI 兼容的 Chat Completions HTTP 端点做几项性能取样（本地 ds4-server 或 DeepSeek 官方等）。

说明：
  • ./ds4-bench 绕过 HTTP：要测引擎本身请用它；这里测 HTTP + JSON + 网络路由。
  • 多 worker 时请用 --parallel N 同时发起 N 个请求；单 worker 时并发多在服务端排队。
  • 对比官方时请设置 DEEPSEEK_API_KEY（或仓库根目录 / 当前目录的 .env），并加 --compare-official。
  • DeepSeek 官方同样支持顶层 JSON 字段 thinking: {\"type\":\"disabled\"}；默认不再从官方请求中删掉它，以便与本地 --no-thinking 对齐（遇 400 可用 --strip-thinking-official）。

对齐口径：
  • 使用 --align 预设短输出或可预期答案；当 completion_tokens 仍不一致时，以「ms/输出 token」为主看延迟。

OpenClaw（可选，**目的不同**）：
  • --openclaw：在 LOCAL HTTP 之后调用 openclaw CLI（默认经 Gateway），用于 **Agent 整轮能力/路径**（稳定性、延迟量级、payload 等），**不是**与上文中「直连官方 API」做同一类对比。
  • --openclaw-model：传给 `openclaw agent --model`（空则使用配置默认主模型）。
  • 每轮独立会话：--session-id 为 ds4-bench-<uuid>（explicit），不占用 main 主会话。
  • 需已安装 openclaw，Gateway 可用（除非 --openclaw-local）。详见 OpenClaw 文档 openclaw agent。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from openclaw_agent_common import (
    load_dotenv_candidates,
    openclaw_json_meta as _openclaw_json_meta,
    openclaw_payload_text_len as _openclaw_payload_text_len,
    run_openclaw_agent_once,
)

Headers = Mapping[str, str]


def post_json(url: str, payload: dict, timeout: float, headers: Headers | None = None) -> tuple[float, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    h = dict(headers or {})
    h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, method="POST", headers=h)
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - start
    return elapsed, json.loads(raw.decode("utf-8"))


def delta_first_emit_time(delta: dict) -> bool:
    """任一可见流式载荷（content / reasoning 等）即视为已开始吐 token."""
    if not isinstance(delta, dict):
        return False
    c = delta.get("content")
    if isinstance(c, str) and c != "":
        return True
    rc = delta.get("reasoning_content")
    if isinstance(rc, str) and rc != "":
        return True
    tc = delta.get("tool_calls")
    if isinstance(tc, list) and len(tc) > 0:
        return True
    return False


def post_stream_sse(
    url: str,
    payload: dict,
    timeout: float,
    headers: Headers | None = None,
) -> tuple[float, Optional[float], Optional[int], Optional[int], Optional[str]]:
    """返回 (wall_s, ttft_s, prompt_tokens, completion_tokens, finish_reason最后一个)"""
    body = json.dumps(payload).encode("utf-8")
    h = dict(headers or {})
    h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, method="POST", headers=h)

    ttft: Optional[float] = None
    usage_prompt: Optional[int] = None
    usage_completion: Optional[int] = None
    finish_reason: Optional[str] = None
    overall_start = time.perf_counter()
    ttft_anchor = overall_start

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            buf += chunk
            while True:
                line_end = buf.find(b"\n")
                if line_end < 0:
                    break
                line = buf[:line_end].decode("utf-8", errors="replace").strip()
                buf = buf[line_end + 1 :]
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    elapsed = time.perf_counter() - overall_start
                    return elapsed, ttft, usage_prompt, usage_completion, finish_reason
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                for ch in choices:
                    delta = ch.get("delta") or {}
                    if delta_first_emit_time(delta):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - ttft_anchor
                        break
                    fr = ch.get("finish_reason")
                    if fr:
                        finish_reason = fr
                u = obj.get("usage")
                if isinstance(u, dict):
                    usage_prompt = u.get("prompt_tokens")
                    usage_completion = u.get("completion_tokens")
        elapsed = time.perf_counter() - overall_start
        return elapsed, ttft, usage_prompt, usage_completion, finish_reason


def read_prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        if args.prompt_chars > 0:
            txt = txt[: args.prompt_chars]
        return txt
    return args.prompt


def timing_get(url: str, timeout: float, headers: Headers | None = None) -> tuple[float, int]:
    h = dict(headers or {})
    req = urllib.request.Request(url, method="GET", headers=h)
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        n = len(resp.read())
    return time.perf_counter() - start, n


def strip_thinking_only(pl: MutableMapping[str, Any]) -> dict[str, Any]:
    d = dict(pl)
    d.pop("thinking", None)
    return d


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class SuiteAgg:
    label: str
    models_latency_ms: float = 0.0
    models_bytes: int = 0

    stream_wall_s: float = float("nan")
    stream_ttft_s: float = float("nan")
    stream_prompt_avg: float = float("nan")
    stream_completion_avg: float = float("nan")
    stream_tps_e2e: float = float("nan")
    stream_tps_decode: float = float("nan")
    stream_ms_per_out_e2e: float = float("nan")
    stream_ms_per_out_decode: float = float("nan")
    stream_finish_reasons: list[str] = field(default_factory=list)
    stream_batch_wall_s: float = float("nan")
    stream_batch_sum_tok_per_s: float = float("nan")

    nons_wall_s: float = float("nan")
    nons_prompt_avg: float = float("nan")
    nons_completion_avg: float = float("nan")
    nons_tps_e2e: float = float("nan")
    nons_ms_per_out_e2e: float = float("nan")
    nons_finish_reasons: list[str] = field(default_factory=list)
    nons_batch_wall_s: float = float("nan")
    nons_batch_sum_tok_per_s: float = float("nan")


def run_openclaw_benchmark(user_message: str, args: argparse.Namespace) -> None:
    """经 OpenClaw CLI 观测 Agent 整轮：main agent + 每轮独立 explicit session。

    与直连 Chat HTTP 延迟对比无关，仅作 Agent 栈能力/时延样本。
    """
    bin_path = (args.openclaw_bin or "openclaw").strip()
    agent_id = (args.openclaw_agent or "main").strip()
    par = max(1, int(args.parallel))
    timeout_sec = int(max(0, args.timeout))

    if argv_explicit("--openclaw-thinking"):
        thinking_oc = args.openclaw_thinking.strip() or None
        if thinking_oc is None and args.no_thinking:
            thinking_oc = "off"
    elif args.no_thinking:
        thinking_oc = "off"
    else:
        thinking_oc = None

    mode = "嵌入式 --local" if args.openclaw_local else "经 Gateway（CLI 默认）"
    prefix = (args.openclaw_session_prefix or "ds4-bench").strip() or "ds4-bench"
    par_note = f" | parallel={par}" if par > 1 else ""
    oc_model = (args.openclaw_model or "").strip() or None
    mo_note = f" | model={oc_model!r}" if oc_model else ""

    print(f"\n{'=' * 16} OPENCLAW（Agent 路径，非 API 对标）{'=' * 16}")
    print("[目的] 验证经 OpenClaw 的 Agent 整轮表现（能力/稳定性/墙钟），不与上段 LOCAL/OFFICIAL 直连 HTTP 做官方延迟对标。")
    print(f"[openclaw] bin={bin_path!r} | agent={agent_id} | 会话=独立 {prefix}-<uuid>（explicit）| {mode}{par_note}{mo_note}")
    print(f"[对齐] prompt_chars={len(user_message)} | repeat={args.repeat} | timeout_s={timeout_sec}")

    for _ in range(args.openclaw_warmup):
        sid = f"{prefix}-{uuid.uuid4().hex}"
        try:
            run_openclaw_agent_once(
                bin_path,
                agent_id,
                sid,
                user_message,
                timeout_sec,
                local=args.openclaw_local,
                thinking=thinking_oc,
                model=oc_model,
            )
        except Exception:
            pass

    acc_wall: list[float] = []
    acc_meta_ms: list[float] = []
    acc_out_chars: list[int] = []
    acc_exit: list[int] = []
    acc_batch: list[tuple[float, int]] = []
    embed_warn = 0

    def invoke_one(sid: str) -> tuple[float, int, int]:
        nonlocal embed_warn
        wall, code, blob, err = run_openclaw_agent_once(
            bin_path,
            agent_id,
            sid,
            user_message,
            timeout_sec,
            local=args.openclaw_local,
            thinking=thinking_oc,
            model=oc_model,
        )
        if code != 0:
            snippet = (err or "")[-500:]
            print(f"  [openclaw] 非零退出 {code} session={sid!r} stderr(尾): {snippet!r}")
        meta = _openclaw_json_meta(blob)
        if meta.get("transport") == "embedded" or meta.get("fallbackFrom") == "gateway":
            embed_warn += 1
        dm = meta.get("durationMs")
        if isinstance(dm, (int, float)) and dm >= 0:
            acc_meta_ms.append(float(dm))
        ochars = _openclaw_payload_text_len(blob)
        acc_out_chars.append(ochars)
        acc_exit.append(code)
        return wall, code, ochars

    for _ in range(args.repeat):
        if par <= 1:
            sid = f"{prefix}-{uuid.uuid4().hex}"
            w, _, oc = invoke_one(sid)
            acc_wall.append(w)
        else:
            sids = [f"{prefix}-{uuid.uuid4().hex}" for _ in range(par)]
            with ThreadPoolExecutor(max_workers=par) as pool:
                futs = [pool.submit(invoke_one, sid) for sid in sids]
                rows = [fu.result() for fu in futs]
            acc_wall.extend([r[0] for r in rows])
            bw = max(r[0] for r in rows)
            sc = sum(r[2] for r in rows)
            acc_batch.append((bw, sc))

    print("  [openclaw agent 单轮墙钟]")
    print(f"    wall_total_s           {mean(acc_wall):.3f}")
    if acc_meta_ms:
        print(f"    meta.durationMs ~(CLI) {mean(acc_meta_ms):.1f}  （若存在，与 wall 可能略有偏差）")
    och_m = mean([float(x) for x in acc_out_chars])
    print(f"    回复文本字符 ~均值      {och_m:.1f}  （自 JSON payloads 累加，非 API token）")
    print(f"    进程退出码（样本数）    {len(acc_exit)} | 非零次数 {sum(1 for x in acc_exit if x != 0)}")
    if par > 1 and acc_batch:
        mbw = mean([b[0] for b in acc_batch])
        ch_per_s = [sc / bw for bw, sc in acc_batch if bw > 0 and sc > 0]
        if ch_per_s:
            print(f"    batch_wall_s (max)     {mbw:.3f}  （每轮 {par} 路 CLI 并发）")
            print(f"    sum_字符/s ~(batch)    {mean(ch_per_s):.2f}  （payload 字符之和 ÷ batch_wall）")
    if embed_warn:
        print(
            f"  ⚠ 有 {embed_warn} 次响应 meta 显示 embedded/fallback；"
            "见 OpenClaw 文档（Gateway 不可用时 CLI 会回落本地）。"
        )


def run_named_suite(
    title: str,
    base: str,
    model: str,
    user_message: str,
    extra: dict[str, Any],
    args: argparse.Namespace,
    *,
    headers: Headers | None = None,
    strip_thinking_for_official: bool = False,
    print_models: bool = True,
) -> SuiteAgg:
    out = SuiteAgg(label=title)
    chat_url = f"{base}/chat/completions"
    models_url = f"{base}/models"

    extra_use = strip_thinking_only(extra) if strip_thinking_for_official else dict(extra)

    align_common: dict[str, Any] = {
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    payload_stream = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **align_common,
        **extra_use,
    }
    payload_complete = dict(payload_stream)
    payload_complete["stream"] = False
    payload_complete.pop("stream_options", None)

    if args.align == "digit-answer":
        payload_stream["stop"] = ["\n", "\r"]
        payload_complete["stop"] = ["\n", "\r"]

    print(f"\n{'=' * 16} {title} {'=' * 16}")
    if print_models:
        try:
            t_models, sz = timing_get(models_url, min(30.0, args.timeout), headers=headers)
            out.models_latency_ms = t_models * 1000
            out.models_bytes = sz
            print(f"[GET /v1/models] {t_models*1000:.1f} ms, body {sz} bytes")
        except urllib.error.HTTPError as e:
            print(f"[GET /v1/models] HTTP {e.code}")
        except urllib.error.URLError as e:
            print(f"[GET /v1/models] 失败: {e}")

    ps_warm = payload_stream.copy()
    par = max(1, int(args.parallel))

    def _warm_one_stream() -> None:
        try:
            post_stream_sse(chat_url, ps_warm, args.timeout, headers=headers)
        except Exception:
            pass

    for _ in range(args.warmup):
        if par <= 1:
            _warm_one_stream()
        else:
            with ThreadPoolExecutor(max_workers=par) as pool:
                futs = [
                    pool.submit(post_stream_sse, chat_url, ps_warm, args.timeout, headers) for _ in range(par)
                ]
                for fu in futs:
                    try:
                        fu.result()
                    except Exception:
                        pass

    par_note = f" | parallel={par}" if par > 1 else ""
    print(
        f"[对齐] align={args.align} | prompt_chars={len(user_message)} | "
        f"model={model} | max_tokens={args.max_tokens} | repeat={args.repeat}{par_note}"
    )

    acc_t: list[float] = []
    acc_ttft: list[float] = []
    acc_tps_decode: list[float] = []
    acc_tps_e2e: list[float] = []
    acc_pt: list[int] = []
    acc_ct: list[int] = []
    acc_ms_e2e: list[float] = []
    acc_ms_dec: list[float] = []

    stream_fr: list[str] = []
    acc_stream_batch: list[tuple[float, int]] = []

    def _absorb_stream_row(total: float, ttft: Optional[float], pt: Optional[int], ct: Optional[int], fr: Optional[str]) -> None:
        acc_t.append(total)
        acc_pt.append(int(pt or 0))
        ct_i = int(ct or 0)
        acc_ct.append(ct_i)
        if ttft is not None:
            acc_ttft.append(ttft)
        if ct_i > 0:
            acc_tps_e2e.append(ct_i / total)
            acc_ms_e2e.append(total * 1000.0 / ct_i)
        else:
            acc_ms_e2e.append(float("nan"))
        if ttft is not None and total - ttft > 0.05 and ct_i > 0:
            acc_tps_decode.append(ct_i / (total - ttft))
            acc_ms_dec.append((total - ttft) * 1000.0 / ct_i)
        if fr:
            stream_fr.append(fr)

    for _ in range(args.repeat):
        if par <= 1:
            total, ttft, pt, ct, fr = post_stream_sse(chat_url, payload_stream, args.timeout, headers=headers)
            _absorb_stream_row(total, ttft, pt, ct, fr)
        else:
            with ThreadPoolExecutor(max_workers=par) as pool:
                futs = [
                    pool.submit(post_stream_sse, chat_url, payload_stream, args.timeout, headers)
                    for _ in range(par)
                ]
                rows = [fu.result() for fu in futs]
            bw = max(r[0] for r in rows)
            sct = sum(int(r[3] or 0) for r in rows)
            acc_stream_batch.append((bw, sct))
            for total, ttft, pt, ct, fr in rows:
                _absorb_stream_row(total, ttft, pt, ct, fr)

    print("  [stream + usage]")
    print(f"    wall_total_s           {mean(acc_t):.3f}")
    if acc_ttft:
        print(f"    TTFT_s                 {mean(acc_ttft):.3f}")
        out.stream_ttft_s = mean(acc_ttft)
    else:
        print("    TTFT_s                 N/A")
    pt_m = mean([float(x) for x in acc_pt])
    ct_m = mean([float(x) for x in acc_ct])
    print(f"    prompt_tokens          {pt_m:.1f}")
    print(f"    completion_tokens      {ct_m:.1f}")
    print(f"    tok/s (e2e)            {mean(acc_tps_e2e):.2f}")
    if acc_tps_decode:
        print(f"    tok/s ~(decode)         {mean(acc_tps_decode):.2f}")
    if acc_ms_e2e:
        me2_clean = [x for x in acc_ms_e2e if x == x]
        if me2_clean:
            m_me2e = mean(me2_clean)
            print(f"    ms/output_tok ~(e2e)    {m_me2e:.2f}  （总耗时 ÷ completion_tokens）")
            out.stream_ms_per_out_e2e = m_me2e
    if acc_ms_dec:
        m_mdec = mean(acc_ms_dec)
        print(f"    ms/output_tok ~(decode) {m_mdec:.2f}  （(wall-TTFT) ÷ completion_tokens）")
        out.stream_ms_per_out_decode = m_mdec
        out.stream_tps_decode = mean(acc_tps_decode)
    if par > 1 and acc_stream_batch:
        mbw = mean([b[0] for b in acc_stream_batch])
        b_tps = [sc / bw for bw, sc in acc_stream_batch if bw > 0 and sc > 0]
        if b_tps:
            m_btps = mean(b_tps)
            print(f"    batch_wall_s (max)     {mbw:.3f}  （每轮 {par} 路并发取最长墙钟）")
            print(f"    sum_tok/s ~(batch)     {m_btps:.2f}  （该轮各请求 completion 之和 ÷ batch_wall）")
            out.stream_batch_wall_s = mbw
            out.stream_batch_sum_tok_per_s = m_btps
    if stream_fr:
        uniq = ",".join(sorted(set(stream_fr)))
        print(f"    finish_reason (stream) [{uniq}]  (各次合计 {len(stream_fr)})")
    out.stream_wall_s = mean(acc_t)
    out.stream_prompt_avg = pt_m
    out.stream_completion_avg = ct_m
    out.stream_tps_e2e = mean(acc_tps_e2e)
    out.stream_finish_reasons = list(stream_fr)

    acc_nt: list[float] = []
    acc_npt: list[int] = []
    acc_nct: list[int] = []
    nons_fr: list[str] = []
    acc_nons_batch: list[tuple[float, int]] = []

    for _ in range(args.repeat):
        if par <= 1:
            t0, robj = post_json(chat_url, payload_complete, args.timeout, headers=headers)
            acc_nt.append(t0)
            u = robj.get("usage") or {}
            acc_npt.append(int(u.get("prompt_tokens") or 0))
            acc_nct.append(int(u.get("completion_tokens") or 0))
            ch0 = (robj.get("choices") or [{}])[0]
            fr_ns = ch0.get("finish_reason")
            if fr_ns:
                nons_fr.append(fr_ns)
        else:
            with ThreadPoolExecutor(max_workers=par) as pool:
                futs = [
                    pool.submit(post_json, chat_url, payload_complete, args.timeout, headers)
                    for _ in range(par)
                ]
                rows = [fu.result() for fu in futs]
            bw = max(r[0] for r in rows)
            sct = 0
            for t0, robj in rows:
                acc_nt.append(t0)
                u = robj.get("usage") or {}
                pti = int(u.get("prompt_tokens") or 0)
                cti = int(u.get("completion_tokens") or 0)
                acc_npt.append(pti)
                acc_nct.append(cti)
                sct += cti
                ch0 = (robj.get("choices") or [{}])[0]
                fr_ns = ch0.get("finish_reason")
                if fr_ns:
                    nons_fr.append(fr_ns)
            acc_nons_batch.append((bw, sct))

    print("  [non-stream]")
    print(f"    wall_total_s           {mean(acc_nt):.3f}")
    npt_m = mean([float(x) for x in acc_npt])
    nct_m = mean([float(x) for x in acc_nct])
    print(f"    prompt_tokens          {npt_m:.1f}")
    print(f"    completion_tokens      {nct_m:.1f}")
    if mean(acc_nct) > 0:
        te2 = mean(acc_nct) / mean(acc_nt)
        me2_ns = mean(acc_nt) * 1000.0 / mean(acc_nct)
        print(f"    tok/s (e2e)            {te2:.2f}")
        print(f"    ms/output_tok ~(e2e)   {me2_ns:.2f}")
        out.nons_tps_e2e = te2
        out.nons_ms_per_out_e2e = me2_ns
    if par > 1 and acc_nons_batch:
        mbw = mean([b[0] for b in acc_nons_batch])
        b_tps = [sc / bw for bw, sc in acc_nons_batch if bw > 0 and sc > 0]
        if b_tps:
            m_btps = mean(b_tps)
            print(f"    batch_wall_s (max)     {mbw:.3f}  （每轮 {par} 路并发）")
            print(f"    sum_tok/s ~(batch)     {m_btps:.2f}")
            out.nons_batch_wall_s = mbw
            out.nons_batch_sum_tok_per_s = m_btps
    if nons_fr:
        uniq = ",".join(sorted(set(nons_fr)))
        print(f"    finish_reason          [{uniq}]")

    out.nons_wall_s = mean(acc_nt)
    out.nons_prompt_avg = npt_m
    out.nons_completion_avg = nct_m
    out.nons_finish_reasons = list(nons_fr)

    return out


def argv_explicit(opt_prefix: str) -> bool:
    for a in sys.argv:
        if a == opt_prefix or a.startswith(opt_prefix + "="):
            return True
    return False


def apply_alignment_preset(args: argparse.Namespace) -> None:
    """在 parse 之后改写 max_tokens / prompt / repeat。"""
    if args.align == "off":
        return
    has_pf = argv_explicit("--prompt-file") or bool(args.prompt_file.strip())
    if args.align == "short-tokens":
        if not argv_explicit("--max-tokens"):
            args.max_tokens = 32
        if not has_pf and not argv_explicit("--prompt"):
            args.prompt = "用不超过20个汉字解释：栈和队列的区别。"
    elif args.align == "digit-answer":
        if not argv_explicit("--max-tokens"):
            args.max_tokens = 8
        if not has_pf:
            args.prompt = "只输出两位数阿拉伯数字作答，不要有其它任何字符（不要空格标点换行）。19+35=?"
    if not argv_explicit("--repeat") and args.repeat < 3:
        args.repeat = 3


def print_compare_summary(loc: SuiteAgg | None, off: SuiteAgg | None) -> None:
    if not loc or not off:
        return
    lc = loc.stream_completion_avg
    oc = off.stream_completion_avg
    ratio = abs(lc - oc) / max(lc, oc, 1e-9)
    warn = ""
    if ratio > 0.15:
        warn = f"  ⚠ completion_tokens 相对差异 ~{ratio*100:.1f}%：优先看 ms/output_tok 而非裸 tok/s"
    print(f"\n{'=' * 18} COMPARE（对齐摘要）{'=' * 18}")
    print(warn)
    hdr = f"{'指标':<34} {'LOCAL':>14} {'OFFICIAL':>14}"
    print(hdr)
    print("-" * len(hdr))

    def row(k: str, a: Any, b: Any, fmt: str) -> None:
        def f(x):
            if x is None:
                return "N/A"
            if isinstance(x, float) and (x != x):
                return "N/A"
            return fmt.format(x)

        print(f"{k:<34} {f(a):>14} {f(b):>14}")

    row("TTFT_s (stream avg)", loc.stream_ttft_s, off.stream_ttft_s, "{:.3f}")
    row("completion_tokens (stream avg)", loc.stream_completion_avg, off.stream_completion_avg, "{:.1f}")
    row("tok/s e2e (stream avg)", loc.stream_tps_e2e, off.stream_tps_e2e, "{:.2f}")
    row("ms / out_tok e2e (stream)", loc.stream_ms_per_out_e2e, off.stream_ms_per_out_e2e, "{:.2f}")
    row("tok/s ~(decode)", loc.stream_tps_decode, off.stream_tps_decode, "{:.2f}")
    row("ms / out_tok ~(decode)", loc.stream_ms_per_out_decode, off.stream_ms_per_out_decode, "{:.2f}")
    row("GET /models ms", loc.models_latency_ms, off.models_latency_ms, "{:.1f}")
    row("wall_s non-stream", loc.nons_wall_s, off.nons_wall_s, "{:.3f}")
    row("completion_tokens nons avg", loc.nons_completion_avg, off.nons_completion_avg, "{:.1f}")
    row("ms / out_tok e2e (non-stream)", loc.nons_ms_per_out_e2e, off.nons_ms_per_out_e2e, "{:.2f}")
    row("batch_wall_s (stream·N 并发)", loc.stream_batch_wall_s, off.stream_batch_wall_s, "{:.3f}")
    row("sum_tok/s batch (stream)", loc.stream_batch_sum_tok_per_s, off.stream_batch_sum_tok_per_s, "{:.2f}")
    row("batch_wall_s (nons·N 并发)", loc.nons_batch_wall_s, off.nons_batch_wall_s, "{:.3f}")
    row("sum_tok/s batch (non-stream)", loc.nons_batch_sum_tok_per_s, off.nons_batch_sum_tok_per_s, "{:.2f}")
    print(
        "\n口径：ms/output_tok 在 completion 条数不同时仍可比较「每输出 token 摊到的墙钟时间」。"
        "decode 列为 (wall−TTFT) 粗分。"
    )


def main() -> None:
    load_dotenv_candidates()
    p = argparse.ArgumentParser(description="OpenAI-compatible Chat HTTP 性能取样 / 与本机对照")
    p.add_argument("--base-url", default="http://127.0.0.1:8005/v1", help="本地或任意兼容端点 base（含 /v1）")
    p.add_argument(
        "--model",
        default=os.environ.get("DS4_API_MODEL") or os.environ.get("DEEPSEEK_CHAT_MODEL") or "deepseek-v4-flash",
        help="默认读环境变量 DS4_API_MODEL（或 DEEPSEEK_CHAT_MODEL），再回落 deepseek-v4-flash",
    )
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="每轮同时发起的 chat 请求数（与多 worker 同开时搭配）；默认 1",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompt", default="用两句话解释什么是哈希表。")
    p.add_argument("--prompt-file", default="")
    p.add_argument("--prompt-chars", type=int, default=200_000)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument(
        "--compare-official",
        action="store_true",
        help="在完成本地测试后，再测 DeepSeek 官方（DEEPSEEK_API_KEY 或 .env）",
    )
    p.add_argument(
        "--official-base-url",
        default=os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1"),
        help="官方 OpenAI 兼容 base，默认 DEEPSEEK_API_BASE_URL 或 https://api.deepseek.com/v1",
    )
    p.add_argument("--official-model", default="", help="默认同 --model（多为 deepseek-v4-flash）")
    p.add_argument(
        "--official-api-key-env",
        default="DEEPSEEK_API_KEY",
        help="存放官方 API Key 的环境变量名",
    )
    p.set_defaults(strip_thinking_official=False)
    p.add_argument(
        "--strip-thinking-official",
        action="store_true",
        dest="strip_thinking_official",
        help="仅从发往官方的 JSON 中移除 thinking（遇 HTTP 400 时再试；默认不移除以保持与本地 --no-thinking 一致）。",
    )
    p.add_argument(
        "--align",
        choices=("off", "short-tokens", "digit-answer"),
        default="off",
        help="对齐预设：short-tokens=短补全+demo短问；digit-answer=极小输出、固定算术问法（易被 EOS 卡住时仍可能字数差）。",
    )
    p.add_argument(
        "--openclaw",
        action="store_true",
        help="在 LOCAL HTTP 之后跑 openclaw agent（测 Agent 能力/整轮路径；非扩展与官方的 API 延迟对比）",
    )
    p.add_argument(
        "--openclaw-bin",
        default=os.environ.get("OPENCLAW_BIN", "openclaw"),
        help="openclaw 可执行文件；默认 OPENCLAW_BIN 或 openclaw",
    )
    p.add_argument("--openclaw-agent", default="main", help="传给 CLI 的 --agent，默认 main")
    p.add_argument(
        "--openclaw-model",
        default="",
        help="传给 CLI 的 --model（如 qwen/qwen3.6-plus）；空则使用会话/配置默认模型",
    )
    p.add_argument(
        "--openclaw-local",
        action="store_true",
        help="传入 CLI --local（强制嵌入式，不经 Gateway）",
    )
    p.add_argument(
        "--openclaw-thinking",
        default="",
        help="传给 CLI 的 --thinking；不设而使用 --no-thinking 时默认为 off",
    )
    p.add_argument(
        "--openclaw-session-prefix",
        default="ds4-bench",
        help="--session-id 前缀，完整 id 为 {prefix}-<uuid>（OpenClaw 解析为 explicit 会话）",
    )
    p.add_argument(
        "--openclaw-warmup",
        type=int,
        default=1,
        help="openclaw 预热次数（每次新 session）",
    )
    args = p.parse_args()

    if args.parallel < 1:
        print("benchmark_http_server: --parallel 须为 >= 1", file=sys.stderr)
        sys.exit(2)
    if args.openclaw_warmup < 0:
        print("benchmark_http_server: --openclaw-warmup 须为 >= 0", file=sys.stderr)
        sys.exit(2)

    apply_alignment_preset(args)

    base_local = args.base_url.rstrip("/")
    official_model = args.official_model.strip() if args.official_model.strip() else args.model

    extra: dict = {}
    if args.no_thinking:
        extra["thinking"] = {"type": "disabled"}

    user_message = read_prompt_text(args)
    th = "disabled（与官方同字段）" if args.no_thinking else "未在 JSON 中指定（两端各用服务端默认，多为开启）"
    off_th = "移除 thinking 字段" if args.strip_thinking_official else ("与本地相同" if args.no_thinking else "与本地相同（空）")
    print(f"[数据来源] prompt 字符数={len(user_message)}（{'文件' if args.prompt_file else '参数'}）")
    print(f"[thinking] 本地请求: {th}；官方请求: {off_th}")

    key_name = args.official_api_key_env.strip() or "DEEPSEEK_API_KEY"
    official_key = os.environ.get(key_name, "").strip()
    hdr_official: Headers | None = {"Authorization": f"Bearer {official_key}"} if official_key else None

    local_metrics = run_named_suite(
        "LOCAL",
        base_local,
        args.model,
        user_message,
        extra,
        args,
        headers=None,
        strip_thinking_for_official=False,
    )

    if args.openclaw:
        run_openclaw_benchmark(user_message, args)

    official_metrics: Optional[SuiteAgg] = None

    if args.compare_official:
        if not official_key:
            print(
                f"\n[跳过官方] 未设置环境变量 {key_name}=...（不要将 Key 写入命令行）；\n"
                f"示例： export {key_name}=sk-xxxx\n"
                f"      cd <ds4-repo> && python3 {sys.argv[0]} --compare-official --repeat 3\n",
                flush=True,
            )
            return
        official_base = args.official_base_url.rstrip("/")
        official_metrics = run_named_suite(
            "OFFICIAL (DeepSeek API)",
            official_base,
            official_model,
            user_message,
            extra,
            args,
            headers=hdr_official,
            strip_thinking_for_official=args.strip_thinking_official,
            print_models=True,
        )

        print_compare_summary(local_metrics, official_metrics)

    print(
        "\n[对比解读] LOCAL↔OFFICIAL：直连 Chat 的 tok/s、e2e 受网络 RTT（官方）、thinking、stop 与后端差异影响；"
        "ms/output_tok（e2e）在 completion_tokens 不一致时更可参照。--parallel>1 时的 batch 列为 HTTP 并发专用。"
        "\nOpenClaw：单独观测 Agent 栈，勿与上表做「对官方 API 谁更快」式解读。"
    )


if __name__ == "__main__":
    main()
