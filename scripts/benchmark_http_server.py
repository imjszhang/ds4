#!/usr/bin/env python3
"""
对 OpenAI 兼容的 Chat Completions HTTP 端点做几项性能取样（本地 ds4-server 或 DeepSeek 官方等）。

说明：
  • ./ds4-bench 绕过 HTTP：要测引擎本身请用它；这里测 HTTP + JSON + 网络路由。
  • ds4-server 多为单 worker 串行推理，并发通常不提高总 tok/s。
  • 对比官方时请设置 DEEPSEEK_API_KEY，并加 --compare-official。
  • DeepSeek 官方同样支持顶层 JSON 字段 thinking: {\"type\":\"disabled\"}；默认不再从官方请求中删掉它，以便与本地 --no-thinking 对齐（遇 400 可用 --strip-thinking-official）。

对齐口径：
  • 使用 --align 预设短输出或可预期答案；当 completion_tokens 仍不一致时，以「ms/输出 token」为主看延迟。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

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

    nons_wall_s: float = float("nan")
    nons_prompt_avg: float = float("nan")
    nons_completion_avg: float = float("nan")
    nons_tps_e2e: float = float("nan")
    nons_ms_per_out_e2e: float = float("nan")
    nons_finish_reasons: list[str] = field(default_factory=list)


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
    for _ in range(args.warmup):
        try:
            post_stream_sse(chat_url, ps_warm, args.timeout, headers=headers)
        except Exception:
            pass

    print(f"[对齐] align={args.align} | prompt_chars={len(user_message)} | model={model} | max_tokens={args.max_tokens} | repeat={args.repeat}")

    acc_t: list[float] = []
    acc_ttft: list[float] = []
    acc_tps_decode: list[float] = []
    acc_tps_e2e: list[float] = []
    acc_pt: list[int] = []
    acc_ct: list[int] = []
    acc_ms_e2e: list[float] = []
    acc_ms_dec: list[float] = []

    stream_fr: list[str] = []

    for _ in range(args.repeat):
        total, ttft, pt, ct, fr = post_stream_sse(chat_url, payload_stream, args.timeout, headers=headers)
        acc_t.append(total)
        acc_pt.append(int(pt or 0))
        ct_i = int(ct or 0)
        acc_ct.append(ct_i)
        if ttft is not None:
            acc_ttft.append(ttft)
        if ct_i > 0:
            acc_tps_e2e.append(ct_i / total)
            ms_e = total * 1000.0 / ct_i
            acc_ms_e2e.append(ms_e)
        else:
            acc_ms_e2e.append(float("nan"))

        if ttft is not None and total - ttft > 0.05 and ct_i > 0:
            dec = ct_i / (total - ttft)
            acc_tps_decode.append(dec)
            acc_ms_dec.append((total - ttft) * 1000.0 / ct_i)
        if fr:
            stream_fr.append(fr)

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

    for _ in range(args.repeat):
        t0, robj = post_json(chat_url, payload_complete, args.timeout, headers=headers)
        acc_nt.append(t0)
        u = robj.get("usage") or {}
        acc_npt.append(int(u.get("prompt_tokens") or 0))
        acc_nct.append(int(u.get("completion_tokens") or 0))
        ch0 = (robj.get("choices") or [{}])[0]
        fr_ns = ch0.get("finish_reason")
        if fr_ns:
            nons_fr.append(fr_ns)

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
    print(
        "\n口径：ms/output_tok 在 completion 条数不同时仍可比较「每输出 token 摊到的墙钟时间」。"
        "decode 列为 (wall−TTFT) 粗分。"
    )


def main() -> None:
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
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompt", default="用两句话解释什么是哈希表。")
    p.add_argument("--prompt-file", default="")
    p.add_argument("--prompt-chars", type=int, default=200_000)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument(
        "--compare-official",
        action="store_true",
        help="在完成本地测试后，再测 DeepSeek 官方（需在环境中设置 DEEPSEEK_API_KEY）",
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
    args = p.parse_args()

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
        "\n[对比解读] tok/s、e2e 受网络 RTT（官方）、thinking、stop 时机与后端实现差异影响。"
        "ms/output_tok（e2e）在 completion_tokens 不同时比裸 tok/s 更可比。"
    )


if __name__ == "__main__":
    main()
