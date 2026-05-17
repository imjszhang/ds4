#!/usr/bin/env bash
# Mac Studio 512GB + q4-imatrix 推荐启动（Metal，本机监听）。
#   端口 8005；上下文默认 1M（模型标称上限）；单 worker；失败时可降 DS4_CTX/DS4_WORKERS。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${DS4_HOST:=127.0.0.1}"
: "${DS4_PORT:=8005}"
: "${DS4_CTX:=1048576}"
: "${DS4_DEFAULT_TOKENS:=393216}"
: "${DS4_WORKERS:=1}"
: "${DS4_KV_DISK_DIR:=$HOME/Library/Caches/ds4-server-kv}"
: "${DS4_KV_DISK_SPACE_MB:=98304}"
LOGICAL_CPUS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 16)"
if ! [[ "$LOGICAL_CPUS" =~ ^[0-9]+$ ]] || [ "$LOGICAL_CPUS" -lt 1 ]; then
  LOGICAL_CPUS=16
fi
THREADS="$LOGICAL_CPUS"
if [ "$THREADS" -gt 24 ]; then THREADS=24; fi

MODEL="${DS4_MODEL:-ds4flash.gguf}"
mkdir -p "$DS4_KV_DISK_DIR"

exec ./ds4-server \
  --metal \
  --model "$MODEL" \
  --host "$DS4_HOST" \
  --port "$DS4_PORT" \
  --ctx "$DS4_CTX" \
  --tokens "$DS4_DEFAULT_TOKENS" \
  --threads "$THREADS" \
  --workers "$DS4_WORKERS" \
  --warm-weights \
  --kv-disk-dir "$DS4_KV_DISK_DIR" \
  --kv-disk-space-mb "$DS4_KV_DISK_SPACE_MB" \
  --kv-cache-cold-max-tokens 100000 \
  --tool-memory-max-ids 200000 \
  "$@"
