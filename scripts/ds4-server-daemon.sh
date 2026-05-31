#!/usr/bin/env bash
# 脱离终端静默运行 ds4-server（Cursor/SSH 关闭后仍存活）。
# 用法: ./scripts/ds4-server-daemon.sh {start|stop|status|restart} [传给启动脚本的额外参数...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SCRIPT="${DS4_START_SCRIPT:-$ROOT/scripts/ds4-server-mac-studio-512g.sh}"
STATE_DIR="${DS4_STATE_DIR:-$HOME/Library/Application Support/ds4-server}"
PIDFILE="${DS4_PIDFILE:-$STATE_DIR/ds4-server.pid}"
LOGFILE="${DS4_LOGFILE:-$HOME/Library/Logs/ds4-server.log}"

mkdir -p "$STATE_DIR" "$(dirname "$LOGFILE")"

read_pid() {
  if [[ -f "$PIDFILE" ]]; then
    tr -d '[:space:]' <"$PIDFILE"
  fi
}

is_running() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cmd_start() {
  local pid
  pid="$(read_pid || true)"
  if is_running "$pid"; then
    echo "ds4-server 已在运行 (pid $pid)，日志: $LOGFILE"
    exit 0
  fi
  [[ -f "$PIDFILE" ]] && rm -f "$PIDFILE"

  if ! [[ -x "$START_SCRIPT" ]]; then
    echo "启动脚本不可执行: $START_SCRIPT" >&2
    exit 1
  fi

  # nohup + 新会话，避免收到终端 SIGHUP
  nohup bash "$START_SCRIPT" "$@" >>"$LOGFILE" 2>&1 </dev/null &
  pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" >"$PIDFILE"

  echo "ds4-server 已在后台启动 (pid $pid)"
  echo "  日志: $LOGFILE"
  echo "  停止: $ROOT/scripts/ds4-server-daemon.sh stop"
}

cmd_stop() {
  local pid
  pid="$(read_pid || true)"
  if ! is_running "$pid"; then
    rm -f "$PIDFILE"
    echo "ds4-server 未在运行"
    exit 0
  fi
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    is_running "$pid" || break
    sleep 1
  done
  if is_running "$pid"; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
  echo "ds4-server 已停止"
}

cmd_status() {
  local pid
  pid="$(read_pid || true)"
  if is_running "$pid"; then
    echo "running  pid=$pid  log=$LOGFILE"
    lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep -E "ds4-serve.*$pid" || true
  else
    echo "stopped"
    [[ -f "$PIDFILE" ]] && rm -f "$PIDFILE"
    exit 1
  fi
}

case "${1:-start}" in
  start)
    shift || true
    cmd_start "$@"
    ;;
  stop)
    cmd_stop
    ;;
  status)
    cmd_status
    ;;
  restart)
    shift || true
    cmd_stop || true
    cmd_start "$@"
    ;;
  *)
    echo "用法: $0 {start|stop|status|restart} [args...]" >&2
    exit 2
    ;;
esac
