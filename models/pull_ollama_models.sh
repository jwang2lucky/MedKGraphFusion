#!/usr/bin/bash

set -u

MODELS=(
  "llama3.1:8b"
  "llama3.1:70b"
  "qwen2.5:14b"
  "qwen2.5:7b"
  "qwen2.5:72b"
  "gemma3:12b"
  "gemma3:8b"
)

INTERVAL="1m"
LOG_FILE="ollama_pull.log"

pull_with_timeout() {
  local model="$1"

  echo "==============================" | tee -a "$LOG_FILE"
  echo "开始处理模型: $model" | tee -a "$LOG_FILE"
  echo "==============================" | tee -a "$LOG_FILE"

  while true; do
    echo "[$(date '+%F %T')] 执行: timeout $INTERVAL ollama pull $model" | tee -a "$LOG_FILE"
    
    timeout --signal=SIGTERM --kill-after=5s "$INTERVAL" ollama pull "$model" 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}

    if ollama list | awk '{print $1}' | grep -Fxq "$model"; then
      echo "[$(date '+%F %T')] 模型下载完成: $model" | tee -a "$LOG_FILE"
      break
    fi

    if [[ $status -eq 124 ]]; then
      echo "[$(date '+%F %T')] 超时中断，准备继续断点续传: $model" | tee -a "$LOG_FILE"
    else
      echo "[$(date '+%F %T')] pull 返回状态码: $status，准备重试: $model" | tee -a "$LOG_FILE"
    fi

    sleep 3
  done
}

for model in "${MODELS[@]}"; do
  pull_with_timeout "$model"
done

echo "全部模型处理完成。" | tee -a "$LOG_FILE"