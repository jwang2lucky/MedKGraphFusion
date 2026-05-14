# llm_client.py
"""
统一的 Ollama LLM 调用封装，使用 /api/chat 接口（stream=False）。
其他模块统一从此处 import call_llm()。
"""
from __future__ import annotations

import json
import logging
import re
import time

import requests

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
)

logger = logging.getLogger(__name__)


def call_llm(
    user_prompt: str,
    system_prompt: str = "You are a biomedical knowledge graph expert. Always output valid JSON.",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    调用本地 Ollama 模型。

    参数:
        user_prompt  : 用户侧 prompt
        system_prompt: 系统侧 prompt
        model        : 覆盖默认模型
        temperature  : 覆盖默认温度
        max_tokens   : 覆盖默认 num_predict
        retries      : 失败重试次数
        retry_delay  : 重试间隔（秒）

    返回:
        模型输出的纯文本字符串（已去除首尾空白）

    异常:
        RuntimeError: 多次重试后仍失败
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "num_predict": max_tokens or LLM_MAX_TOKENS,
        },
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"].strip()
            return content
        except Exception as e:
            last_error = e
            logger.warning(
                f"[LLM] Attempt {attempt}/{retries} failed: {e}. "
                f"Retrying in {retry_delay}s ..."
            )
            time.sleep(retry_delay)

    raise RuntimeError(
        f"[LLM] All {retries} attempts failed. Last error: {last_error}"
    )


def extract_json(text: str) -> dict | list | None:
    """
    从 LLM 输出文本中提取第一个合法 JSON 对象或数组。
    兼容 markdown 代码块包裹的情况。
    """
    # 去除 markdown 代码块
    text = re.sub(r"```(?:json)?", "", text).strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取第一个 {...} 或 [...]
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    return None