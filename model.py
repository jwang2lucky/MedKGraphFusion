from typing_extensions import TypedDict, Annotated, List
from pydantic import BaseModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, Any
from langchain_core.language_models.llms import LLM
import json
import os
import re
import time
import requests

# ------------------ 辅助函数 ------------------
def extract_triples_from_text(text: str):
    """
    从模型输出中提取 triples 列表。
    示例输入:
    [{"s": "EGFR", "p": "treats", "o": "cancer"}]
    """
    try:
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.S)
        if not match:
            return [{"s": "None", "p": "None", "o": "ParseError"}]

        triples = json.loads(match.group())
        if isinstance(triples, list):
            return triples
        else:
            return [{"s": "None", "p": "None", "o": "ParseError"}]
    except Exception as e:
        print("解析异常:", e)
        return [{"s": "None", "p": "None", "o": "ParseError"}]


# ------------------ TypedDict & BaseModel ------------------
class Triple(TypedDict):
    s: Annotated[str, ..., "Subject of the extracted Knowledge Graph Triple"]
    p: Annotated[str, ..., "Relation of the extracted Knowledge Graph Triple"]
    o: Annotated[str, ..., "Object of the extracted Knowledge Graph Triple"]


class Triples(BaseModel):
    triples: List[Triple]


# ------------------ KnowledgeGraphLLM (Ollama 版本) ------------------
class KnowledgeGraphLLM(LLM):
    model_name: str
    max_tokens: int
    ollama_host: str = "http://localhost:11434" 

    def __init__(
        self,
        model_name: str = "llama3.1:20b",  # 注意：Ollama 中模型名通常是 llama3:70b 或 llama3.3:70b
        max_tokens: int = 200,
        ollama_host: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model_name=model_name, max_tokens=max_tokens, ollama_host=ollama_host, **kwargs)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.ollama_host = ollama_host

        # 测试连接
        try:
            resp = requests.get(f"{self.ollama_host}/")
            if resp.status_code != 200:
                raise ConnectionError("Ollama server returned non-200 status")
        except Exception as e:
            raise RuntimeError(f"无法连接到 Ollama 服务 ({self.ollama_host})，请先运行 'ollama serve' 或确保服务已启动: {e}")

    @property
    def _llm_type(self) -> str:
        return f"KnowledgeGraphLLM (Ollama) - {self.model_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            # 🔹 调用 Ollama /api/chat 接口
            url = f"{self.ollama_host}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a biomedical knowledge graph triple extractor. Please output a JSON array of triples."},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "temperature": 0.0,
                    "num_predict": self.max_tokens,
                },
                "stream": False,
            }

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            result = response.json()
            text = result["message"]["content"]
            print("\n=== Raw Model Output ===\n", text)
        except Exception as e:
            print("Ollama 调用异常:", e)
            text = ""

        # 可选：加一点延迟避免本地 GPU 过载（通常不需要）
        # time.sleep(1)

        # 🔹 提取 triples
        triples = extract_triples_from_text(text)
        return json.dumps(triples, ensure_ascii=False).replace("\n", "")