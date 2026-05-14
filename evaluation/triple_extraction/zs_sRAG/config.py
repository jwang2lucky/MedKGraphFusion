import requests
import json
import time

# === 配置区域 ===
# Ollama 默认地址
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# ⚠️ 注意：请确保这里写的名字和你终端里运行 `ollama list` 看到的名字完全一致
# Ollama 中通常是冒号分隔，例如 "qwen2.5:14b"
MODEL_NAME = "qwen2.5:14b"  

def call_llm(prompt, model=MODEL_NAME):
    """
    使用 Ollama 原生 API 调用模型
    """
    headers = {"Content-Type": "application/json"}
    
    # Ollama 的请求格式
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,  # 必须关闭流式输出，否则返回的是一堆 JSON片段
        "options": {
            "temperature": 0.1,   # 低温度保证结果稳定
            "num_predict": 50     # 对应 OpenAI 的 max_tokens，限制输出长度
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        # Ollama 的返回格式解析
        result = response.json()
        return result['message']['content'].strip()
        
    except requests.exceptions.ConnectionError:
        print("❌ 错误: 无法连接到 Ollama。请确保已运行 'ollama serve'")
        return "Error"
    except Exception as e:
        print(f"❌ API 调用错误: {e}")
        return "Error"

def load_relations(path):
    """
    加载关系定义文件
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 格式化关系描述供 Prompt 使用
        rel_text = ""
        valid_labels = []
        for key, val in data.items():
            rel_text += f"- {val['label']}: {val['description']}\n"
            valid_labels.append(val['label'])
        return rel_text, valid_labels
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {path}")
        return "", []

def load_pairs(path):
    """
    加载实体对文件 (兼容 json list 和 jsonl)
    """
    pairs = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
                
            try:
                # 尝试作为整个列表读取 [{}, {}]
                pairs = json.loads(content)
            except json.JSONDecodeError:
                # 尝试逐行读取 (jsonl)
                f.seek(0)
                for line in f:
                    if line.strip():
                        pairs.append(json.loads(line))
        return pairs
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {path}")
        return []