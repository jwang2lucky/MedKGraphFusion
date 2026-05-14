import asyncio
import os
import json
import random
import pandas as pd
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
import logging
import networkx as nx  # 用于 fallback 加载

# === 启用日志 ===
logging.basicConfig(level=logging.INFO)

# === 配置 ===
TEXT_FILE = "all_texts.txt"
OUTPUT_DIR = "./nano_graphrag_cache"
OUTPUT_CSV = "results_graphrag_nano.csv"
LLM_LOG_FILE = os.path.join(OUTPUT_DIR, "llm_raw_outputs.log")  # 新增：保存 LLM 原始输出

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_texts(path, limit=None):
    with open(path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    if limit:
        return texts[:limit]
    return texts

# === LLM with logging ===
from openai import OpenAI
OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = "EMPTY"
MODEL_NAME = "qwen2.5:7b"
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

async def ollama_llm(prompt: str, **kwargs) -> str:
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
            timeout=120,
        )
        content = response.choices[0].message.content.strip()

        # 🔥 关键：保存原始 prompt 和 response 到日志文件
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PROMPT (first 300 chars):\n")
            f.write(prompt[:300].replace('\n', '\\n') + ("..." if len(prompt) > 300 else "") + "\n")
            f.write("-" * 40 + "\n")
            f.write("RESPONSE:\n")
            f.write(content + "\n\n")

        return content
    except Exception as e:
        error_msg = f"⚠️ LLM error: {e}"
        print(error_msg)
        # 也记录错误
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR:\n")
            f.write(error_msg + "\n\n")
        return ""

# === Embedding ===
from sentence_transformers import SentenceTransformer
print("🔄 Loading local embedding model...")
emb_model = SentenceTransformer("/home/wangjie/KGC/all-mpnet-base-v2")

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=384)
async def local_mpnet_embedding(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: emb_model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
    )
    return [emb.tolist() for emb in embeddings]

# === 主函数 ===
async def main():
    print("Initializing Nano-GraphRAG with LLM output logging...")

    rag = GraphRAG(
        working_dir=OUTPUT_DIR,
        best_model_func=ollama_llm,
        cheap_model_func=ollama_llm,
        embedding_func=local_mpnet_embedding,
        enable_naive_rag=False,
    )

    texts = load_texts(TEXT_FILE, limit=20)
    print(f"✅ Loaded {len(texts)} texts.")

    # 尝试构建图 —— 允许最后一步失败
    try:
        await rag.ainsert(texts)
        print("✅ ainsert completed successfully.")
    except Exception as e:
        print(f"❌ ainsert failed at final step: {e}")
        print("ℹ️ Checking if graph was already saved...")

    # 检查图是否已生成（关键！）
    graph_path = os.path.join(OUTPUT_DIR, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graph_path):
        raise RuntimeError("Graph file not found. Entity extraction likely failed early.")

    # 加载图（优先用 rag 对象，失败则手动加载）
    try:
        G = rag.chunk_entity_relation_graph
    except:
        print("⚠️ Failed to load via rag object, loading manually with NetworkX...")
        G = nx.read_graphml(graph_path)

    print(f"📊 Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 提取三元组
    all_triplets = []
    for u, v, data in G.edges(data=True):
        description = data.get('description', 'related_to')
        all_triplets.append({
            "head": u,
            "relation": "related_to",
            "tail": v,
            "description": description
        })

    sampled = random.sample(all_triplets, min(100, len(all_triplets)))
    df = pd.DataFrame(sampled)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"💾 Saved {len(sampled)} triplets to {OUTPUT_CSV}")
    print(f"📄 LLM raw outputs saved to: {LLM_LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())