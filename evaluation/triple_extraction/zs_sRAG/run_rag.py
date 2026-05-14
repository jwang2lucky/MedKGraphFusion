import json
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import call_llm, load_relations, load_pairs

# === 文件路径 ===
TEXT_FILE = "all_texts.txt"  # 你的所有原始文本合并文件
PAIRS_FILE = "ht.json"
RELATION_FILE = "relation_types_n.json"
OUTPUT_FILE = "results_rag.csv"

# === RAG 配置 ===
TOP_K = 3  # 检索最相关的 3 段文本
EMBEDDING_MODEL = "/home/wangjie/KGC/all-mpnet-base-v2" # 一个轻量级且效果好的标准 baseline 模型

class RAGEngine:
    def __init__(self, text_path, model_name):
        print("Loading Embedding Model...")
        self.encoder = SentenceTransformer(model_name)
        
        print("Loading texts and building index...")
        with open(text_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]
        
        # 编码所有文本 (如果文本量巨大，这里可能需要分批处理)
        embeddings = self.encoder.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)
        
        # 建立 FAISS 索引
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Inner Product (Cosine similarity if normalized)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Index built with {len(self.texts)} documents.")

    def retrieve(self, query, k=TOP_K):
        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        
        retrieved_docs = []
        for idx in I[0]:
            if idx < len(self.texts):
                retrieved_docs.append(self.texts[idx])
        return retrieved_docs

def generate_rag_prompt(head, tail, context_list, relation_desc):
    context_text = "\n".join([f"[{i+1}] {txt}" for i, txt in enumerate(context_list)])
    return f"""
You are a medical expert assistant.
Task: Determine the relationship between "{head}" and "{tail}" based PRIMARILY on the provided Context.

Context:
{context_text}

Allowed Relations:
{relation_desc}
- No_Relation: If the context does not support any specific relationship.

Output Requirement:
Output ONLY the relation label. Do not explain.

Answer:
"""

def main():
    # 1. 初始化 RAG 引擎
    rag = RAGEngine(TEXT_FILE, EMBEDDING_MODEL)
    
    # 2. 加载数据
    pairs = load_pairs(PAIRS_FILE)
    rel_desc, valid_rels = load_relations(RELATION_FILE)
    
    results = []
    print(f"Starting RAG evaluation on {len(pairs)} pairs...")
    
    # 3. 循环预测
    for item in tqdm(pairs):
        h = item['s']
        t = item['o']
        
        # 构造检索 Query: 简单把两个实体拼起来通常就很有效
        query = f"{h} {t} relationship medical context"
        
        # 检索
        retrieved_docs = rag.retrieve(query)
        
        # 生成
        prompt = generate_rag_prompt(h, t, retrieved_docs, rel_desc)
        pred_relation = call_llm(prompt)
        
        # 清洗结果
        pred_relation = pred_relation.replace("Relationship:", "").strip().split('\n')[0]
        
        results.append({
            "head": h,
            "tail": t,
            "predicted_relation": pred_relation,
            "context_used": retrieved_docs, # 保存上下文以备查验
            "method": "Standard RAG"
        })
        
    # 4. 保存
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
