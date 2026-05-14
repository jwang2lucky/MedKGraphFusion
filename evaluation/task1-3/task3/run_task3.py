import json
import pandas as pd
from tqdm import tqdm
import requests
import os
from sentence_transformers import SentenceTransformer, util
import torch

# === 配置 ===
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:70b"
KG_FILE = "step03.txt"
TEST_CASES_FILE = "test3_cases_custom.json"
OUTPUT_CSV = "results_task3_graphrag_n.csv"

# === 核心：GraphRAG 检索引擎 (Vector + Structure) ===
class GraphRAGEngine:
    def __init__(self, kg_file):
        print("Loading Knowledge Graph...")
        self.encoder = SentenceTransformer('/mnt/gpu04_data/wangjie/KGC/all-mpnet-base-v2')
        self.triplets = []    # 存原始三元组文本
        self.triplet_objs = [] # 存结构化对象 {'s':, 'p':, 'o':}
        self.adj = {}         # 邻接表，用于结构化扩展
        
        with open(kg_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    s, p, o = d.get('s'), d.get('p'), d.get('o')
                    if isinstance(s, str) and isinstance(o, str):
                        text = f"{s} {p} {o}"
                        self.triplets.append(text)
                        self.triplet_objs.append(d)
                        
                        # 构建邻接表
                        if s not in self.adj: self.adj[s] = []
                        if o not in self.adj: self.adj[o] = []
                        self.adj[s].append(f"{s} --{p}--> {o}")
                        self.adj[o].append(f"{o} <--{p}-- {s}") # 双向
                except: pass
        
        print(f"Encoding {len(self.triplets)} facts for Vector Index...")
        # 1. 建立向量索引 (这一步跟 RAG 一样)
        if self.triplets:
            self.embeddings = self.encoder.encode(self.triplets, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
        else:
            self.embeddings = None

    def retrieve(self, query, top_k=10):
        """
        混合检索策略：
        1. 先用向量检索找到 Top-K 最相关的三元组 (Anchor Knowledge)
        2. 再把这些三元组里的实体作为锚点，去图里找它们的邻居 (Structural Expansion)
        """
        if self.embeddings is None: return ""
        
        # Step 1: Vector Search (语义匹配)
        query_emb = self.encoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)[0]
        
        retrieved_facts = set()
        anchor_entities = set()
        
        for hit in hits:
            idx = hit['corpus_id']
            fact_text = self.triplets[idx]
            retrieved_facts.add(fact_text)
            
            # 记录锚点实体
            obj = self.triplet_objs[idx]
            anchor_entities.add(obj['s'])
            anchor_entities.add(obj['o'])
            
        # Step 2: Structural Expansion (结构扩展 - 这是 RAG 做不到的)
        # 这一步能找回 "隐形" 的联系
        structure_facts = set()
        for entity in list(anchor_entities)[:5]: # 限制扩展数量，防爆炸
            neighbors = self.adj.get(entity, [])
            for n in neighbors[:3]: # 每个锚点只扩充 3 个邻居
                structure_facts.add(n)
        
        # 合并证据
        final_evidence = "--- Direct Semantic Matches ---\n"
        final_evidence += "\n".join(list(retrieved_facts))
        final_evidence += "\n\n--- Structural Context (Reasoning) ---\n"
        final_evidence += "\n".join(list(structure_facts))
        
        return final_evidence

# === LLM 调用 ===
def call_llm(messages):
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512}
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=data).json()
        return resp['message']['content'].strip()
    except: return "Error"

# === 主流程 ===
if __name__ == "__main__":
    with open(TEST_CASES_FILE, 'r') as f: cases = json.load(f)
    engine = GraphRAGEngine(KG_FILE) # 它是 RAG 和 Graph 的合体
    
    results = []
    print("🚀 Starting GraphRAG Pipeline...")
    
    for item in tqdm(cases):
        text = item.get('text', item.get('description', ''))
        expected = item.get('expected_diagnosis', '')
        if not text: continue
        
        # 1. Baseline
        p_zs = call_llm([{"role": "user", "content": f"Case: {text}\nDiagnose the patient concisely."}])
        
        # 2. RAG (纯向量，作为对比列，虽然我们现在不需要它来跑分了，但为了表格完整)
        # 这里简单复用 engine 的第一步逻辑作为 RAG
        emb_hits = engine.retrieve(text, top_k=10).split('--- Structural')[0] # 只取前半部分
        p_rag = call_llm([{"role": "user", "content": f"Context:\n{emb_hits}\n\nCase: {text}\nDiagnose."}])
        
        # 3. Ours (GraphRAG: Vector + Structure)
        evidence = engine.retrieve(text, top_k=10)
        #Task: Provide a definitive diagnosis.
        prompt_ours = f"""
        You are a medical expert assisted by a Knowledge Graph.
        
        [Graph Evidence]:
        {evidence}
        
        [Patient Case]:
        {text}
        
        Task:Provide a definitive diagnosis and discuss potential differential diagnoses briefly to cover all possibilities
        
        Instructions:
        1. **Semantic Matches** show direct hits. **Structural Context** shows hidden connections. Use BOTH.
        2. Explicitly cite the evidence (e.g., "The graph links X to Y").
        3. If the evidence matches the case symptoms, trust it.
        4. Output format: Diagnosis first, then reasoning.
        """
        p_ours = call_llm([{"role": "user", "content": prompt_ours}])
        
        results.append({
            "case_id": item.get('id'),
            "case_text": text,
            "expected": expected,
            "prediction_baseline": p_zs,
            "prediction_rag": p_rag,
            "prediction_ours": p_ours,
            "graph_evidence": evidence
        })
        
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved GraphRAG results to {OUTPUT_CSV}")