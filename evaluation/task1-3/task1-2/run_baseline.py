import requests
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ================= 配置区域 =================
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 请确保这些 tag 和你 `ollama list` 里的一致
OLLAMA_MODELS = {
    "Llama-3.1-70B": "llama3.1:70b",    
}

KG_FILE = "step-03.jsonl"
TEST_FILES = {
    "disambiguation": "test1_disambiguation.json",
    "pathways": "test2_pathways.json"
}
# ===========================================

# === 1. LLM 调用函数 (基于你的提供修改) ===
def call_llm(prompt, model_tag):
    """
    使用 Ollama 原生 API 调用模型
    """
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": model_tag,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.1,   
            "num_predict": 128    # 这里稍微改大了一点，防止 Task 2 解释被截断
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['message']['content'].strip()
        
    except requests.exceptions.ConnectionError:
        print("❌ 错误: 无法连接到 Ollama。请确保已运行 'ollama serve'")
        return "Error"
    except Exception as e:
        print(f"❌ API 调用错误: {e}")
        return "Error"

# === 2. 简易 RAG 引擎 ===
class SimpleRAG:
    """把 step03.txt 当作纯文本文档库来检索"""
    def __init__(self, kg_file):
        print("Loading RAG Database (Embedding Step 3 triplets)...")
        # 使用轻量级模型进行检索匹配
        self.encoder = SentenceTransformer('/mnt/gpu04_data/wangjie/KGC/all-mpnet-base-v2') 
        self.docs = []
        
        # 读取图谱文件，将其展平为文本列表
        with open(kg_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # 将三元组拼接成句子: "A treats B"
                    text = f"{d['s']} {d['p']} {d['o']}"
                    self.docs.append(text)
                except:
                    pass
        
        print(f"Encoding {len(self.docs)} triplets...")
        self.doc_embeddings = self.encoder.encode(self.docs, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

    def retrieve(self, query, k=5):
        """检索 Top-K 相关的三元组文本"""
        query_emb = self.encoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.doc_embeddings, top_k=k)[0]
        # 拼接检索结果
        context = "\n".join([f"- {self.docs[hit['corpus_id']]}" for hit in hits])
        return context

# === 3. 评测任务逻辑 ===

def eval_task1_disambiguation(model_name, model_tag, rag, use_rag=False):
    """Task 1: 实体消歧/链接"""
    mode = "RAG" if use_rag else "ZeroShot"
    print(f"\n--- Running Task 1: {model_name} ({mode}) ---")
    
    with open(TEST_FILES['disambiguation'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in tqdm(data, desc="Disambiguating"):
        mention = item['mention']
        target = item['target_entity']
        
        # 构建 Prompt
        context_str = ""
        if use_rag:
            retrieved_info = rag.retrieve(mention, k=5)
            context_str = f"Reference Knowledge:\n{retrieved_info}\n"
            
        prompt = (
            f"{context_str}"
            f"Question: What is the standard canonical medical term for the mention '{mention}'?\n"
            f"Answer with the exact entity name only."
        )
        
        # 调用 LLM
        pred = call_llm(prompt, model_tag)
        
        # 判定逻辑：宽松匹配 (Target 是否出现在回答中)
        is_hit = False
        if target.lower() in pred.lower():
            is_hit = True
            
        results.append({
            "query": mention,
            "target": target,
            "pred": pred,
            "correct": is_hit
        })
        
    # 保存结果
    df = pd.DataFrame(results)
    acc = df['correct'].mean()
    print(f"Accuracy: {acc:.2%}")
    df.to_csv(f"baseline_task1_{model_name}_{mode}.csv", index=False)

def eval_task2_pathways(model_name, model_tag, rag, use_rag=False):
    """Task 2: 多跳路径推理"""
    mode = "RAG" if use_rag else "ZeroShot"
    print(f"\n--- Running Task 2: {model_name} ({mode}) ---")
    
    with open(TEST_FILES['pathways'], 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = []
    
    for item in tqdm(data, desc="Reasoning"):
        start = item['start_entity']
        end = item['end_entity']
        gt_path = item.get('ground_truth_path', [])
        # 提取中间节点作为验证标准 (去掉头尾)
        middle_nodes = gt_path[1:-1] if len(gt_path) > 2 else []
        
        # 构建 Prompt
        context_str = ""
        if use_rag:
            # 模拟 RAG：分别检索起点和终点，试图拼凑链路
            query = f"What is the physiological or causal mechanism connecting {start} and {end}?"
            context = rag.retrieve(query, k=3)
            context_str = f"Reference Knowledge:\n{context}\n"
            
        prompt = (
            f"{context_str}"
            f"Question: Briefly explain the physiological or causal mechanism connecting '{start}' and '{end}'.\n"
            f"Provide a step-by-step chain of reasoning."
        )
        
        # 调用 LLM
        pred = call_llm(prompt, model_tag)
        
        # 判定逻辑：是否命中中间节点
        is_hit = False
        if not middle_nodes:
            is_hit = True # 没有中间节点的短路径，只要有回答就算过
        else:
            # 至少提到2个关键中间节点就算成功
            hit_count = 0
            for mid in middle_nodes:
                if mid.lower() in pred.lower():
                    hit_count += 1

            if not middle_nodes:
                is_hit = True
            else:
                is_hit = hit_count >= min(2, len(middle_nodes))
        
        results.append({
            "start": start,
            "end": end,
            "pred": pred,
            "success": is_hit
        })
        
    # 保存结果
    df = pd.DataFrame(results)
    rate = df['success'].mean()
    print(f"Success Rate: {rate:.2%}")
    df.to_csv(f"baseline_task2_{model_name}_{mode}.csv", index=False)

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 准备 RAG 数据库 (加载 step03.txt 并 Embedding)
    # 这一步只需要做一次
    rag_engine = SimpleRAG(KG_FILE)
    
    # 2. 遍历所有配置的模型
    for display_name, ollama_tag in OLLAMA_MODELS.items():
        print(f"\n🚀 Evaluating Model: {display_name} [{ollama_tag}]")
        
        # --------------------
        # 运行 Task 1 (Disambiguation)
        # --------------------
        # Zero-shot (纯靠模型内隐知识)
        #eval_task1_disambiguation(display_name, ollama_tag, rag_engine, use_rag=False)
        # RAG (给检索到的 step03 文本片段)
        #eval_task1_disambiguation(display_name, ollama_tag, rag_engine, use_rag=True)
        
        # --------------------
        # 运行 Task 2 (Pathways)
        # --------------------
        # Zero-shot
        eval_task2_pathways(display_name, ollama_tag, rag_engine, use_rag=False)
        # RAG
        eval_task2_pathways(display_name, ollama_tag, rag_engine, use_rag=True)

    print("\n✅ All baseline evaluations completed.")