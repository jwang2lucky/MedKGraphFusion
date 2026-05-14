import json
import re
import requests
import networkx as nx
import numpy as np
import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 配置区域
# =========================
OLLAMA_API_URL = "http://localhost:11434/api/chat"

OLLAMA_MODELS = {
    "Llama-3.1-70B": "llama3.1:70b",
}

GRAPH_FILE = "step-03.jsonl"
TEST_DISAMBIGUATION = "test1_disambiguation.json"
TEST_PATHWAYS = "test2_pathways.json"
EMBEDDING_MODEL = "/mnt/gpu04_data/wangjie/KGC/all-mpnet-base-v2"

# =========================
# 1. LLM 调用
# =========================
def call_llm(prompt, model_tag):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_tag,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 128
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print("❌ 错误: 无法连接到 Ollama，请确认已运行 `ollama serve`")
        return "Error"
    except Exception as e:
        print(f"❌ API 调用错误: {e}")
        return "Error"


# =========================
# 2. 简易 RAG 引擎
# =========================
class SimpleRAG:
    """把 step-03.jsonl 中的 triplets 当作文档库来做检索"""
    def __init__(self, kg_file, embedding_model):
        print("Loading RAG Database (Embedding triplets)...")
        self.encoder = SentenceTransformer(embedding_model)
        self.docs = []

        with open(kg_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    text = f"{d['s']} {d['p']} {d['o']}"
                    self.docs.append(text)
                except Exception:
                    continue

        print(f"Encoding {len(self.docs)} triplets for RAG...")
        self.doc_embeddings = self.encoder.encode(
            self.docs,
            batch_size=64,
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def retrieve(self, query, k=5):
        query_emb = self.encoder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.doc_embeddings, top_k=k)[0]
        context = "\n".join([f"- {self.docs[hit['corpus_id']]}" for hit in hits])
        return context


# =========================
# 3. 图谱方法
# =========================
class MedicalGraph:
    def __init__(self, graph_path, embedding_model_name):
        print("Loading Graph...")
        self.G = nx.DiGraph()
        self.nodes = set()
        self.triplets = []

        with open(graph_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    s, p, o = data['s'], data['p'], data['o']
                    self.G.add_edge(s, o, relation=p, origins=data.get('origins', []))
                    self.nodes.add(s)
                    self.nodes.add(o)
                    self.triplets.append(data)
                except Exception:
                    continue

        print(f"Graph Loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")

        print("Loading Embedding Model for node search...")
        self.encoder = SentenceTransformer(embedding_model_name)

        print("Encoding all graph nodes...")
        self.node_list = list(self.nodes)
        self.node_embeddings = self.encoder.encode(self.node_list, show_progress_bar=True)

    def search_node(self, query, top_k=1):
        query_emb = self.encoder.encode([query])
        scores = cosine_similarity(query_emb, self.node_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.node_list[idx], scores[idx]))
        return results

    def find_path(self, start, end):
        try:
            path = nx.shortest_path(self.G, source=start, target=end)
            explanation = []
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                edge_data = self.G.get_edge_data(u, v)
                rel = edge_data.get('relation', 'related_to')
                explanation.append(f"{u} --[{rel}]--> {v}")
            return path, explanation
        except nx.NetworkXNoPath:
            return None, []
        except nx.NodeNotFound:
            return None, ["One of the nodes is not in the graph."]


# =========================
# 4. 统一的 Task 2 评估函数
# =========================
def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip().lower()

def count_middle_node_hits(pred_text, middle_nodes):
    """
    统一标准：
    - exact string inclusion
    - 统计命中的 gt middle nodes 数量
    """
    pred_norm = normalize_text(pred_text)
    hit_nodes = []

    for mid in middle_nodes:
        mid_norm = normalize_text(mid)
        if mid_norm and mid_norm in pred_norm:
            hit_nodes.append(mid)

    return len(hit_nodes), hit_nodes

def evaluate_reference_middle_node_hit(pred_text, gt_path):
    """
    统一 Task 2 标准：
    success iff hit_count >= min(2, len(middle_nodes))
    """
    middle_nodes = gt_path[1:-1] if len(gt_path) > 2 else []

    if not middle_nodes:
        return True, 0, [], 0

    hit_count, hit_nodes = count_middle_node_hits(pred_text, middle_nodes)
    required_hits = min(2, len(middle_nodes))
    success = hit_count >= required_hits

    return success, hit_count, hit_nodes, required_hits


# =========================
# 5. Baseline: Task 1
# =========================
def eval_baseline_task1_disambiguation(model_name, model_tag, rag, use_rag=False):
    mode = "RAG" if use_rag else "ZeroShot"
    print(f"\n--- Running Baseline Task 1: {model_name} ({mode}) ---")

    with open(TEST_DISAMBIGUATION, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc=f"Baseline Task1 {mode}"):
        mention = item['mention']
        target = item['target_entity']

        context_str = ""
        if use_rag:
            retrieved_info = rag.retrieve(mention, k=5)
            context_str = f"Reference Knowledge:\n{retrieved_info}\n"

        prompt = (
            f"{context_str}"
            f"Question: What is the standard canonical medical term for the mention '{mention}'?\n"
            f"Answer with the exact entity name only."
        )

        pred = call_llm(prompt, model_tag)
        is_hit = target.lower() in pred.lower()

        results.append({
            "query": mention,
            "target": target,
            "pred": pred,
            "correct": is_hit
        })

    df = pd.DataFrame(results)
    acc = df["correct"].mean()
    print(f"Task 1 Accuracy: {acc:.2%}")
    df.to_csv(f"baseline_task1_{model_name}_{mode}.csv", index=False)


# =========================
# 6. Baseline: Task 2
# =========================
def eval_baseline_task2_pathways(model_name, model_tag, rag, use_rag=False):
    mode = "RAG" if use_rag else "ZeroShot"
    print(f"\n--- Running Baseline Task 2: {model_name} ({mode}) ---")
    print("Metric: Reference Middle-node Hit Rate (>= 2 gt middle nodes)")

    with open(TEST_PATHWAYS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc=f"Baseline Task2 {mode}"):
        start = item['start_entity']
        end = item['end_entity']
        gt_path = item.get('ground_truth_path', [])

        context_str = ""
        if use_rag:
            query = f"What is the physiological or causal mechanism connecting {start} and {end}?"
            context = rag.retrieve(query, k=3)
            context_str = f"Reference Knowledge:\n{context}\n"

        prompt = (
            f"{context_str}"
            f"Question: Briefly explain the physiological or causal mechanism connecting '{start}' and '{end}'.\n"
            f"Provide a step-by-step chain of reasoning."
        )

        pred = call_llm(prompt, model_tag)

        success, hit_count, hit_nodes, required_hits = evaluate_reference_middle_node_hit(pred, gt_path)

        results.append({
            "start": start,
            "end": end,
            "gt_path": " -> ".join(gt_path),
            "pred": pred,
            "required_hits": required_hits,
            "hit_count": hit_count,
            "hit_nodes": " | ".join(hit_nodes),
            "success": success
        })

    df = pd.DataFrame(results)
    rate = df["success"].mean()
    print(f"Task 2 Reference Middle-node Hit Rate: {rate:.2%}")
    df.to_csv(f"baseline_task2_{model_name}_{mode}.csv", index=False)


# =========================
# 7. Ours: Task 1
# =========================
def eval_ours_task1_disambiguation(kg):
    print("\n=== Running Ours Task 1: Entity Disambiguation ===")

    with open(TEST_DISAMBIGUATION, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct = 0
    total = len(data)
    results = []

    for item in tqdm(data, desc="Ours Task1"):
        mention = item['mention']
        target = item['target_entity']

        prediction_list = kg.search_node(mention, top_k=1)
        predicted_node = prediction_list[0][0]
        score = prediction_list[0][1]

        is_correct = predicted_node.lower() == target.lower()
        if is_correct:
            correct += 1

        results.append({
            "query": mention,
            "target": target,
            "predicted": predicted_node,
            "score": score,
            "correct": is_correct
        })

    acc = correct / total
    print(f"Task 1 Accuracy: {acc:.2%} ({correct}/{total})")
    pd.DataFrame(results).to_csv("ours_task1_disambiguation.csv", index=False)


# =========================
# 8. Ours: Task 2
# =========================
def eval_ours_task2_pathways(kg):
    print("\n=== Running Ours Task 2: Multi-hop Reasoning ===")
    print("Metric: Reference Middle-node Hit Rate (>= 2 gt middle nodes)")

    with open(TEST_PATHWAYS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc="Ours Task2"):
        start = item['start_entity']
        end = item['end_entity']
        gt_path = item.get('ground_truth_path', [])

        path, explanation = kg.find_path(start, end)
        found = path is not None
        path_str = " -> ".join(path) if path else "No Path"
        explanation_str = " | ".join(explanation)

        # 用统一文本进行 middle-node matching
        pred_text = f"{path_str} | {explanation_str}"

        success, hit_count, hit_nodes, required_hits = evaluate_reference_middle_node_hit(pred_text, gt_path)

        results.append({
            "start": start,
            "end": end,
            "gt_path": " -> ".join(gt_path),
            "graph_path_found": found,
            "pred_path": path_str,
            "pred_explanation": explanation_str,
            "required_hits": required_hits,
            "hit_count": hit_count,
            "hit_nodes": " | ".join(hit_nodes),
            "success": success
        })

    df = pd.DataFrame(results)
    rate = df["success"].mean()
    print(f"Task 2 Reference Middle-node Hit Rate (Ours): {rate:.2%}")
    df.to_csv("ours_task2_pathways.csv", index=False)


# =========================
# 9. 汇总结果
# =========================
def summarize_results():
    print("\n=== Summary Convention ===")
    print("Task 2 metric name: Reference Middle-node Hit Rate")
    print("Success criterion: hit_count >= min(2, len(gt_middle_nodes))")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    print("===== Initializing Resources =====")
    rag_engine = SimpleRAG(GRAPH_FILE, EMBEDDING_MODEL)
    my_kg = MedicalGraph(GRAPH_FILE, EMBEDDING_MODEL)

    print("\n===== Running Baselines =====")
    for display_name, ollama_tag in OLLAMA_MODELS.items():
        print(f"\n🚀 Evaluating Baseline Model: {display_name} [{ollama_tag}]")

        # Task 1
        eval_baseline_task1_disambiguation(display_name, ollama_tag, rag_engine, use_rag=False)
        eval_baseline_task1_disambiguation(display_name, ollama_tag, rag_engine, use_rag=True)

        # Task 2
        eval_baseline_task2_pathways(display_name, ollama_tag, rag_engine, use_rag=False)
        eval_baseline_task2_pathways(display_name, ollama_tag, rag_engine, use_rag=True)

    print("\n===== Running Ours =====")
    eval_ours_task1_disambiguation(my_kg)
    eval_ours_task2_pathways(my_kg)

    summarize_results()
    print("\n✅ All evaluations completed.")