import json
import re
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from config import call_llm

# === 配置 ===
GRAPH_FILE = "step-03.jsonl"
TEST_DISAMBIGUATION = "test1_disambiguation.json" 
TEST_PATHWAYS = "test2_pathways.json"             
EMBEDDING_MODEL = "/mnt/gpu04_data/wangjie/KGC/all-mpnet-base-v2" 

class MedicalGraph:
    def __init__(self, graph_path, embedding_model_name):
        print("1. Loading Graph...")
        self.G = nx.DiGraph()
        self.nodes = set()
        self.triplets = []
        
        # 读取 step03.txt (每一行是一个 JSON)
        with open(graph_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    # 假设 step03 的格式是 {"s":..., "p":..., "o":...}
                    s, p, o = data['s'], data['p'], data['o']
                    self.G.add_edge(s, o, relation=p, origins=data.get('origins', []))
                    self.nodes.add(s)
                    self.nodes.add(o)
                    self.triplets.append(data)
                except Exception as e:
                    continue
        
        print(f"   Graph Loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")

        print("2. Loading Embedding Model for Search...")
        self.encoder = SentenceTransformer(embedding_model_name)
        
        # 预计算所有节点的 Embedding，用于 Task 1 检索
        print("   Encoding all graph nodes (this may take a moment)...")
        self.node_list = list(self.nodes)
        self.node_embeddings = self.encoder.encode(self.node_list, show_progress_bar=True)
    def split_pred_to_chunks(self, text):
        """
        将路径说明文本切成片段，用于和 gt middle node 做语义匹配
        """
        if not text:
            return []

        parts = re.split(r'[\n\r\|;:,]+', text)
        chunks = []

        for p in parts:
            p = p.strip()
            if len(p) < 2:
                continue
            chunks.append(p)

            # 对路径表达再按箭头切一下
            subparts = re.split(r'--\[.*?\]-->|<--\[.*?\]--|->', p)
            for sp in subparts:
                sp = sp.strip()
                if len(sp) >= 2:
                    chunks.append(sp)

        chunks = list(dict.fromkeys(chunks))
        return chunks[:30]

    def semantic_match_entity(self, pred_text, target_entity, threshold=0.7):
        """
        判定 pred_text 是否命中 target_entity：
        - exact match
        - or embedding semantic match
        """
        if not pred_text or not target_entity:
            return False, 0.0, ""

        # exact string match
        if target_entity.lower() in pred_text.lower():
            return True, 1.0, target_entity

        chunks = self.split_pred_to_chunks(pred_text)
        if not chunks:
            return False, 0.0, ""

        target_emb = self.encoder.encode(target_entity, convert_to_tensor=True)
        chunk_embs = self.encoder.encode(chunks, convert_to_tensor=True)

        sims = util.cos_sim(target_emb, chunk_embs)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        best_chunk = chunks[best_idx]

        return best_score >= threshold, best_score, best_chunk

    def hit_any_middle_node_semantic(self, pred_text, middle_nodes, threshold=0.7):
        """
        是否命中任意一个 gt middle node 或其语义等价表达
        """
        if not middle_nodes:
            return True, None, 1.0, ""

        best_mid = None
        best_score = 0.0
        best_chunk = ""
        hit = False

        for mid in middle_nodes:
            mid_hit, score, chunk = self.semantic_match_entity(
                pred_text,
                mid,
                threshold=threshold
            )

            if score > best_score:
                best_score = score
                best_mid = mid
                best_chunk = chunk

            if mid_hit:
                hit = True
                best_mid = mid
                best_score = score
                best_chunk = chunk
                break

        return hit, best_mid, best_score, best_chunk    
    def search_node(self, query, top_k=1):
        """
        Task 1 核心功能: 语义检索图谱节点
        """
        query_emb = self.encoder.encode([query])
        # 计算相似度
        scores = cosine_similarity(query_emb, self.node_embeddings)[0]
        # 获取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.node_list[idx], scores[idx]))
        return results

    def find_path(self, start, end):
        """
        Task 2 核心功能: 寻找路径
        """
        try:
            path = nx.shortest_path(self.G, source=start, target=end)
            # 将路径转化为三元组描述
            explanation = []
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                edge_data = self.G.get_edge_data(u, v)
                rel = edge_data.get('relation', 'related_to')
                explanation.append(f"{u} --[{rel}]--> {v}")
            return path, explanation
        except nx.NetworkXNoPath:
            return None, []
        except nx.NodeNotFound:
            return None, ["One of the nodes is not in the graph."]

    def get_neighbors_context(self, entities):
        """
        Task 3 核心功能: 获取实体的邻居作为上下文
        """
        context = []
        for ent in entities:
            # 先找到图谱里对应的标准名
            matches = self.search_node(ent, top_k=1)
            if not matches or matches[0][1] < 0.7: # 阈值过滤
                continue
            
            node_name = matches[0][0]
            # 获取出度和入度邻居 (1-hop)
            if node_name in self.G:
                # Outgoing
                for neighbor in self.G.successors(node_name):
                    rel = self.G[node_name][neighbor]['relation']
                    context.append(f"{node_name} {rel} {neighbor}")
                # Incoming (Optional, might be noisy)
                # for neighbor in self.G.predecessors(node_name):
                #     rel = self.G[neighbor][node_name]['relation']
                #     context.append(f"{neighbor} {rel} {node_name}")
        
        # 限制上下文长度，只取前20条最有用的
        return context[:20]

# === 评测逻辑 ===
def eval_task_1_disambiguation(kg, test_file):
    print("\n=== Running Task 1: Entity Disambiguation ===")
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    correct = 0
    total = len(data)
    
    results = []
    
    for item in data:
        mention = item['mention']
        target = item['target_entity']
        
        # 在图谱里搜
        prediction_list = kg.search_node(mention, top_k=1)
        predicted_node = prediction_list[0][0]
        
        is_correct = (predicted_node.lower() == target.lower())
        if is_correct:
            correct += 1
            
        results.append({
            "query": mention,
            "target": target,
            "predicted": predicted_node,
            "score": prediction_list[0][1],
            "correct": is_correct
        })
        
    acc = correct / total
    print(f"Task 1 Accuracy: {acc:.2%} ({correct}/{total})")
    pd.DataFrame(results).to_csv("results_task1_disambiguation_llama.csv", index=False)

def eval_task_2_pathways(kg, test_file):
    print("\n=== Running Task 2: Multi-hop Reasoning ===")
    with open(test_file, 'r') as f:
        data = json.load(f)
        
    results = []
    
    for item in data:
        start = item['start_entity']
        end = item['end_entity']
        
        # Graphusion 找路径
        path, explanation = kg.find_path(start, end)
        
        found = path is not None
        path_str = " -> ".join(path) if path else "No Path"
        
        print(f"Checking path: {start} -> {end} | Found: {found}")
        
        results.append({
            "start": start,
            "end": end,
            "found": found,
            "path": path_str,
            "explanation": " | ".join(explanation)
        })
    
    success_rate = sum(1 for r in results if r['found']) / len(results)
    print(f"Task 2 Success Rate: {success_rate:.2%}")
    pd.DataFrame(results).to_csv("results_task2_pathways_llama.csv", index=False)
def eval_task_2_pathways_baseline_style(kg, test_file):
    """
    用和 baseline 一致的标准评估 Ours：
    - Ours 先找路径
    - 再检查输出是否命中 gt middle node 或 semantic equivalent
    """
    print("\n=== Running Task 2: Multi-hop Reasoning (Baseline-style Evaluation for Ours) ===")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc="Evaluating Ours Task2"):
        start = item['start_entity']
        end = item['end_entity']
        gt_path = item.get('ground_truth_path', [])
        middle_nodes = gt_path[1:-1] if len(gt_path) > 2 else []

        # Ours: find path in graph
        path, explanation = kg.find_path(start, end)

        found = path is not None
        path_str = " -> ".join(path) if path else "No Path"
        explanation_str = " | ".join(explanation)

        # 把 Ours 的结构化输出转成“待评估文本”
        pred_text = f"{path_str} | {explanation_str}"

        # 按 baseline 标准判定：exact or semantic middle-node hit
        is_hit, matched_mid, sim_score, matched_chunk = kg.hit_any_middle_node_semantic(
            pred_text=pred_text,
            middle_nodes=middle_nodes,
            threshold=0.7
        )

        results.append({
            "start": start,
            "end": end,
            "gt_path": " -> ".join(gt_path),
            "middle_nodes": " | ".join(middle_nodes),
            "graph_path_found": found,
            "pred_path": path_str,
            "pred_explanation": explanation_str,
            "success_baseline_style": is_hit,
            "matched_middle_node": matched_mid,
            "semantic_score": sim_score,
            "matched_chunk": matched_chunk
        })

    df = pd.DataFrame(results)
    rate = df['success_baseline_style'].mean()
    print(f"Task 2 Baseline-style Success Rate (Ours): {rate:.2%}")
    df.to_csv("results_task2_pathways_ours_baseline_style.csv", index=False)

# === 主函数 ===
if __name__ == "__main__":
    # 1. 初始化图谱 (只加载一次，比较慢)
    my_kg = MedicalGraph(GRAPH_FILE, EMBEDDING_MODEL)
    
    # 2. 运行任务
    # 请确保 JSON 文件存在，否则注释掉对应行
    if True:
        eval_task_1_disambiguation(my_kg, TEST_DISAMBIGUATION)
        
    if True:
        eval_task_2_pathways(my_kg, TEST_PATHWAYS)
        
    if True:
        eval_task_2_pathways_baseline_style(my_kg, TEST_PATHWAYS)