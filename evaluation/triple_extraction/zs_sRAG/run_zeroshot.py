import json
import pandas as pd
from tqdm import tqdm
from config import call_llm, load_relations, load_pairs

# === 文件路径 ===
PAIRS_FILE = "ht.json"
RELATION_FILE = "relation_types_n.json"
OUTPUT_FILE = "results_zeroshot.csv"

def generate_zeroshot_prompt(head, tail, relation_desc):
    return f"""
You are a strict medical reasoning engine.
Task: Identify the most accurate clinical relationship between the following two entities.

Entities:
- Subject (Head): {head}
- Object (Tail): {tail}

Allowed Relations (Choose EXACTLY one from this list):
{relation_desc}
- No_Relation: If no direct medical relationship exists.

Output Requirement:
Output ONLY the relation label (e.g., "Treats"). Do not output any explanation.

Answer:
"""

def main():
    # 1. 加载数据
    pairs = load_pairs(PAIRS_FILE)
    rel_desc, valid_rels = load_relations(RELATION_FILE)
    
    results = []
    
    print(f"Starting Zero-shot evaluation on {len(pairs)} pairs...")
    
    # 2. 循环预测
    for item in tqdm(pairs):
        h = item['s']
        t = item['o']
        
        prompt = generate_zeroshot_prompt(h, t, rel_desc)
        pred_relation = call_llm(prompt)
        
        # 简单的后处理，去掉可能多余的标点
        pred_relation = pred_relation.replace("Relationship:", "").strip().split('\n')[0]
        
        results.append({
            "head": h,
            "tail": t,
            "predicted_relation": pred_relation,
            "method": "Zero-shot"
        })
        
    # 3. 保存结果
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()