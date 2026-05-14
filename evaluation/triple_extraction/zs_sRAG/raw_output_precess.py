import re
import pandas as pd
import os

# === 配置 ===
INPUT_LOG_FILE = "./nano_graphrag_cache/llm_raw_outputs.log"
OUTPUT_ENTITY_CSV = "graphrag_entities_extracted.csv"
OUTPUT_RELATION_CSV = "graphrag_relations_extracted.csv"

def parse_graphrag_log(log_path):
    entities = []
    relations = []
    
    # GraphRAG 的分隔符通常是 <|>
    # 格式示例: ("relationship"<|>"Source"<|>"Target"<|>"Description"<|>Weight)
    
    if not os.path.exists(log_path):
        print(f"? 找不到文件: {log_path}")
        return [], []

    print(f"?? Reading log file: {log_path}...")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 预处理：有些模型输出会带 markdown 代码块，先去掉
    content = content.replace("```csv", "").replace("```", "")
    
    # 2. 按行或按 ## 分割 (Qwen 有时用 ## 分割)
    # 先把 ## 替换成换行符，统一处理
    normalized_content = content.replace("##", "\n")
    lines = normalized_content.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 移除外层的括号
        if line.startswith('("') and line.endswith(')'):
            line = line[1:-1]
        elif line.startswith('(') and line.endswith(')'):
            line = line[1:-1]
            
        # 分割字段
        parts = line.split('<|>')
        
        # 清理引号 (有些模型输出 "entity" 带引号，有些不带)
        parts = [p.strip().strip('"').strip("'") for p in parts]
        
        if len(parts) < 3:
            continue

        item_type = parts[0] # "entity" or "relationship"

        try:
            if item_type == "entity":
                # 格式: entity <|> Name <|> Type <|> Description
                if len(parts) >= 4:
                    entities.append({
                        "entity": parts[1],
                        "type": parts[2],
                        "description": parts[3]
                    })
            
            elif item_type == "relationship":
                # 格式: relationship <|> Source <|> Target <|> Description <|> Weight
                # 注意：Qwen 有时会漏掉 Weight，或者 Description 很长
                if len(parts) >= 4:
                    src = parts[1]
                    tgt = parts[2]
                    desc = parts[3]
                    weight = parts[4] if len(parts) > 4 else "1"
                    
                    relations.append({
                        "head": src,
                        "relation": "related_to", # GraphRAG 默认没有具体谓语，统一叫 related_to
                        "tail": tgt,
                        "description": desc, # 这才是 GraphRAG 的核心
                        "weight": weight
                    })
        except Exception as e:
            print(f"?? 解析错误行: {line} | Error: {e}")

    return entities, relations

# === 主程序 ===
if __name__ == "__main__":
    ents, rels = parse_graphrag_log(INPUT_LOG_FILE)
    
    print(f"? Extracted {len(ents)} entities.")
    print(f"? Extracted {len(rels)} relations.")
    
    # 保存实体
    if ents:
        df_ent = pd.DataFrame(ents)
        df_ent.drop_duplicates(subset=['entity'], inplace=True)
        df_ent.to_csv(OUTPUT_ENTITY_CSV, index=False)
        print(f"?? Entities saved to {OUTPUT_ENTITY_CSV}")
        
    # 保存关系
    if rels:
        df_rel = pd.DataFrame(rels)
        # 去重
        df_rel.drop_duplicates(subset=['head', 'tail', 'description'], inplace=True)
        df_rel.to_csv(OUTPUT_RELATION_CSV, index=False)
        print(f"?? Relations saved to {OUTPUT_RELATION_CSV}")
        
        # 打印几条看看样子
        print("\n--- Relation Preview ---")
        print(df_rel[['head', 'tail', 'description']].head())