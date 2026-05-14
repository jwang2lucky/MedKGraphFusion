import random
import json
import argparse

def sample_jsonl(input_path: str, output_path: str, n: int = 100, seed: int = 42):
    """
    从 JSONL 文件中读取所有有效三元组（含额外字段也没关系），
    提取并标准化为仅包含 's', 'p', 'o' 的格式，
    然后随机抽取 n 个，写入新文件。
    """
    random.seed(seed)

    clean_triples = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过无效行

            # 只要包含 s, p, o 就保留（不管有没有其他字段）
            if all(k in item for k in ['s', 'p', 'o']):
                # 构造干净的三元组字典
                clean_item = {
                    "s": item["s"],
                    "p": item["p"],
                    "o": item["o"]
                }
                clean_triples.append(clean_item)

    total = len(clean_triples)
    print(f"✅ 成功解析 {total} 个有效三元组（已移除 id/concept/origins 等字段）。")

    if total == 0:
        print("⚠️ 没有找到包含 s/p/o 的有效三元组！")
        return

    # 随机采样
    sample_size = min(n, total)
    sampled_items = random.sample(clean_triples, sample_size)

    # 写入新 JSONL 文件（每行一个标准三元组）
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for item in sampled_items:
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 成功抽取并保存 {len(sampled_items)} 个干净三元组到 {output_path}")

if __name__ == "__main__":
    #sample_jsonl('./kg_spo_only-02.jsonl', '02-100.json', 100, 42)
    sample_jsonl('./kg_spo_only-03.jsonl', '03-100.json', 100, 64)