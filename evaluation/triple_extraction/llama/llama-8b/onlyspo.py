import json

def extract_spo(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            spo = {
                "s": data["s"],
                "p": data["p"],
                "o": data["o"]
            }
            fout.write(json.dumps(spo, ensure_ascii=False) + "\n")

# 使用示例
if __name__ == "__main__":
    extract_spo("step-03.jsonl", "kg_spo_only-03.jsonl")
    extract_spo("step-02.jsonl", "kg_spo_only-02.jsonl")
    print("SPO extraction completed. Output saved to kg_spo_only.jsonl")