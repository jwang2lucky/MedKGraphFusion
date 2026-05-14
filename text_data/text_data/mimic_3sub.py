import random
import re
import os

# ================= 配置 =================
input_path = "/mnt/nvme/extra_data/wangjie/note/discharge_clean_medical_2.txt"
output_dir = "./mimic_subsets"
os.makedirs(output_dir, exist_ok=True)

# 每个 domain 的抽样数量
sample_sizes = {
    "cardiovascular": 800,
    "infection": 800,
    "cancer": 800
}

# 随机种子
random_seed = 42
random.seed(random_seed)

# ================= 临床关键词 =================
DOMAIN_KEYWORDS = {
    "cardiovascular": ["ecg", "troponin", "heart failure", "cardio", "heart", "cardiac", "myocardial", "coronary", "arrhythmia", "hypertension", "stroke"],
    "infection": ["fever", "septic", "antimicrobial", "infection", "infectious", "sepsis", "bacterial", "viral", "pathogen", "antibiotic", "pneumonia"],
    "cancer": ["chemo", "radiation", "biopsy", "cancer", "tumor", "tumour", "carcinoma", "oncology", "metastasis", "malignancy"]
}

# ================= 读取原始文本 =================
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"原始文本总数: {len(lines)}")

# ================= 按 domain 抽样 =================
for domain, keywords in DOMAIN_KEYWORDS.items():
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    
    # 过滤包含关键词的文本
    filtered_lines = [line for line in lines if pattern.search(line)]
    print(f"{domain} 过滤后文本数量: {len(filtered_lines)}")
    
    # 检查数量
    sample_size = sample_sizes[domain]
    if len(filtered_lines) < sample_size:
        raise ValueError(f"{domain} 过滤后文本数量 {len(filtered_lines)} 少于要求抽样 {sample_size}")
    
    # 随机抽样
    sampled_lines = random.sample(filtered_lines, sample_size)
    
    # 写出文件
    output_path = os.path.join(output_dir, f"mimic_{domain}_{sample_size}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)
    
    print(f"{domain} 子集已生成: {output_path} (共 {sample_size} 条)")
