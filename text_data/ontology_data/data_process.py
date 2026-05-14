
# ======================
# -*- coding: utf-8 -*-
import re
from collections import defaultdict

SEM_FILE = "Semtypes 20231215.txt"
CONCEPT_FILE = "Concepts.txt"
OUTPUT_FILE = "clean_bios_entities_en.txt"

# 白名单：只保留这些语义类型的实体
BIOMEDICAL_SEMANTIC_TYPES = {
    # 疾病与异常
    "Disease, Syndrome or Pathologic Function",
    "Neoplastic Process",
    "Injury or Poisoning",
    "Anatomical Abnormality",
    "Cell or Molecular Dysfunction",
    "Mental or Behavioral Dysfunction",
    "Sign, Symptom, or Finding",          # 如 fever, rash — 常作为临床实体保留
    
    # 基因与分子
    "Gene or Genome",
    "Cell Component",                     # 如 mitochondria, nucleus
    "Cell",                               # 如 T cells, stem cells
    
    # 化学物质与药物
    "Chemical or Drug",                   # 覆盖药物、化合物、代谢物等
    
    # 解剖结构
    "Anatomical Structure",               # 如 liver, brain, aorta
    
    # 生物体
    "Microorganism",                      # 细菌、病毒等病原体
    "Eukaryote",                          # 真核生物（含真菌、寄生虫等）
    "Animal",                             # 实验动物或病媒（如 mouse, mosquito）
    "Plant",                              # 药用植物或毒素来源
    "Human",                              # 虽少，但有时用于 population studies
    
    # 生物材料
    "Body Substance",                     # 如 blood, urine, saliva
    "Food",                               # 营养/过敏相关研究中常视为实体（如 peanut, gluten）
}

# ======================
# Step 1: Load cid -> semantic type
# ======================
print("[INFO] Loading semantic types...")
cid_to_sty = {}

with open(SEM_FILE, encoding="utf-8") as f:
    # next(f)  # 如果文件有 header 才启用；根据你提供的样例，无 header，所以注释掉
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        cid, sty = parts[0], parts[1]
        cid_to_sty[cid] = sty

print(f"[INFO] Loaded {len(cid_to_sty)} semantic type mappings.")

# ======================
# Step 2: Extract biomedical English entities
# ======================
print("[INFO] Extracting biomedical English entities...")

entities = set()

with open(CONCEPT_FILE, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        cid, tid, string, tty, lang = parts[0], parts[1], parts[2], parts[3], parts[4]

        # 只处理英文
        if lang != "EN":
            continue

        # 获取语义类型
        sty = cid_to_sty.get(cid)
        if not sty:
            continue

        # 只保留生物医药相关类型
        if sty not in BIOMEDICAL_SEMANTIC_TYPES:
            continue

        # ✅ 启用 PT 过滤：只保留首选术语（Preferred Term）
        if tty != "PT":
            continue

        # 清理并添加
        clean_str = string.strip()
        if clean_str:
            entities.add(clean_str)

print(f"[INFO] Extracted {len(entities)} unique biomedical English entities.")

# ======================
# Step 3: Save to file
# ======================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ent in sorted(entities):
        f.write(ent + "\n")

print(f"[DONE] Saved to '{OUTPUT_FILE}'")