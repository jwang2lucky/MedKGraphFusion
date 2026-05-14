import random
import re
import os

# =========================
# Domain keyword definition
# =========================
DOMAIN_KEYWORDS = {
    "cardiovascular": [
        "cardio", "heart", "cardiac", "myocardial",
        "coronary", "arrhythmia", "hypertension", "stroke"
    ],
    "infection": [
        "infection", "infectious", "sepsis", "bacterial",
        "viral", "pathogen", "antibiotic", "pneumonia"
    ],
    "cancer": [
        "cancer", "tumor", "tumour", "carcinoma",
        "oncology", "metastasis", "chemotherapy", "malignancy"
    ]
}

# =========================
# Configuration
# =========================
INPUT_PATH = "/home/wangjie/KGC/text_data/merged_articles_pubmed.txt"
OUTPUT_DIR = "./pubmed_domain_subsets"

SAMPLE_SIZE = 800      # 每个 domain 抽样数量
RANDOM_SEED = 42

# =========================
# Initialization
# =========================
random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"[INFO] Total input documents: {len(lines)}")

# =========================
# Domain-wise extraction
# =========================
for domain, keywords in DOMAIN_KEYWORDS.items():
    print(f"\n[INFO] Processing domain: {domain}")

    pattern = re.compile("|".join(keywords), re.IGNORECASE)

    filtered_lines = [
        line for line in lines if pattern.search(line)
    ]

    print(f"[INFO] Matched documents: {len(filtered_lines)}")

    if len(filtered_lines) < SAMPLE_SIZE:
        raise ValueError(
            f"[ERROR] Domain '{domain}' has only {len(filtered_lines)} "
            f"documents, less than required {SAMPLE_SIZE}"
        )

    sampled_lines = random.sample(filtered_lines, SAMPLE_SIZE)

    output_path = os.path.join(
        OUTPUT_DIR, f"pubmed_{domain}_{SAMPLE_SIZE}.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(sampled_lines)

    print(f"[INFO] Saved {SAMPLE_SIZE} documents → {output_path}")

print("\n[INFO] All domain subsets generated successfully.")
