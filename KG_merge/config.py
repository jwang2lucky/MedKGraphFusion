# config.py
import os

# ─── Ollama LLM 配置 ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1:70b")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 512          # num_predict
LLM_TIMEOUT     = 120          # 秒，70b 模型推理较慢

# ─── 文件路径 ──────────────────────────────────────────────────────────────────
KG_CSV_PATH      = "kg.csv"
INPUT_JSONL_PATH = "input_triples.jsonl"
OUTPUT_CSV_PATH  = "kg_merged.csv"
CACHE_DIR        = ".cache"

# ─── 实体对齐阈值 ──────────────────────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD   = 80    # RapidFuzz 字符串相似度阈值 (0-100)
EMBEDDING_SIM_THRESHOLD = 0.85  # 语义相似度阈值 (0-1)
TOP_K_CANDIDATES        = 5     # 每个实体最多取 Top-K 候选
LLM_CONFIRM_THRESHOLD   = 0.6   # 模糊分 < 该值*100 时调用 LLM 二次确认

# ─── primeKG 原生合法 type 集合 ────────────────────────────────────────────────
PRIMEKG_NATIVE_TYPES = {
    "gene/protein",
    "disease",
    "drug",
    "effect/phenotype",
    "biological_process",
    "cellular_component",
    "molecular_function",
    "pathway",
    "anatomy",
    "exposure",
}

# ─── 扩展 type（运行时动态注册，初始为空） ────────────────────────────────────
# 格式: { "custom_type_name": "描述/来源" }
EXTENDED_TYPES: dict[str, str] = {}

# ─── 支持度过滤 ────────────────────────────────────────────────────────────────
MIN_SUPPORT = 1

# ─── primeKG 原生 relation 集合 ────────────────────────────────────────────────
PRIMEKG_NATIVE_RELATIONS = {
    "anatomy_anatomy", "anatomy_protein_absent", "anatomy_protein_present",
    "bioprocess_bioprocess", "bioprocess_protein",
    "cellcomp_cellcomp", "cellcomp_protein",
    "contraindication", "disease_disease",
    "disease_phenotype_negative", "disease_phenotype_positive", "disease_protein",
    "drug_drug", "drug_effect", "drug_protein",
    "exposure_bioprocess", "exposure_cellcomp", "exposure_disease",
    "exposure_exposure", "exposure_molfunc", "exposure_protein",
    "indication", "molfunc_molfunc", "molfunc_protein",
    "off-label use", "pathway_pathway", "pathway_protein",
    "phenotype_phenotype", "phenotype_protein", "protein_protein",
}

# ─── 扩展 relation（运行时动态注册） ─────────────────────────────────────────
# 格式: { "relation_name": {"display_relation": str, "description": str} }
EXTENDED_RELATIONS: dict[str, dict] = {}