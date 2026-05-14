import os
import json
import re
import nltk
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Set, Dict, Any, Tuple

# NLTK resources
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# ML/NLP Libraries
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(
    min_cluster_size=10,        # ⭐ 非常重要
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom"
)
# 设置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# === NLTK 下载检查 ===
# 初始化 NLTK 资源（只下载一次）
def download_if_needed(package):
    try:
        find(f"corpora/{package}")
    except LookupError:
        nltk.download(package)


download_if_needed("wordnet")
download_if_needed("omw-1.4")
lemmatizer = WordNetLemmatizer()

# === 工具函数 ===

def clean_text(text: str) -> str:
    """清洗文本：去除 LaTeX 命令，保留字母数字和部分符号"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\\usepackage\{[^}]+\}", "", text)
    text = re.sub(
        r"\b([a-z]+ )?(et al|al)\s*\d{4}[a-z]?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 单独的 et al / al
    text = re.sub(
        r"\b(et al|al)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
    r"\b\d{1,2}(\s?\d{2})?\s?(am|pm)\b",
    "",
    text,
    flags=re.IGNORECASE,)
    text = re.sub(
    r"\b\d{1,2}(\s?\d{2})?\s?(am|pm)\s+(blood|urine|serum|plasma)\b",
    "",
    text,
    flags=re.IGNORECASE,)
    # 40 mg / 5 ml / 10 mcg
    text = re.sub(
    r"\b\d+\s?(mg|g|mcg|ug|ml|iu)\b",
    "",
    text,
    flags=re.IGNORECASE,)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # 保留字母、数字、空格、连字符、斜杠
    text = re.sub(r"[^a-zA-Z0-9\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def singularize(text: str) -> str:
    """将文本中的名词转为单数"""
    if not text:
        return ""
    return " ".join([lemmatizer.lemmatize(w, wordnet.NOUN) for w in text.split()])

def load_ontology_terms(path: str) -> Set[str]:
    """加载医学本体词表"""
    if not path or not os.path.exists(path):
        logger.warning(f"Ontology file not found at {path}. Proceeding without ontology boost.")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

# === 本体感知 Representation 模型 ===

class OntologyAwareRepresentation(BaseRepresentation):
    def __init__(self, ontology_terms: Set[str], top_n: int = 10, boost: float = 0.5):
        self.ontology_terms = ontology_terms
        self.top_n = top_n
        self.boost = boost

    def fit(self, documents: List[str], y=None):
        return self

    def transform(self, topic_model, topic_ids=None):
        all_topics = topic_model.get_topics()
        results = []

        for topic_id in topic_ids:
            if topic_id == -1: # Outlier topic
                results.append([])
                continue

            weighted_words = all_topics.get(topic_id, [])
            # 这里的逻辑是：如果词在本体中，给予权重加成
            boosted_words = [
                (word, weight + (self.boost if word.lower() in self.ontology_terms else 0.0))
                for word, weight in weighted_words
            ]

            # 重新排序并取 Top N
            boosted_words.sort(key=lambda x: x[1], reverse=True)
            top_words = [w for w, _ in boosted_words[:self.top_n]]
            results.append(top_words)

        return results

# === 核心提取逻辑 ===

def extract_topic_concepts(
    texts: List[str],
    ontology_terms: Set[str],
    embedding_model_path: str,
    top_n: int = 10,
    top_k: int = 5000,
    stop_words: str = "english"
) -> List[str]:
    
    valid_texts = [t for t in texts if len(t) >= 10]
    if not valid_texts:
        logger.warning("Not enough valid texts for topic modeling.")
        return []

    # 1. 加载 Embedding 模型 (支持本地路径或 HF hub)
    if os.path.exists(embedding_model_path):
        logger.info(f"Loading local embedding model: {embedding_model_path}")
        sentence_model = SentenceTransformer(embedding_model_path)
    else:
        logger.warning(f"Local model not found at {embedding_model_path}. Downloading/Using 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'...")
        sentence_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    # 2. 配置组件
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words, min_df=2)
    
    # 根据样本量动态调整 UMAP 参数
    n_samples = len(valid_texts)
    umap_model = UMAP(
        n_neighbors=min(15, max(2, n_samples - 1)), 
        n_components = min(5, max(3, n_samples - 1)), 
        min_dist=0.0, 
        metric='cosine',
        random_state=42
    )

    ontology_repr = OntologyAwareRepresentation(ontology_terms, top_n=top_n, boost=0.5)

    # 3. BERTopic 初始化与训练
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True), # 开启 reduce_frequent_words 抑制通用词
        hdbscan_model=hdbscan_model,
        representation_model=ontology_repr,
        nr_topics= 1500, # 自动缩减主题数，或者设为 1500
        calculate_probabilities=False,
        verbose=True
    )

    logger.info("Fitting BERTopic model...")
    topic_model.fit(valid_texts)
    
    # 4. 提取候选词
    all_topics = topic_model.get_topics()
    candidates = []
    
    # 遍历所有 Topic 收集高分词
    for topic_id, kw_list in all_topics.items():
        if topic_id == -1: continue
        for word, weight in kw_list:
            w_clean = singularize(clean_text(word))
            if len(w_clean) < 3: continue # 跳过过短的词
            
            # 如果在本体中，给予额外全局打分 boost
            is_ontology = w_clean in ontology_terms
            final_score = weight * (1.5 if is_ontology else 1.0)
            candidates.append((final_score, w_clean))

    # 排序去重
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    top_concepts = []
    for _, c in candidates:
        if c not in seen:
            top_concepts.append(c)
            seen.add(c)
        if len(top_concepts) >= top_k:
            break
            
    return top_concepts


# === 主处理流程 ===

def step_01_concept_extraction(
    texts: List[str],
    origins: List[str], # ✅ 修复：添加 origins 参数
    concept_extraction_output_file: str,
    concept_abstracts_output_file: str,
    config: Dict[str, Any] = None
) -> None:

    if config is None: config = {}
    
    # 路径配置
    ontology_file = config.get("ontology_file", "")
    embedding_path = config.get("embedding_path", "")
    gold_file = config.get("gold_concept_file", "")
    
    logger.info("Step 1: Starting concept extraction.")

    # 1. 预处理所有文本
    cleaned_data = [] # Stores tuple (clean_text, original_origin)
    for t, o in zip(texts, origins):
        if isinstance(t, str) and t.strip():
            c_text = clean_text(t)
            if c_text:
                cleaned_data.append((c_text, str(o)))
    
    if not cleaned_data:
        logger.error("No valid texts found. Exiting.")
        return

    # 解压用于训练的文本列表
    train_texts = [x[0] for x in cleaned_data]
    ontology_terms = load_ontology_terms(ontology_file)

    # 2. 提取概念
    candidate_concepts = extract_topic_concepts(
        texts=train_texts,
        ontology_terms=ontology_terms,
        embedding_model_path=embedding_path,
        top_k=5000
    )
    logger.info(f"Extracted {len(candidate_concepts)} unique concepts.")

    # 3. 加载 Gold Standard (用于打标签)
    gold_concepts = set()
    if gold_file and os.path.exists(gold_file):
        try:
            df_gold = pd.read_csv(gold_file, delimiter="|", header=None)
            gold_concepts = set(singularize(str(c).lower()) for c in df_gold[1].dropna())
        except Exception as e:
            logger.warning(f"Failed to load gold file: {e}")

    # 4. 映射 Concept 到 Abstracts (性能优化版)
    # 使用 Regex 全词匹配，比 fuzz 快很多且更准确
    concept_abstracts = {}
    MAX_EVIDENCE = 3 #设置origin文献来源的数量
    logger.info("Mapping concepts back to abstracts (using optimized regex)...")
    
    for concept in tqdm(candidate_concepts, desc="Processing Concepts"):
        is_gold = 1 if concept in gold_concepts else 0
        
        matched_abstracts = []
        matched_origins = []
        
        # 编译正则：\b 表示单词边界，防止 "cat" 匹配到 "scatter"
        # re.escape 确保概念中的特殊字符被转义
        pattern = re.compile(r'\b' + re.escape(concept) + r'\b', re.IGNORECASE)
        
        for text, origin in cleaned_data:
            if pattern.search(text):
                matched_abstracts.append(text)
                matched_origins.append(origin)
                #达到上限，立即停止
                if len(matched_abstracts) >= MAX_EVIDENCE:
                    break
        # 只有当该概念至少出现在一篇文档中时才保存 (可选)
        if matched_abstracts: 
            concept_abstracts[concept] = {
                "abstracts": matched_abstracts,
                "origins": matched_origins,
                "label": is_gold
            }

    # 5. 输出文件
    os.makedirs(os.path.dirname(concept_extraction_output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(concept_abstracts_output_file) or ".", exist_ok=True)

    # TSV 输出
    valid_concepts_list = sorted(concept_abstracts.keys())
    df_out = pd.DataFrame(
        [[i+1, c] for i, c in enumerate(valid_concepts_list)],
        columns=["id", "concept"]
    )
    df_out.to_csv(concept_extraction_output_file, sep="|", index=False, header=False, encoding="utf-8")

    # JSON 输出
    with open(concept_abstracts_output_file, "w", encoding="utf-8") as f:
        json.dump(concept_abstracts, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved concepts to {concept_extraction_output_file}")
    logger.info(f"Saved mappings to {concept_abstracts_output_file}")
    logger.info("Step 1: ✅ Completed.")


