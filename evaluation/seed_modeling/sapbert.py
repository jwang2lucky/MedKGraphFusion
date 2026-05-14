import os
import json
import nltk
import re
import pandas as pd
from tqdm import tqdm
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer


# 初始化 NLTK 资源（只下载一次）
def download_if_needed(package):
    try:
        find(f"corpora/{package}")
    except LookupError:
        nltk.download(package)


download_if_needed("wordnet")
download_if_needed("omw-1.4")
lemmatizer = WordNetLemmatizer()


def singularize(text):
    return " ".join([lemmatizer.lemmatize(word, wordnet.NOUN) for word in text.split()])


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


def step_01_concept_extraction(
    texts: list,
    concept_extraction_output_file: str,
    concept_abstracts_output_file: str,
    logging: any,
    stop_words: list[str] = None,
    config: dict = None
):
    """
    Step 1: Concept Extraction  (SapBERT + Gold concepts + origins 支持增强版)
    """

    if config is None:
        config = {}
    stop_words = config.get("stop_words", "english")
    gold_file = config.get("gold_concept_file", "")

    # ✅ 输入兼容 dict / str
    cleaned_texts = []
    for t in texts:
        if isinstance(t, dict) and "abstract" in t and "origin" in t:
            abs_clean = clean_text(t["abstract"])
            cleaned_texts.append({"abstract": abs_clean, "origin": t["origin"]})
        elif isinstance(t, str):
            cleaned_texts.append({"abstract": clean_text(t), "origin": "unknown"})

    abstracts_only = [item["abstract"] for item in cleaned_texts]
    n_samples = len(abstracts_only)
    print(f"输入的文本数量: {n_samples}")

    # ✅ UMAP 参数自适应小数据集
    n_neighbors = min(20, max(3, n_samples - 1))
    n_components = min(50, max(3, n_samples - 1))
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric="cosine", min_dist=0.0)

    # ✅ SapBERT
    vectorizer_model = CountVectorizer(ngram_range=(1, 4), stop_words=stop_words)
    sentence_model = SentenceTransformer("/home/wangjie/KGC/Graphusion-main/SapBERT_local_test")

    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=False),
        representation_model=KeyBERTInspired(),
        nr_topics=50,
        calculate_probabilities=False,
        verbose=True,
        low_memory=True,
    )

    topics, _ = topic_model.fit_transform(abstracts_only)
    all_topics = topic_model.get_topics()

    # ✅ 提取概念 + 消噪
    candidate_concepts = set()
    for topic_id, kw_list in all_topics.items():
        if topic_id == -1:
            continue
        for word, weight in kw_list:
            if weight > 0.05:  # 更严格过滤
                concept = singularize(clean_text(word))
                if len(concept.split()) > 1:
                    candidate_concepts.add(concept)

    # ✅ Gold concepts 处理 —— 保留缺失 gold 项
    gold_concepts = []
    if gold_file and os.path.exists(gold_file):
        df_gold = pd.read_csv(gold_file, delimiter="|", header=None)
        gold_concepts = df_gold[1].dropna().astype(str).tolist()
        gold_concepts = [singularize(c.lower()) for c in gold_concepts]

    # ✅ 构建 DataFrame 并合并 gold → 与第一段一致的逻辑
    df_candidates = pd.DataFrame(candidate_concepts, columns=["concept"])
    df_candidates["label"] = 0

    if gold_concepts:
        df_gold_df = pd.DataFrame(gold_concepts, columns=["concept"])
        df_gold_df["label"] = 1
        df = pd.concat([df_candidates, df_gold_df], ignore_index=True)
    else:
        df = df_candidates

    df = df.drop_duplicates(subset="concept", keep="first")
    df = df.sort_values(by="label", ascending=False)  # gold concepts 排前

    # ✅ 概念 - 摘要 + origins 映射
    concept_data = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing concepts"):
        concept = row["concept"]
        label = row["label"]

        matched_abstracts = []
        matched_origins = set()

        for item in cleaned_texts:
            if fuzz.partial_ratio(concept.lower(), item["abstract"]) >= 70:
                matched_abstracts.append(item["abstract"])
                matched_origins.add(item["origin"])

        concept_data[concept] = {
            "label": label,
            "abstracts": matched_abstracts,
            "origins": list(matched_origins)
        }

    # ✅ 输出概念列表 TXT
    with open(concept_extraction_output_file, "w", encoding="utf-8") as f:
        for i, c in enumerate(df["concept"], start=1):
            f.write(f"{i}|{c}\n")

    # ✅ 输出 JSON
    with open(concept_abstracts_output_file, "w", encoding="utf-8") as f:
        json.dump(concept_data, f, indent=2, ensure_ascii=False)

    # ✅ 日志统计（与第一段一致）
    logging.info(f"Concepts saved to: {concept_extraction_output_file}")
    logging.info(f"Concept-abstract mapping saved to: {concept_abstracts_output_file}")
    logging.info(f"Total concepts extracted: {len(concept_data)}")

    empty_count = sum(1 for details in concept_data.values() if not details["abstracts"])
    logging.info(f"Concepts without abstracts: {empty_count}")

    if gold_concepts:
        label_0_count = sum(1 for details in concept_data.values() if details["label"] == 0)
        logging.info(f"Concepts from BERTopic (label=0): {label_0_count}")
import os
import logging

if __name__ == "__main__":
    input_dir = "/home/wangjie/KGC/Graphusion-main/Evaluation_MedGraphusion/data"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    config = {
        "stop_words": "english",
        "language": "english",
        "gold_concept_file": ""  # 如有gold concept可填
    }

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        input_file = os.path.join(input_dir, filename)
        sample_texts = []

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split("|||")
                if len(parts) >= 3:
                    origin_id = parts[0]
                    title = parts[1]
                    abstract = parts[2]
                elif len(parts) == 2:
                    origin_id = parts[0]
                    title = ""
                    abstract = parts[1]
                else:
                    continue

                # ✅ 可选：把 title 合并到 abstract 中，增强概念提取效果
                full_text = f"{title} {abstract}".strip() if title else abstract
                sample_texts.append({
                    "origin": origin_id,
                    "abstract": full_text
                })

        base_name = os.path.splitext(filename)[0]
        concept_file = os.path.join(output_dir, f"{base_name}_concepts_v1.tsv")
        concept_abstract_file = os.path.join(output_dir, f"{base_name}_concept_abstracts_v1.json")

        print(f"Processing {filename} ...")
        step_01_concept_extraction(
            texts=sample_texts,
            concept_extraction_output_file=concept_file,
            concept_abstracts_output_file=concept_abstract_file,
            logging=logging,
            config=config
        )
        print(f"Finished {filename}. Outputs:\n  {concept_file}\n  {concept_abstract_file}\n")
