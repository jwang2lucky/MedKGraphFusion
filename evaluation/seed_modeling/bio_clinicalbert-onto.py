import os
import json
import logging
from tqdm import tqdm
import pandas as pd
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
import nltk
from sentence_transformers import SentenceTransformer, models

def load_bio_clinical_bert_sentence_model():
    word_embedding_model = models.Transformer(
        "/home/wangjie/KGC/bio_clinicalbert",
        max_seq_length=512
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    return SentenceTransformer(
        modules=[word_embedding_model, pooling_model]
    )
# === Helper functions ===

def download_if_needed(package):
    try:
        find(f"corpora/{package}")
    except LookupError:
        nltk.download(package)

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
    return text.lower()# 可以根据需要扩展清洗规则

# === Main function for concept extraction ===
def run_concept_extraction(texts, output_concept_file, output_json_file, config=None):
    if config is None:
        config = {}

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Step01")

    download_if_needed("wordnet")
    download_if_needed("omw-1.4")
    logger.info("Step 1: Starting concept extraction.")

    if 'language' not in config:
        config['language'] = "english"
    if 'gold_concept_file' not in config:
        config['gold_concept_file'] = ""

    # 创建BERTopic模型
    embedding_model_path="/home/wangjie/KGC/bio_clinicalbert"
    if config['language'] == "english":
        vectorizer_model = CountVectorizer(ngram_range=(1, 4), stop_words="english")
        logger.info(f"Loading local embedding model: {embedding_model_path}")
        sentence_model = load_bio_clinical_bert_sentence_model()
    else:
        logger.info(f"Using language {config['language']}. Not supported yet.")
        exit(0)

    n_samples = len(texts)
    print("输入的text有多少条："+str(n_samples))
    n_neighbors = min(20, max(3, n_samples - 1))
    n_components = min(50, max(3, n_samples - 1))
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric="cosine", min_dist=0.0, random_state=42)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(verbose=True,
                           umap_model=umap_model,
                           ctfidf_model=ctfidf_model,
                           vectorizer_model=vectorizer_model,
                           embedding_model=sentence_model,
                           representation_model=representation_model,
                           nr_topics=50,
                           low_memory=True,
                           calculate_probabilities=False)

    # 支持两种格式：dict包含abstract或直接文本
    if isinstance(texts[0], dict) and "abstract" in texts[0]:
        abstracts = [t["abstract"] for t in texts]
    else:
        abstracts = texts

    topics, _ = topic_model.fit_transform(abstracts)
    all_topics = topic_model.get_topics()

    extracted_concepts = []
    for topic_num, keywords in all_topics.items():
        if topic_num != -1:
            extracted_concepts.extend([word for word, _ in keywords])
    extracted_concepts = list(set(k.lower() for k in extracted_concepts))

    # 写入concept文件
    with open(output_concept_file, "w", encoding="utf-8") as f:
        for idx, concept in enumerate(extracted_concepts, 1):
            f.write(f"{idx}|{concept}\n")
    logger.info(f"Concepts written to {output_concept_file}")

    # 词形还原
    lemmatizer = WordNetLemmatizer()
    def singularize_concept(concept):
        return ' '.join([lemmatizer.lemmatize(w, wordnet.NOUN) for w in concept.split()])

    extracted_concept = [singularize_concept(c) for c in extracted_concepts]

    df_concepts = pd.DataFrame(extracted_concept, columns=["concept"])
    df_concepts["label"] = 0

    # 合并gold概念（如果提供）
    if config['gold_concept_file'] and os.path.exists(config['gold_concept_file']):
        gold_df = pd.read_csv(config['gold_concept_file'], delimiter="|", header=None)
        gold_concepts = [singularize_concept(str(c)) for c in gold_df[1].dropna()]
        gold_concepts = [c.lower() for c in gold_concepts]
        df_gold = pd.DataFrame(gold_concepts, columns=["concept"])
        df_gold["label"] = 1
        df_concepts = pd.concat([df_concepts, df_gold]).sort_values("label")
    df_concepts = df_concepts.drop_duplicates(subset="concept", keep="first")

    # 根据概念过滤文本
    def filter_abstracts_by_term(term, abstracts, threshold=70):
        filtered, origins = [], set()
        for entry in abstracts:
            if isinstance(entry, dict):
                abstract_text = entry['abstract']
                origin_id = entry.get('origin', "")
            else:
                abstract_text = entry
                origin_id = ""
            if fuzz.partial_ratio(term.lower(), abstract_text.lower()) >= threshold:
                filtered.append(abstract_text)
                origins.add(origin_id)
        return filtered, list(origins)

    concept_abstracts = {}
    for _, row in tqdm(df_concepts.iterrows(), total=df_concepts.shape[0], desc="Processing concepts"):
        concept = row["concept"]
        label = row["label"]
        filtered_abstracts, origins = filter_abstracts_by_term(concept, texts)
        concept_abstracts[concept] = {"abstracts": filtered_abstracts, "label": label, "origin": origins}

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(concept_abstracts, f, ensure_ascii=False, indent=4)
    logger.info(f"Abstracts written to {output_json_file}")

    logger.info(f"Step 1 completed. Number of concepts: {len(extracted_concept)}")

# === Main script for quick testing ===
'''if __name__ == "__main__":
    input_file = "/home/wangjie/KGC/Graphusion-main/Evaluation_MedGraphusion/data/pubmed_cancer_800.txt"
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
            sample_texts.append({
                "origin": origin_id,
                "abstract": abstract
            })

    config = {
        "stop_words": "english",
        "language": "english"
    }

    run_concept_extraction(
        texts=sample_texts,
        output_concept_file="./outputs/v0_concepts_pcer.tsv",
        output_json_file="./outputs/v0_concept_abstracts_pcer.json",
        config=config
    )
'''
import os

if __name__ == "__main__":
    input_dir = "/home/wangjie/KGC/Graphusion-main/Evaluation_MedGraphusion/data"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "stop_words": "english",
        "language": "english"
    }

    # 遍历目录下所有txt文件
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
                sample_texts.append({
                    "origin": origin_id,
                    "abstract": abstract
                })

        # 去掉 .txt 后缀生成输出文件名
        base_name = os.path.splitext(filename)[0]
        concept_file = os.path.join(output_dir, f"{base_name}_concepts_v5.tsv")
        concept_abstract_file = os.path.join(output_dir, f"{base_name}_concept_abstracts_v4.json")

        print(f"Processing {filename} ...")
        run_concept_extraction(
            texts=sample_texts,
            output_concept_file=concept_file,
            output_json_file=concept_abstract_file,
            config=config
        )
        print(f"Finished {filename}, outputs: {concept_file}, {concept_abstract_file}\n")
