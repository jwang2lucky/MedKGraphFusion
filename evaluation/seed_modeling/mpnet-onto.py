import os
import json
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Set, Dict, Any
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import logging
# Ensure required NLTK resources
def download_if_needed(package: str):
    try:
        find(f"corpora/{package}")
    except LookupError:
        nltk.download(package)

download_if_needed("wordnet")
download_if_needed("omw-1.4")

lemmatizer = WordNetLemmatizer()

# === 医学本体加载 ===
def load_ontology_terms(path: str = "/home/wangjie/KGC/Graphusion-main/BIOS_v3/extracted_str_en_unique.txt") -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

# === 本体感知 representation（继承 BaseRepresentation） ===
class OntologyAwareRepresentation(BaseRepresentation):
    def __init__(self, ontology_terms: Set[str], top_n: int = 10):
        self.ontology_terms = ontology_terms
        self.top_n = top_n

    def fit(self, documents: List[str], y=None):
        return self

    def transform(self, topic_model, topic_ids=None):
        all_topics = topic_model.get_topics()
        results = []
        for topic_id in topic_ids:
            if topic_id == -1:
                results.append([])
                continue
            weighted_words = all_topics.get(topic_id, [])
            selected = []
            for word, weight in weighted_words:
                if word.lower() in self.ontology_terms:
                    selected.append(word)
                if len(selected) >= self.top_n:
                    break
            if len(selected) < self.top_n:
                for word, weight in weighted_words:
                    if word not in selected:
                        selected.append(word)
                    if len(selected) >= self.top_n:
                        break
            results.append(selected)
        return results

# text cleaning and singularize
_CLEAN_KEEP = r"[^a-zA-Z0-9\s\-/]"

def clean_text(text: str) -> str:
    text = re.sub(r"\\usepackage\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(_CLEAN_KEEP, " ", text)   # keep - and /
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def singularize(text: str) -> str:
    return " ".join([lemmatizer.lemmatize(w, wordnet.NOUN) for w in text.split()])

# topic-concept extraction (ontology-aware)
import os
import json
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Set, Dict, Any
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
from bertopic import BERTopic
from bertopic.representation import BaseRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import logging

# Ensure required NLTK resources
def download_if_needed(package: str):
    try:
        find(f"corpora/{package}")
    except LookupError:
        nltk.download(package)

download_if_needed("wordnet")
download_if_needed("omw-1.4")

lemmatizer = WordNetLemmatizer()

# === 医学本体加载 ===
def load_ontology_terms(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

# === 本体感知 representation（继承 BaseRepresentation） ===
class OntologyAwareRepresentation(BaseRepresentation):
    def __init__(self, ontology_terms: Set[str], top_n: int = 10):
        self.ontology_terms = ontology_terms
        self.top_n = top_n

    def fit(self, documents: List[str], y=None):
        return self

    def transform(self, topic_model, topic_ids=None):
        all_topics = topic_model.get_topics()
        results = []
        for topic_id in topic_ids:
            if topic_id == -1:
                results.append([])
                continue
            weighted_words = all_topics.get(topic_id, [])
            selected = []
            # 只保留 ontology 中的词
            for word, weight in weighted_words:
                if word.lower() in self.ontology_terms:
                    selected.append(word)
                if len(selected) >= self.top_n:
                    break
            results.append(selected)
        return results

# 文本清理 & 单数化
_CLEAN_KEEP = r"[^a-zA-Z0-9\s\-/]"

def clean_text(text: str) -> str:
    text = re.sub(r"\\usepackage\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(_CLEAN_KEEP, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def singularize(text: str) -> str:
    return " ".join([lemmatizer.lemmatize(w, wordnet.NOUN) for w in text.split()])

# === topic-concept extraction（BERTopic + ontology filter） ===
# === topic-concept extraction (ontology-aware, filtered + optional fuzzy) ===
from fuzzywuzzy import fuzz

def extract_topic_concepts(
    texts: List[str],
    ontology_terms: Set[str],
    top_n: int = 10,
    stop_words="english",
    fuzzy_threshold: int = 90,  # 可调，0-100
    use_fuzzy: bool = False
) -> Set[str]:
    texts = [t for t in texts if len(t.split()) >= 3]
    if not texts:
        return set()

    # embedding model (SapBERT local)
    sentence_model = SentenceTransformer("/home/wangjie/KGC/all-mpnet-base-v2")
    vectorizer_model = CountVectorizer(ngram_range=(2, 4), stop_words=stop_words)

    # UMAP only when enough samples
    umap_model = None
    if len(texts) >= 5:       
        n_samples = len(texts)
        n_neighbors = min(20, max(3, n_samples - 1))
        n_components = min(50, max(3, n_samples - 1))
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components,
                          metric="cosine", min_dist=0.0, random_state=42)

    ontology_repr = OntologyAwareRepresentation(ontology_terms, top_n=top_n)

    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=False),
        representation_model=ontology_repr,
        nr_topics=50,
        calculate_probabilities=False,
        low_memory=True,
    )

    try:
        topics, _ = topic_model.fit_transform(texts)
        all_topics = topic_model.get_topics()

        topic_concepts = set()

        def match_ontology(word):
            word_clean = singularize(clean_text(word.lower()))
            # 过滤数字或短词
            if word_clean.isdigit() or len(word_clean) <= 2:
                return None
            # 严格匹配
            if word_clean in ontology_terms:
                return word_clean
            # 可选模糊匹配
            if use_fuzzy:
                for term in ontology_terms:
                    if fuzz.ratio(word_clean, term) >= fuzzy_threshold:
                        return term
            return None

        for topic_id, kw_list in all_topics.items():
            if topic_id == -1:
                continue
            selected = []
            for word, weight in kw_list:
                matched = match_ontology(word)
                if matched:
                    selected.append(matched)
                if len(selected) >= top_n:
                    break
            topic_concepts.update(selected)

        return topic_concepts

    except Exception as e:
        # fallback: frequency-based strict matching
        vec = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words)
        X = vec.fit_transform(texts)
        vocab = vec.get_feature_names_out()
        counts = np.asarray(X.sum(axis=0)).ravel()
        pairs = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)
        topic_concepts = set()
        for w, c in pairs:
            matched = match_ontology(w)
            if matched:
                topic_concepts.add(matched)
        return topic_concepts


# === 主入口 ===
def step_01_concept_extraction(
    texts: List[Any],
    concept_extraction_output_file: str,
    concept_abstracts_output_file: str,
    logging: Any,
    stop_words: List[str] = None,
    config: Dict[str, Any] = None
) -> None:
    if config is None:
        config = {}
    if stop_words is None:
        stop_words = config.get("stop_words", "english")

    logging.info("Step 1: Starting concept extraction (ontology-aware with origins).")

    # clean texts
    cleaned_texts = []
    for t in texts:
        if isinstance(t, dict) and "abstract" in t and "origin" in t:
            abs_clean = clean_text(t["abstract"])
            cleaned_texts.append({"abstract": abs_clean, "origin": t["origin"]})
        elif isinstance(t, str):
            cleaned_texts.append({"abstract": clean_text(t), "origin": "unknown"})
    if not cleaned_texts:
        logging.warning("No valid texts after cleaning. Exiting.")
        return

    abstracts_only = [item["abstract"] for item in cleaned_texts]

    # load ontology
    ontology_file = config.get("ontology_file", "/home/wangjie/KGC/Graphusion-main/BIOS_v3/extracted_str_en_unique-n.txt")
    ontology_terms = load_ontology_terms(ontology_file)

    # extract candidate concepts
    candidate_concepts = extract_topic_concepts(
        abstracts_only, ontology_terms, top_n=10, stop_words=stop_words, use_fuzzy=False
    )
    logging.info(f"Extracted {len(candidate_concepts)} candidate concepts from topics.")

    # load gold concepts if provided
    gold_concepts = set()
    gold_file = config.get("gold_concept_file", "")
    if gold_file and os.path.exists(gold_file):
        try:
            df_gold = pd.read_csv(gold_file, delimiter="|", header=None)
            gold_concepts = set(singularize(str(c).lower()) for c in df_gold[1].dropna())
            logging.info(f"Loaded {len(gold_concepts)} gold concepts from {gold_file}.")
        except Exception as e:
            logging.warning(f"Failed to load gold concept file: {e}")

    # build dataframe and labels
    df = pd.DataFrame(sorted(candidate_concepts), columns=["concept"])
    df["label"] = df["concept"].apply(lambda x: 1 if x in gold_concepts else 0)

    # concept -> abstracts + origins mapping
    concept_abstracts = {}
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing concepts"):
        concept = row["concept"]
        label = int(row["label"])

        matched_abstracts = []
        matched_origins = set()

        for item in cleaned_texts:
            if fuzz.partial_ratio(concept.lower(), item["abstract"]) >= 70:
                matched_abstracts.append(item["abstract"])
                matched_origins.add(item["origin"])

        concept_abstracts[concept] = {
            "abstracts": matched_abstracts,
            "origins": list(matched_origins),
            "label": label
        }

    # ensure output dirs exist
    os.makedirs(os.path.dirname(concept_extraction_output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(concept_abstracts_output_file) or ".", exist_ok=True)

    # write concept list
    df_out = pd.DataFrame([[i+1, c] for i, c in enumerate(concept_abstracts.keys())],
                          columns=["id", "concept"])
    df_out.to_csv(concept_extraction_output_file, sep="|", index=False, header=False, encoding="utf-8")

    # write concept -> abstracts + origins mapping
    with open(concept_abstracts_output_file, "w", encoding="utf-8") as f:
        json.dump(concept_abstracts, f, indent=2, ensure_ascii=False)

    logging.info(f"Concepts written to {concept_extraction_output_file}.")
    logging.info(f"Abstracts w/ origins written to {concept_abstracts_output_file}.")
    logging.info(f"Total concepts extracted: {len(candidate_concepts)}")
    empty_abstracts_count = sum(1 for v in concept_abstracts.values() if not v["abstracts"])
    logging.info(f"Concepts with empty abstracts: {empty_abstracts_count}")

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
        concept_file = os.path.join(output_dir, f"{base_name}_concepts_v2.tsv")
        concept_abstract_file = os.path.join(output_dir, f"{base_name}_concept_abstracts_v2.json")

        print(f"Processing {filename} ...")
        step_01_concept_extraction(
            texts=sample_texts,
            concept_extraction_output_file=concept_file,
            concept_abstracts_output_file=concept_abstract_file,
            logging=logging,
            config=config
        )
        print(f"Finished {filename}. Outputs:\n  {concept_file}\n  {concept_abstract_file}\n")