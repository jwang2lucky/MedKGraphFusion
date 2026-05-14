from __future__ import annotations

"""
实体对齐与消歧模块。

流程：
1. RapidFuzz 快速字符串匹配 → 候选集（Top-K）
2. Sentence-Transformers 语义相似度精排
3. 分数仍低于阈值 → 调用 Ollama LLM 最终裁决
4. 结果缓存到本地（避免重复 API 调用）
"""

import hashlib
import json
import logging
import os
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logging.warning("sentence-transformers 未安装，跳过语义相似度排序。")

from config import (
    CACHE_DIR,
    EMBEDDING_SIM_THRESHOLD,
    FUZZY_MATCH_THRESHOLD,
    LLM_CONFIRM_THRESHOLD,
    TOP_K_CANDIDATES,
)
from llm_client import call_llm, extract_json

logger = logging.getLogger(__name__)

_embed_model: Optional[object] = None

os.makedirs(CACHE_DIR, exist_ok=True)
_CACHE_FILE = os.path.join(CACHE_DIR, "entity_align_cache.json")


def _load_cache() -> dict:
    if not os.path.exists(_CACHE_FILE):
        return {}

    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"加载实体对齐缓存失败，将使用空缓存：{e}")
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存实体对齐缓存失败：{e}")


_align_cache: dict = _load_cache()


def _cache_key(entity_name: str, entity_type_hint: str) -> str:
    raw = f"{entity_name.lower()}|{entity_type_hint.lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_entity_index(kg_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    按 type 分组建立实体索引。
    返回 {type_str: DataFrame[index, id, type, name, source]}
    """
    records: list[pd.DataFrame] = []
    for cols in [
        ("x_index", "x_id", "x_type", "x_name", "x_source"),
        ("y_index", "y_id", "y_type", "y_name", "y_source"),
    ]:
        sub = kg_df[list(cols)].copy()
        sub.columns = ["index", "id", "type", "name", "source"]
        records.append(sub)

    all_entities = pd.concat(records, ignore_index=True).drop_duplicates(
        subset=["name", "type"]
    )

    index_dict: dict[str, pd.DataFrame] = {}
    for etype, grp in all_entities.groupby("type", dropna=False):
        index_dict[str(etype)] = grp.reset_index(drop=True)
    return index_dict


def _fuzzy_candidates(
    entity_name: str,
    entity_df: pd.DataFrame,
    top_k: int = TOP_K_CANDIDATES,
) -> list[dict]:
    """返回 Top-K 模糊匹配候选，每个包含 name/score/row 信息。"""
    if entity_df.empty:
        return []

    names = entity_df["name"].fillna("").astype(str).tolist()
    if not names:
        return []

    results = rfprocess.extract(
        entity_name,
        names,
        scorer=fuzz.token_sort_ratio,
        limit=top_k,
    )

    candidates: list[dict] = []
    for match_name, score, idx in results:
        row = entity_df.iloc[idx]
        candidates.append(
            {
                "name": str(match_name),
                "score": float(score),
                "index": int(row["index"]),
                "id": str(row["id"]),
                "type": str(row["type"]),
                "source": str(row["source"]),
            }
        )
    return candidates


def _get_embed_model():
    global _embed_model
    if _embed_model is None and _HAS_ST:
        _embed_model = SentenceTransformer("/mnt/gpu04_data/wangjie/KGC/Graphusion-main/SapBERT_local_test")
    return _embed_model


def _semantic_rerank(entity_name: str, candidates: list[dict]) -> list[dict]:
    """用语义相似度对候选列表重排，添加 sem_score 字段。"""
    if not candidates:
        return []

    model = _get_embed_model()
    if model is None:
        for c in candidates:
            c["sem_score"] = c["score"] / 100.0
        return candidates

    try:
        names = [entity_name] + [c["name"] for c in candidates]
        embeddings = model.encode(names, convert_to_tensor=True)
        query_emb = embeddings[0]
        cand_embs = embeddings[1:]
        scores = st_util.cos_sim(query_emb, cand_embs)[0].tolist()

        for c, s in zip(candidates, scores):
            c["sem_score"] = float(s)

        return sorted(candidates, key=lambda x: x["sem_score"], reverse=True)
    except Exception as e:
        logger.warning(f"语义重排失败，回退到 fuzzy 分数：{e}")
        for c in candidates:
            c["sem_score"] = c["score"] / 100.0
        return candidates


def _llm_align(
    entity_name: str,
    entity_type_hint: str,
    candidates: list[dict],
) -> dict | None:
    """
    调用 LLM 从候选列表中选出最匹配的实体，或判断无匹配（返回 None）。
    """
    if not candidates:
        return None

    cand_text = "\n".join(
        f"{i + 1}. name={c['name']}, type={c['type']}, "
        f"fuzzy_score={c['score']:.1f}, sem_score={c.get('sem_score', 0):.3f}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are a biomedical knowledge graph expert.

Task: Determine if the query entity matches any candidate entity from PrimeKG.

Query entity:
  Name: "{entity_name}"
  Type hint: "{entity_type_hint}"

Candidate entities from PrimeKG:
{cand_text}

Instructions:
1. Consider synonyms, abbreviations, and biomedical naming conventions.
2. If a candidate clearly refers to the same biomedical concept, select it.
3. If NO candidate is a valid match, output 0.

Output ONLY a JSON object, no explanation:
{{"choice": <1-{len(candidates)} or 0>, "confidence": <0.0-1.0>, "reason": "<brief>"}}
"""

    try:
        raw = call_llm(prompt, max_tokens=128)
        result = extract_json(raw)
        if not isinstance(result, dict):
            return None

        choice = int(result.get("choice", 0))
        if choice <= 0 or choice > len(candidates):
            return None

        return candidates[choice - 1]
    except Exception as e:
        logger.warning(f"LLM align failed for '{entity_name}': {e}")
        return None


def align_entity(
    entity_name: str,
    type_hint: str,
    entity_index: dict[str, pd.DataFrame],
    force_llm: bool = False,
) -> dict | None:
    """
    对单个实体进行对齐。

    参数:
        entity_name  : 待对齐实体名称
        type_hint    : LLM 推断的 type（primeKG 格式）
        entity_index : build_entity_index 返回的字典
        force_llm    : 强制走 LLM 确认

    返回:
        匹配到的 primeKG 实体字典（含 name/index/id/type/source），
        或 None（表示新实体，无匹配）
    """
    ck = _cache_key(entity_name, type_hint)
    if ck in _align_cache:
        logger.debug(f"[cache hit] {entity_name}")
        cached = _align_cache[ck]
        return None if cached == "NEW" else cached

    search_dfs: list[pd.DataFrame] = []
    if type_hint in entity_index:
        search_dfs.append(entity_index[type_hint])

    for t, df in entity_index.items():
        if t != type_hint:
            search_dfs.append(df)

    if not search_dfs:
        _align_cache[ck] = "NEW"
        _save_cache(_align_cache)
        return None

    combined_df = pd.concat(search_dfs, ignore_index=True).drop_duplicates(
        subset=["name", "type"]
    )

    candidates = _fuzzy_candidates(entity_name, combined_df)
    if not candidates:
        _align_cache[ck] = "NEW"
        _save_cache(_align_cache)
        return None

    best_fuzzy_score = candidates[0]["score"]

    if best_fuzzy_score >= FUZZY_MATCH_THRESHOLD and not force_llm:
        result = candidates[0]
        _align_cache[ck] = result
        _save_cache(_align_cache)
        logger.info(
            f"[fuzzy match] '{entity_name}' → '{result['name']}' "
            f"(score={best_fuzzy_score:.1f})"
        )
        return result

    candidates = _semantic_rerank(entity_name, candidates)
    best_sem_score = candidates[0].get("sem_score", 0.0)

    if best_sem_score >= EMBEDDING_SIM_THRESHOLD and not force_llm:
        result = candidates[0]
        _align_cache[ck] = result
        _save_cache(_align_cache)
        logger.info(
            f"[semantic match] '{entity_name}' → '{result['name']}' "
            f"(sem={best_sem_score:.3f})"
        )
        return result

    if best_fuzzy_score >= LLM_CONFIRM_THRESHOLD * 100 or force_llm:
        result = _llm_align(entity_name, type_hint, candidates[:TOP_K_CANDIDATES])
        _align_cache[ck] = result if result else "NEW"
        _save_cache(_align_cache)
        if result:
            logger.info(f"[LLM match] '{entity_name}' → '{result['name']}'")
        else:
            logger.info(f"[NEW entity] '{entity_name}' (no match found)")
        return result

    _align_cache[ck] = "NEW"
    _save_cache(_align_cache)
    logger.info(f"[NEW entity] '{entity_name}' (low score={best_fuzzy_score:.1f})")
    return None


def infer_entity_type_llm(
    entity_name: str,
    context: str = "",
) -> str:
    """
    调用 LLM 推断实体的 primeKG type。
    context 为三元组上下文（predicate + 对端实体）。
    """
    ck = _cache_key(f"TYPE|{entity_name}", context)
    if ck in _align_cache:
        cached = _align_cache[ck]
        if isinstance(cached, str):
            return cached

    prompt = f"""You are a biomedical knowledge graph expert.

Given a biomedical entity, classify it into ONE of these PrimeKG types:
- gene/protein
- disease
- drug
- effect/phenotype
- biological_process
- cellular_component
- molecular_function
- pathway
- anatomy
- exposure

Entity name: "{entity_name}"
Context (relation and partner entity): "{context}"

Output ONLY a JSON object:
{{"type": "<one of the 10 types above>", "confidence": <0.0-1.0>}}
"""

    try:
        raw = call_llm(prompt, max_tokens=64)
        result = extract_json(raw)
        if isinstance(result, dict):
            etype = str(result.get("type", "gene/protein")).strip() or "gene/protein"
            _align_cache[ck] = etype
            _save_cache(_align_cache)
            return etype
    except Exception as e:
        logger.warning(f"LLM type inference failed for '{entity_name}': {e}")

    return "gene/protein"