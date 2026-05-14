from __future__ import annotations

"""
主合并逻辑：
1. 加载 primeKG kg.csv 和输入 jsonl
2. 对每个三元组做实体对齐、type 推断、relation 映射
3. 冲突检测
4. 写出合并后的 CSV
"""

import json
import logging

import pandas as pd
from tqdm import tqdm

from config import (
    INPUT_JSONL_PATH,
    KG_CSV_PATH,
    MIN_SUPPORT,
    OUTPUT_CSV_PATH,
)
from conflict_resolver import ConflictResolver
from entity_aligner import (
    align_entity,
    build_entity_index,
    infer_entity_type_llm,
)
from relation_mapper import map_relation
from type_mapper import map_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_kg(path: str) -> pd.DataFrame:
    logger.info(f"Loading primeKG from {path} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    for col in ["x_index", "y_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    logger.info(f"  Loaded {len(df):,} triples.")
    return df


def load_input_jsonl(path: str) -> list[dict]:
    triples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    logger.info(f"Loaded {len(triples)} input triples from {path}.")
    return triples


def process_triple(
    triple: dict,
    entity_index: dict[str, pd.DataFrame],
    existing_df: pd.DataFrame,
    resolver: ConflictResolver,
) -> tuple[dict | None, str]:
    """
    处理单个输入三元组。

    返回:
        (row, "added")      → 正常新增
        (row, "conflict")   → 存在关系冲突，但保留并标注
        (None, "duplicate") → 完全重复，跳过
        (None, "skipped")   → 其他跳过（如 support 不足）
    """
    s_name = str(triple["s"]).strip()
    p_name = str(triple["p"]).strip()
    o_name = str(triple["o"]).strip()
    support = int(triple.get("support", 1))

    if support < MIN_SUPPORT:
        logger.debug(f"Skip (low support={support}): {s_name} -{p_name}-> {o_name}")
        return None, "skipped"

    context_s = f"{p_name} -> {o_name}"
    context_o = f"{s_name} -> {p_name}"

    raw_type_s = infer_entity_type_llm(s_name, context_s)
    raw_type_o = infer_entity_type_llm(o_name, context_o)

    type_s = map_type(raw_type_s, s_name)
    type_o = map_type(raw_type_o, o_name)

    aligned_s = align_entity(s_name, type_s, entity_index)
    aligned_o = align_entity(o_name, type_o, entity_index)

    if aligned_s:
        final_s_name = aligned_s["name"]
        final_s_type = aligned_s["type"]
        final_s_index = int(aligned_s["index"])
        final_s_id = str(aligned_s["id"])
        final_s_source = aligned_s["source"]
    else:
        final_s_name = s_name
        final_s_type = type_s
        final_s_index, final_s_id = resolver.get_or_create_entity_id(
            s_name, type_s, existing_df, source="CUSTOM"
        )
        final_s_source = "CUSTOM"

    if aligned_o:
        final_o_name = aligned_o["name"]
        final_o_type = aligned_o["type"]
        final_o_index = int(aligned_o["index"])
        final_o_id = str(aligned_o["id"])
        final_o_source = aligned_o["source"]
    else:
        final_o_name = o_name
        final_o_type = type_o
        final_o_index, final_o_id = resolver.get_or_create_entity_id(
            o_name, type_o, existing_df, source="CUSTOM"
        )
        final_o_source = "CUSTOM"

    relation, display_relation = map_relation(p_name, final_s_type, final_o_type)

    candidate_row = {
        "x_name": final_s_name,
        "y_name": final_o_name,
        "relation": relation,
    }
    status, conflict_msg = resolver.check(candidate_row)

    if status == "duplicate":
        logger.info(f"[SKIP duplicate] {final_s_name} -{relation}-> {final_o_name}")
        return None, "duplicate"

    if status == "conflict":
        logger.warning(f"[CONFLICT kept] {conflict_msg}")
        final_s_source = f"{final_s_source}|CONFLICT"

    result = {
        "relation": relation,
        "display_relation": display_relation,
        "x_index": final_s_index,
        "x_id": final_s_id,
        "x_type": final_s_type,
        "x_name": final_s_name,
        "x_source": final_s_source,
        "y_index": final_o_index,
        "y_id": final_o_id,
        "y_type": final_o_type,
        "y_name": final_o_name,
        "y_source": final_o_source,
        "_origins": json.dumps(triple.get("origins", []), ensure_ascii=False),
        "_support": support,
        "_input_predicate": p_name,
    }
    return result, ("conflict" if status == "conflict" else "added")


def merge(
    kg_path: str = KG_CSV_PATH,
    input_path: str = INPUT_JSONL_PATH,
    output_path: str = OUTPUT_CSV_PATH,
    save_extra_cols: bool = False,
):
    original_kg_df = load_kg(kg_path)
    kg_df = original_kg_df.copy()
    triples = load_input_jsonl(input_path)

    entity_index = build_entity_index(kg_df)
    resolver = ConflictResolver(kg_df)

    new_rows: list[dict] = []
    stats = {
        "total": 0,
        "added": 0,
        "duplicate": 0,
        "conflict": 0,
        "skipped": 0,
    }

    for triple in tqdm(triples, desc="Processing triples"):
        stats["total"] += 1

        result, result_status = process_triple(triple, entity_index, kg_df, resolver)

        if result is None:
            stats[result_status] += 1
            continue

        new_rows.append(result)
        stats["added"] += 1
        if result_status == "conflict":
            stats["conflict"] += 1

        # 回灌运行时状态，使后续 triple 能看到本批新增实体/关系
        resolver.register_row(result)
        kg_df = pd.concat([kg_df, pd.DataFrame([result])], ignore_index=True)
        entity_index = build_entity_index(kg_df)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Statistics: {stats}")
    logger.info(f"{'=' * 50}")

    if not new_rows:
        logger.info("No new triples to add. Output equals input.")
        original_kg_df.to_csv(output_path, index=False)
        return

    new_df = pd.DataFrame(new_rows)

    primekg_cols = [
        "relation",
        "display_relation",
        "x_index",
        "x_id",
        "x_type",
        "x_name",
        "x_source",
        "y_index",
        "y_id",
        "y_type",
        "y_name",
        "y_source",
    ]
    extra_cols = ["_origins", "_support", "_input_predicate"]
    output_cols = primekg_cols + extra_cols if save_extra_cols else primekg_cols

    for col in output_cols:
        if col not in new_df.columns:
            new_df[col] = ""

    merged_df = pd.concat(
        [
            original_kg_df.reindex(columns=output_cols),
            new_df[output_cols],
        ],
        ignore_index=True,
    )

    merged_df.to_csv(output_path, index=False)
    logger.info(
        f"Merged KG saved to {output_path} "
        f"({len(original_kg_df):,} + {len(new_rows)} = {len(merged_df):,} triples)"
    )

    summary_path = output_path.replace(".csv", "_new_triples.csv")
    new_df[output_cols].to_csv(summary_path, index=False)
    logger.info(f"New triples summary saved to {summary_path}")