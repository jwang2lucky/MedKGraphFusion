# type_mapper.py
"""
将自定义三元组中的实体 type 映射到 primeKG type。

策略（按优先级）：
1. 若 raw_type 已在 PRIMEKG_NATIVE_TYPES → 直接返回
2. 直接归一化映射表（字符串等价）
3. 关键词规则匹配 → 映射到最近的合法 type
4. 调用 LLM 判断是否能归类到合法 type
5. 确实无法归类 → 注册为扩展 type 并返回

扩展 type 会写入 config.EXTENDED_TYPES，
最终由 merger.py 输出到单独的报告文件。
"""
from __future__ import annotations

import logging
import re

from config import PRIMEKG_NATIVE_TYPES, EXTENDED_TYPES

logger = logging.getLogger(__name__)

# ── 直接等价映射（小写规范化后） ───────────────────────────────────────────────
_DIRECT_MAP: dict[str, str] = {
    # gene/protein
    "gene":                  "gene/protein",
    "protein":               "gene/protein",
    "gene/protein":          "gene/protein",
    "enzyme":                "gene/protein",
    "receptor":              "gene/protein",
    "kinase":                "gene/protein",
    "antibody":              "gene/protein",
    "peptide":               "gene/protein",
    "hormone":               "gene/protein",
    "transcription factor":  "gene/protein",
    "biomarker":             "gene/protein",
    "genetic variant":       "gene/protein",
    "mutation":              "gene/protein",
    # disease
    "disease":               "disease",
    "disorder":              "disease",
    "syndrome":              "disease",
    "infection":             "disease",
    "cancer":                "disease",
    "tumor":                 "disease",
    "condition":             "disease",
    "pathology":             "disease",
    "injury":                "disease",
    # drug
    "drug":                  "drug",
    "medication":            "drug",
    "pharmaceutical":        "drug",
    "compound":              "drug",
    "inhibitor":             "drug",
    "therapy":               "drug",
    "treatment":             "drug",
    "medical device":        "drug",
    "device":                "drug",
    "anticoagulant":         "drug",
    "vaccine":               "drug",
    "supplement":            "drug",
    # effect/phenotype
    "phenotype":             "effect/phenotype",
    "symptom":               "effect/phenotype",
    "sign":                  "effect/phenotype",
    "side effect":           "effect/phenotype",
    "adverse effect":        "effect/phenotype",
    "clinical feature":      "effect/phenotype",
    "effect/phenotype":      "effect/phenotype",
    # anatomy
    "anatomy":               "anatomy",
    "organ":                 "anatomy",
    "tissue":                "anatomy",
    "cell type":             "anatomy",
    "anatomical structure":  "anatomy",
    "body part":             "anatomy",
    "anatomical location":   "anatomy",
    # biological_process
    "biological_process":    "biological_process",
    "biological process":    "biological_process",
    "process":               "biological_process",
    "pathway":               "pathway",
    "signaling pathway":     "pathway",
    "metabolic pathway":     "pathway",
    # cellular_component
    "cellular_component":    "cellular_component",
    "cellular component":    "cellular_component",
    "organelle":             "cellular_component",
    "membrane":              "cellular_component",
    # molecular_function
    "molecular_function":    "molecular_function",
    "molecular function":    "molecular_function",
    "function":              "molecular_function",
    # exposure
    "exposure":              "exposure",
    "toxin":                 "exposure",
    "chemical":              "exposure",
    "pollutant":             "exposure",
    "environmental factor":  "exposure",
}

# ── 关键词规则（优先级从上到下） ──────────────────────────────────────────────
_KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["gene", "protein", "enzyme", "receptor", "kinase", "channel",
      "transporter protein", "antibody", "peptide", "hormone",
      "transcription", "biomarker", "variant", "mutation",
      "psa", "egfr", "brca", "her2"], "gene/protein"),

    (["disease", "disorder", "syndrome", "infection", "cancer",
      "tumor", "carcinoma", "sarcoma", "lymphoma", "leukemia",
      "insufficiency", "failure", "thrombosis", "embolism",
      "injury", "patholog", "lesion", "malignancy"], "disease"),

    (["drug", "medication", "pharmaceutical", "compound", "inhibitor",
      "agonist", "antagonist", "anticoagulant", "antiplatelet",
      "antibiotic", "vaccine", "therapy", "treatment",
      "medical device", "filter", "catheter", "implant",
      "supplement", "capsule", "tablet", "injection"], "drug"),

    (["symptom", "sign", "phenotype", "side effect", "adverse",
      "clinical feature", "manifestation", "presentation",
      "pain", "swelling", "fever", "edema", "strain",
      "nausea", "fatigue", "dyspnea", "tachycardia"], "effect/phenotype"),

    (["anatomy", "organ", "tissue", "vein", "artery", "heart",
      "lung", "liver", "kidney", "extremit", "vena cava",
      "vessel", "muscle", "bone", "brain", "nerve",
      "cell type", "anatomical", "body part", "region",
      "pancreatic", "alveolus", "colon", "stomach"], "anatomy"),

    (["biological process", "process", "coagulation", "thrombogenesis",
      "fibrinolysis", "apoptosis", "proliferation", "inflammation",
      "metabolism", "regulation", "signaling", "immune response",
      "prevention", "pathogenesis", "mechanism"], "biological_process"),

    (["pathway", "signaling pathway", "metabolic pathway",
      "wnt pathway", "mapk", "pi3k"], "pathway"),

    (["cell", "membrane", "nucleus", "mitochondria", "ribosome",
      "cytoplasm", "organelle", "cellular component"], "cellular_component"),

    (["molecular function", "binding", "catalysis",
      "enzymatic activity", "transport activity"], "molecular_function"),

    (["exposure", "toxin", "chemical", "pollutant",
      "radiation", "environmental", "stressor"], "exposure"),
]


def _keywords_match(text: str) -> str | None:
    """关键词匹配，返回 primeKG type 或 None。"""
    text_lower = text.lower()
    for keywords, mapped_type in _KEYWORD_RULES:
        for kw in keywords:
            if re.search(re.escape(kw), text_lower):
                return mapped_type
    return None


def _register_extended_type(raw_type: str, entity_name: str) -> str:
    """
    注册扩展 type。
    规范化为 snake_case，记录到 config.EXTENDED_TYPES。
    """
    # 规范化
    ext_type = raw_type.strip().lower().replace(" ", "_").replace("/", "_")
    if ext_type not in EXTENDED_TYPES:
        EXTENDED_TYPES[ext_type] = (
            f"Extended type inferred from entity '{entity_name}', "
            f"raw_type='{raw_type}'"
        )
        logger.info(f"[TypeMapper] Registered new extended type: '{ext_type}' "
                    f"(entity='{entity_name}')")
    return ext_type


def map_type(
    raw_type: str,
    entity_name: str = "",
    use_llm: bool = False,
) -> str:
    """
    主入口：将 LLM 返回的 raw_type 映射到 primeKG type 或扩展 type。

    参数:
        raw_type    : LLM 推断的原始类型字符串
        entity_name : 实体名称（辅助判断）
        use_llm     : 是否允许二次调用 LLM 确认（默认关，避免递归）

    返回:
        primeKG 合法 type 或注册后的扩展 type 字符串
    """
    # 1. 已合法 → 直接返回
    if raw_type in PRIMEKG_NATIVE_TYPES:
        return raw_type

    # 2. 直接等价映射
    normalized = raw_type.strip().lower()
    if normalized in _DIRECT_MAP:
        return _DIRECT_MAP[normalized]

    # 3. 关键词规则
    combined_text = f"{raw_type} {entity_name}"
    result = _keywords_match(combined_text)
    if result:
        logger.debug(f"[TypeMapper] '{entity_name}' ({raw_type}) → '{result}' via keywords")
        return result

    # 4. 可选 LLM 二次确认（避免在 entity_aligner 内部递归调用）
    if use_llm:
        try:
            from llm_client import call_llm, extract_json
            prompt = f"""Classify the biomedical entity into one of these PrimeKG types:
{', '.join(sorted(PRIMEKG_NATIVE_TYPES))}

Entity: "{entity_name}"
Inferred raw type: "{raw_type}"

If it fits one of the above types, return that type.
If it genuinely does NOT fit any of the above, return "new_type" with a suggested name.

Output ONLY JSON: {{"type": "<primekg_type_or_new_type>", "new_type_name": "<only if new_type>"}}"""
            raw_resp = call_llm(prompt)
            parsed = extract_json(raw_resp)
            if parsed and isinstance(parsed, dict):
                suggested = parsed.get("type", "").strip()
                if suggested in PRIMEKG_NATIVE_TYPES:
                    logger.info(f"[TypeMapper] LLM mapped '{entity_name}' → '{suggested}'")
                    return suggested
                if suggested == "new_type":
                    new_name = parsed.get("new_type_name", raw_type)
                    return _register_extended_type(new_name, entity_name)
        except Exception as e:
            logger.warning(f"[TypeMapper] LLM fallback failed: {e}")

    # 5. 兜底：尝试用实体名单独做关键词匹配
    result = _keywords_match(entity_name)
    if result:
        logger.debug(f"[TypeMapper] '{entity_name}' → '{result}' via entity name keywords")
        return result

    # 6. 确实无法归类 → 注册扩展 type
    return _register_extended_type(raw_type if raw_type else entity_name, entity_name)


def get_all_valid_types() -> set[str]:
    """返回所有合法 type（原生 + 已注册扩展）。"""
    return PRIMEKG_NATIVE_TYPES | set(EXTENDED_TYPES.keys())