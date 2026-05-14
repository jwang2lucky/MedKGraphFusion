from __future__ import annotations

"""
将自定义三元组的 predicate 映射到 primeKG (relation, display_relation)。

映射策略（三层）：
  Layer 1: predicate 精确映射表（覆盖当前给出的 25 种 predicate）
  Layer 2: (x_type, y_type) 类型对约束 → 优先选择最贴近的 native relation
  Layer 3: 若 primeKG 原生 relation 无法忠实表达语义 → 自动注册扩展 relation

扩展 relation 记录在 config.EXTENDED_RELATIONS 中，格式：
{
    "ext_xxx": {
        "display_relation": "...",
        "description": "..."
    }
}
"""

import logging
import re

from config import EXTENDED_RELATIONS, PRIMEKG_NATIVE_RELATIONS

logger = logging.getLogger(__name__)


def _normalize_predicate(predicate: str) -> str:
    """
    统一 predicate 形式：
    - 小写
    - 空格 / 下划线 归一到连字符
    - 压缩重复连字符
    """
    text = str(predicate or "").strip().lower()
    text = text.replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text


def _sanitize_relation_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "custom_relation"


def _register_extended_relation(
    relation: str,
    display_relation: str,
    predicate: str,
    x_type: str,
    y_type: str,
    description: str = "",
) -> str:
    """
    注册扩展 relation，返回最终 relation 名称。
    与 config.EXTENDED_RELATIONS 的结构完全对齐。
    """
    rel = _sanitize_relation_name(relation)
    if not rel.startswith("ext_"):
        rel = f"ext_{rel}"

    if rel not in EXTENDED_RELATIONS:
        EXTENDED_RELATIONS[rel] = {
            "display_relation": display_relation,
            "description": description or (
                f"Extended relation for predicate='{predicate}' ({x_type} -> {y_type})"
            ),
        }
        logger.info(
            f"[RelationMapper] Registered extended relation: '{rel}' "
            f"for predicate='{predicate}' ({x_type} -> {y_type})"
        )
    return rel


def _finalize_relation(
    relation: str,
    display_relation: str,
    predicate: str,
    x_type: str,
    y_type: str,
    description: str = "",
) -> tuple[str, str]:
    """
    如果是 primeKG 原生 relation，直接返回；
    否则自动注册为扩展 relation。
    """
    if relation in PRIMEKG_NATIVE_RELATIONS:
        return relation, display_relation

    rel = _register_extended_relation(
        relation=relation,
        display_relation=display_relation,
        predicate=predicate,
        x_type=x_type,
        y_type=y_type,
        description=description,
    )
    return rel, display_relation


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1: predicate 精确映射
# 这里严格覆盖你给出的 25 种 predicate
# ══════════════════════════════════════════════════════════════════════════════
_PREDICATE_MAP: dict[str, dict] = {
    # 1. Treats
    "treats": {
        "relation": "indication",
        "display_relation": "indication",
        "description": "Therapeutic application: intervention -> disease/phenotype",
    },

    # 2. Causes
    "causes": {
        "relation": None,
        "display_relation": "causes",
        "description": "Etiological causation",
        "type_pair_rules": {
            ("drug", "effect/phenotype"): ("drug_effect", "side effect"),
            ("drug", "disease"): ("drug_effect", "side effect"),
            ("exposure", "disease"): ("exposure_disease", "linked to"),
            ("exposure", "biological_process"): ("exposure_bioprocess", "associated with"),
            ("exposure", "cellular_component"): ("exposure_cellcomp", "associated with"),
            ("exposure", "molecular_function"): ("exposure_molfunc", "associated with"),
            ("exposure", "gene/protein"): ("exposure_protein", "associated with"),
            ("disease", "effect/phenotype"): ("disease_phenotype_positive", "phenotype present"),
        },
        "default": ("ext_causes", "causes"),
    },

    # 3. Associated-With
    "associated-with": {
        "relation": None,
        "display_relation": "associated with",
        "description": "General non-causal association",
        "symmetric": True,
        "type_pair_rules": {
            ("disease", "disease"): ("disease_disease", "associated with"),
            ("disease", "effect/phenotype"): ("disease_phenotype_positive", "phenotype present"),
            ("disease", "gene/protein"): ("disease_protein", "associated with"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
            ("drug", "drug"): ("drug_drug", "synergistic interaction"),
            ("drug", "gene/protein"): ("drug_protein", "target"),
            ("exposure", "disease"): ("exposure_disease", "linked to"),
            ("exposure", "gene/protein"): ("exposure_protein", "associated with"),
            ("exposure", "biological_process"): ("exposure_bioprocess", "associated with"),
            ("exposure", "cellular_component"): ("exposure_cellcomp", "associated with"),
            ("exposure", "molecular_function"): ("exposure_molfunc", "associated with"),
            ("pathway", "gene/protein"): ("pathway_protein", "associated with"),
            ("biological_process", "gene/protein"): ("bioprocess_protein", "associated with"),
            ("cellular_component", "gene/protein"): ("cellcomp_protein", "associated with"),
            ("molecular_function", "gene/protein"): ("molfunc_protein", "associated with"),
            ("effect/phenotype", "effect/phenotype"): ("phenotype_phenotype", "associated with"),
            ("effect/phenotype", "gene/protein"): ("phenotype_protein", "associated with"),
        },
        "default": ("ext_associated_with", "associated with"),
    },

    # 4. Expressed-In
    "expressed-in": {
        "relation": None,
        "display_relation": "expression present",
        "description": "Gene/protein expression localization",
        "type_pair_rules": {
            ("gene/protein", "anatomy"): ("anatomy_protein_present", "expression present"),
        },
        "default": ("ext_expressed_in", "expressed in"),
    },

    # 5. Located-In
    "located-in": {
        "relation": None,
        "display_relation": "located in",
        "description": "Anatomical localization",
        "type_pair_rules": {
            ("gene/protein", "anatomy"): ("anatomy_protein_present", "expression present"),
            ("anatomy", "anatomy"): ("anatomy_anatomy", "parent-child"),
        },
        "default": ("ext_located_in", "located in"),
    },

    # 6. Part-Of
    "part-of": {
        "relation": None,
        "display_relation": "parent-child",
        "description": "Part-whole relation",
        "type_pair_rules": {
            ("anatomy", "anatomy"): ("anatomy_anatomy", "parent-child"),
            ("disease", "disease"): ("disease_disease", "parent-child"),
            ("biological_process", "biological_process"): ("bioprocess_bioprocess", "parent-child"),
            ("pathway", "pathway"): ("pathway_pathway", "parent-child"),
            ("cellular_component", "cellular_component"): ("cellcomp_cellcomp", "parent-child"),
            ("molecular_function", "molecular_function"): ("molfunc_molfunc", "parent-child"),
            ("effect/phenotype", "effect/phenotype"): ("phenotype_phenotype", "parent-child"),
        },
        "default": ("ext_part_of", "part of"),
    },

    # 7. Subtype-Of
    "subtype-of": {
        "relation": None,
        "display_relation": "parent-child",
        "description": "Subtype hierarchy",
        "type_pair_rules": {
            ("disease", "disease"): ("disease_disease", "parent-child"),
            ("anatomy", "anatomy"): ("anatomy_anatomy", "parent-child"),
            ("biological_process", "biological_process"): ("bioprocess_bioprocess", "parent-child"),
            ("pathway", "pathway"): ("pathway_pathway", "parent-child"),
            ("cellular_component", "cellular_component"): ("cellcomp_cellcomp", "parent-child"),
            ("molecular_function", "molecular_function"): ("molfunc_molfunc", "parent-child"),
            ("effect/phenotype", "effect/phenotype"): ("phenotype_phenotype", "parent-child"),
        },
        "default": ("ext_subtype_of", "subtype of"),
    },

    # 8. Interacts-With
    "interacts-with": {
        "relation": None,
        "display_relation": "interacts with",
        "description": "Physical or functional interaction",
        "symmetric": True,
        "type_pair_rules": {
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
            ("drug", "drug"): ("drug_drug", "synergistic interaction"),
            ("drug", "gene/protein"): ("drug_protein", "target"),
        },
        "default": ("ext_interacts_with", "interacts with"),
    },

    # 9. Inhibits
    "inhibits": {
        "relation": None,
        "display_relation": "inhibits",
        "description": "Negative regulation",
        "type_pair_rules": {
            ("drug", "gene/protein"): ("drug_protein", "target"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
            ("pathway", "gene/protein"): ("pathway_protein", "associated with"),
            ("biological_process", "gene/protein"): ("bioprocess_protein", "associated with"),
            ("molecular_function", "gene/protein"): ("molfunc_protein", "associated with"),
        },
        "default": ("ext_inhibits", "inhibits"),
    },

    # 10. Activates
    "activates": {
        "relation": None,
        "display_relation": "activates",
        "description": "Positive regulation",
        "type_pair_rules": {
            ("drug", "gene/protein"): ("drug_protein", "target"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
            ("pathway", "gene/protein"): ("pathway_protein", "associated with"),
            ("biological_process", "gene/protein"): ("bioprocess_protein", "associated with"),
            ("molecular_function", "gene/protein"): ("molfunc_protein", "associated with"),
        },
        "default": ("ext_activates", "activates"),
    },

    # 11. Diagnoses
    "diagnoses": {
        "relation": "ext_diagnoses",
        "display_relation": "diagnoses",
        "description": "Diagnostic utility: tool/test -> disease",
    },

    # 12. Has-Symptom
    "has-symptom": {
        "relation": "disease_phenotype_positive",
        "display_relation": "phenotype present",
        "description": "Disease manifests symptom/phenotype",
    },

    # 13. Used-For
    "used-for": {
        "relation": None,
        "display_relation": "used for",
        "description": "General medical purpose",
        "type_pair_rules": {
            ("drug", "disease"): ("indication", "indication"),
        },
        "default": ("ext_used_for", "used for"),
    },

    # 14. Contraindicated-For
    "contraindicated-for": {
        "relation": "contraindication",
        "display_relation": "contraindication",
        "description": "Drug/procedure contraindication",
    },

    # 15. Biomarker-Of
    "biomarker-of": {
        "relation": "ext_biomarker_of",
        "display_relation": "biomarker of",
        "description": "Biomarker indicates disease or disease state",
    },

    # 16. Agonism-or-Antagonism
    "agonism-or-antagonism": {
        "relation": None,
        "display_relation": "target",
        "description": "Receptor modulation",
        "type_pair_rules": {
            ("drug", "gene/protein"): ("drug_protein", "target"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
        },
        "default": ("ext_agonism_or_antagonism", "agonism or antagonism"),
    },

    # 17. Binding
    "binding": {
        "relation": None,
        "display_relation": "binding",
        "description": "Physical binding",
        "type_pair_rules": {
            ("drug", "gene/protein"): ("drug_protein", "target"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
            ("drug", "drug"): ("drug_drug", "synergistic interaction"),
        },
        "default": ("ext_binding", "binding"),
    },

    # 18. Affects-Expression
    "affects-expression": {
        "relation": None,
        "display_relation": "affects expression",
        "description": "Expression-level regulation",
        "type_pair_rules": {
            ("drug", "gene/protein"): ("drug_protein", "target"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
            ("pathway", "gene/protein"): ("pathway_protein", "associated with"),
            ("biological_process", "gene/protein"): ("bioprocess_protein", "associated with"),
            ("molecular_function", "gene/protein"): ("molfunc_protein", "associated with"),
        },
        "default": ("ext_affects_expression", "affects expression"),
    },

    # 19. Metabolizes
    "metabolizes": {
        "relation": None,
        "display_relation": "metabolizes",
        "description": "Metabolic conversion",
        "type_pair_rules": {
            ("gene/protein", "drug"): ("drug_protein", "target"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
        },
        "default": ("ext_metabolizes", "metabolizes"),
    },

    # 20. Transports
    "transports": {
        "relation": None,
        "display_relation": "transports",
        "description": "Molecular transport",
        "type_pair_rules": {
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
        },
        "default": ("ext_transports", "transports"),
    },

    # 21. Enzyme-Activity
    "enzyme-activity": {
        "relation": None,
        "display_relation": "enzyme activity",
        "description": "Modulation of enzyme activity",
        "type_pair_rules": {
            ("drug", "gene/protein"): ("drug_protein", "target"),
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
        },
        "default": ("ext_enzyme_activity", "enzyme activity"),
    },

    # 22. Side-Effect
    "side-effect": {
        "relation": None,
        "display_relation": "side effect",
        "description": "Adverse reaction",
        "type_pair_rules": {
            ("drug", "effect/phenotype"): ("drug_effect", "side effect"),
            ("drug", "disease"): ("drug_effect", "side effect"),
        },
        "default": ("ext_side_effect", "side effect"),
    },

    # 23. Contributes-To-Pathogenesis
    "contributes-to-pathogenesis": {
        "relation": None,
        "display_relation": "contributes to pathogenesis",
        "description": "Mechanism/pathway contributes to disease",
        "type_pair_rules": {
            ("pathway", "disease"): ("disease_disease", "associated with"),
            ("biological_process", "disease"): ("disease_disease", "associated with"),
            ("gene/protein", "disease"): ("disease_protein", "associated with"),
        },
        "default": ("ext_contributes_to_pathogenesis", "contributes to pathogenesis"),
    },

    # 24. Mutation-Affects-Disease
    "mutation-affects-disease": {
        "relation": None,
        "display_relation": "affects disease risk",
        "description": "Mutation/variant modulates disease risk",
        "type_pair_rules": {
            ("gene/protein", "disease"): ("disease_protein", "associated with"),
        },
        "default": ("ext_mutation_affects_disease", "affects disease risk"),
    },

    # 25. Gene-Regulates-Gene
    "gene-regulates-gene": {
        "relation": None,
        "display_relation": "regulates",
        "description": "Gene regulatory network",
        "type_pair_rules": {
            ("gene/protein", "gene/protein"): ("protein_protein", "ppi"),
        },
        "default": ("ext_gene_regulates_gene", "regulates"),
    },
}


# 兼容不同写法
_ALIAS_MAP = {
    "associated_with": "associated-with",
    "expressed_in": "expressed-in",
    "located_in": "located-in",
    "part_of": "part-of",
    "subtype_of": "subtype-of",
    "interacts_with": "interacts-with",
    "has_symptom": "has-symptom",
    "used_for": "used-for",
    "contraindicated_for": "contraindicated-for",
    "biomarker_of": "biomarker-of",
    "agonism_or_antagonism": "agonism-or-antagonism",
    "affects_expression": "affects-expression",
    "enzyme_activity": "enzyme-activity",
    "side_effect": "side-effect",
    "contributes_to_pathogenesis": "contributes-to-pathogenesis",
    "mutation_affects_disease": "mutation-affects-disease",
    "gene_regulates_gene": "gene-regulates-gene",
}


def _resolve_rule(rule: dict, predicate: str, x_type: str, y_type: str) -> tuple[str, str]:
    x_type = str(x_type or "").strip()
    y_type = str(y_type or "").strip()

    if rule.get("relation"):
        return _finalize_relation(
            relation=rule["relation"],
            display_relation=rule["display_relation"],
            predicate=predicate,
            x_type=x_type,
            y_type=y_type,
            description=rule.get("description", ""),
        )

    pair_rules = rule.get("type_pair_rules", {})
    pair = (x_type, y_type)

    if pair in pair_rules:
        rel, disp = pair_rules[pair]
        return _finalize_relation(
            relation=rel,
            display_relation=disp,
            predicate=predicate,
            x_type=x_type,
            y_type=y_type,
            description=rule.get("description", ""),
        )

    if rule.get("symmetric"):
        rev_pair = (y_type, x_type)
        if rev_pair in pair_rules:
            rel, disp = pair_rules[rev_pair]
            return _finalize_relation(
                relation=rel,
                display_relation=disp,
                predicate=predicate,
                x_type=x_type,
                y_type=y_type,
                description=rule.get("description", ""),
            )

    rel, disp = rule.get(
        "default",
        (f"ext_{_sanitize_relation_name(predicate)}", predicate.replace("-", " ")),
    )
    return _finalize_relation(
        relation=rel,
        display_relation=disp,
        predicate=predicate,
        x_type=x_type,
        y_type=y_type,
        description=rule.get("description", ""),
    )


def map_relation(
    predicate: str,
    x_type: str,
    y_type: str,
) -> tuple[str, str]:
    """
    将输入 predicate 映射到 (relation, display_relation)。

    参数:
        predicate: 原始输入谓词
        x_type:    主语实体 type（primeKG type）
        y_type:    宾语实体 type（primeKG type）

    返回:
        (relation, display_relation)
    """
    pred_norm = _normalize_predicate(predicate)
    pred_norm = _ALIAS_MAP.get(pred_norm, pred_norm)

    rule = _PREDICATE_MAP.get(pred_norm)
    if rule is not None:
        relation, display_relation = _resolve_rule(rule, pred_norm, x_type, y_type)
        logger.debug(
            f"[RelationMapper] '{predicate}' ({x_type} -> {y_type}) -> "
            f"({relation}, {display_relation})"
        )
        return relation, display_relation

    relation, display_relation = _finalize_relation(
        relation=f"ext_{_sanitize_relation_name(pred_norm)}",
        display_relation=pred_norm.replace("-", " "),
        predicate=pred_norm,
        x_type=x_type,
        y_type=y_type,
        description=f"Fallback extended relation for unknown predicate '{predicate}'",
    )
    logger.warning(
        f"[RelationMapper] Unknown predicate '{predicate}', fallback to "
        f"extended relation '{relation}'"
    )
    return relation, display_relation


def get_all_relations() -> set[str]:
    """
    返回当前可用的全部 relation（原生 + 已注册扩展）。
    """
    return set(PRIMEKG_NATIVE_RELATIONS) | set(EXTENDED_RELATIONS.keys())