from __future__ import annotations

"""
冲突检测与解决。

冲突类型：
1. 重复三元组（完全相同的 x_name, relation, y_name）→ 跳过
2. 关系冲突（同一实体对存在矛盾关系，如 indication vs contraindication）→ 保留两者并标注
3. 实体 ID/Index 复用：优先复用已存在实体；新实体使用运行时全局递增 index
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

_CONFLICTING_RELATIONS: list[frozenset[str]] = [
    frozenset({"indication", "contraindication"}),
    frozenset({"anatomy_protein_present", "anatomy_protein_absent"}),
    frozenset({"disease_phenotype_positive", "disease_phenotype_negative"}),
]


class ConflictResolver:
    def __init__(self, existing_df: pd.DataFrame):
        """
        existing_df: 已有 primeKG DataFrame，需含
                     [x_name, y_name, relation, display_relation] 等列
        """
        self.existing_df = existing_df.copy()

        self._pair_index: dict[tuple[str, str], set[str]] = {}
        self._entity_cache: dict[tuple[str, str], tuple[int, str]] = {}
        self._next_index = self.resolve_new_index(existing_df)

        for _, row in existing_df.iterrows():
            self.register_row(row, update_df=False)

    def _normalize_name(self, name: str) -> str:
        return str(name).strip().lower()

    def _slugify_entity_id(self, name: str) -> str:
        slug = self._normalize_name(name)
        slug = slug.replace("/", "_")
        slug = re.sub(r"\s+", "_", slug)
        slug = re.sub(r"[^a-z0-9_]+", "_", slug)
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug or "custom_entity"

    def _register_entity_side(
        self,
        name: str,
        etype: str,
        index: int,
        entity_id: str,
    ) -> None:
        key = (self._normalize_name(name), str(etype))
        self._entity_cache[key] = (int(index), str(entity_id))

    def register_row(self, row: dict | pd.Series, update_df: bool = True) -> None:
        """
        将一条已确认保留的三元组注册到运行时索引中，
        使后续输入可以看到当前批次新增的数据。
        """
        x_name = str(row["x_name"])
        y_name = str(row["y_name"])
        relation = str(row["relation"])

        pair_key = (self._normalize_name(x_name), self._normalize_name(y_name))
        self._pair_index.setdefault(pair_key, set()).add(relation)

        self._register_entity_side(
            x_name, str(row["x_type"]), int(row["x_index"]), str(row["x_id"])
        )
        self._register_entity_side(
            y_name, str(row["y_type"]), int(row["y_index"]), str(row["y_id"])
        )

        try:
            self._next_index = max(
                self._next_index,
                int(row["x_index"]) + 1,
                int(row["y_index"]) + 1,
            )
        except Exception:
            pass

        if update_df:
            self.existing_df = pd.concat(
                [self.existing_df, pd.DataFrame([dict(row)])],
                ignore_index=True,
            )

    def check(self, new_row: dict) -> tuple[str, str | None]:
        """
        检查新三元组是否与已有数据冲突。

        返回:
            ("ok",        None) → 无冲突，可直接插入
            ("duplicate", None) → 完全重复，跳过
            ("conflict",  msg)  → 关系冲突，保留但标注
        """
        x_key = self._normalize_name(new_row["x_name"])
        y_key = self._normalize_name(new_row["y_name"])
        new_rel = str(new_row["relation"])

        existing_rels = self._pair_index.get((x_key, y_key), set())

        if not existing_rels:
            return "ok", None

        if new_rel in existing_rels:
            return "duplicate", None

        for conflict_pair in _CONFLICTING_RELATIONS:
            if new_rel not in conflict_pair:
                continue

            for ex_rel in existing_rels:
                if ex_rel in conflict_pair and ex_rel != new_rel:
                    msg = (
                        f"Conflict: ({new_row['x_name']}, {new_row['y_name']}) "
                        f"has '{ex_rel}' but new triple has '{new_rel}'"
                    )
                    logger.warning(msg)
                    return "conflict", msg

        return "ok", None

    def resolve_new_index(self, existing_df: pd.DataFrame) -> int:
        """获取下一个可用的 x_index / y_index（全局最大值 + 1）。"""
        if existing_df.empty:
            return 0

        max_x = pd.to_numeric(existing_df["x_index"], errors="coerce").fillna(-1).max()
        max_y = pd.to_numeric(existing_df["y_index"], errors="coerce").fillna(-1).max()
        return int(max(max_x, max_y)) + 1

    def get_or_create_entity_id(
        self,
        name: str,
        etype: str,
        existing_df: pd.DataFrame | None = None,
        source: str = "CUSTOM",
    ) -> tuple[int, str]:
        """
        查找已有实体的 (index, id)，若不存在则分配新 index。
        返回 (index, entity_id)。

        说明：
        - existing_df 参数保留仅为兼容旧调用方式，实际以运行时缓存为准。
        - source 当前不参与 ID 生成，仅保留接口兼容性。
        """
        key = (self._normalize_name(name), str(etype))
        if key in self._entity_cache:
            return self._entity_cache[key]

        new_idx = self._next_index
        self._next_index += 1

        new_id = self._slugify_entity_id(name)
        self._entity_cache[key] = (new_idx, new_id)
        return new_idx, new_id