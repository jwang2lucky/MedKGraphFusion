from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
except Exception as e:
    raise ImportError(
        "This script requires scikit-survival. Install it first, e.g. "
        "`pip install scikit-survival`."
    ) from e


DEFAULT_CANCER_TYPES = [
    "Non-Small Cell Lung Cancer",
    "Colorectal Cancer",
    "Breast Cancer",
    "Prostate Cancer",
    "Pancreatic Cancer",
]


def norm_text(x: object) -> str:
    text = str(x or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compact_text(x: object) -> str:
    text = norm_text(x)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def safe_slug(x: str) -> str:
    x = norm_text(x)
    x = re.sub(r"[^a-z0-9]+", "_", x).strip("_")
    return x or "unknown"


def parse_json_list(cell: object) -> list[str]:
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return []


def safe_bool_int(x: object) -> int:
    text = str(x).strip().lower()
    return 1 if text in {"1", "1.0", "true", "yes"} else 0


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_surv_y(df: pd.DataFrame, time_col: str = "os_time", event_col: str = "os_event") -> np.ndarray:
    return np.array(
        [(bool(e), float(t)) for e, t in zip(df[event_col], df[time_col])],
        dtype=[("event", "?"), ("time", "<f8")],
    )


def choose_eval_times(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    event_times = y_train["time"][y_train["event"]]
    if len(event_times) == 0:
        return np.array([float(np.median(y_train["time"]))], dtype=float)

    raw = np.quantile(event_times, [0.25, 0.5, 0.75]).astype(float)
    lower = max(float(np.min(y_test["time"])) + 1e-6, float(np.min(y_train["time"])) + 1e-6)
    upper = min(float(np.max(y_test["time"])) - 1e-6, float(np.max(y_train["time"])) - 1e-6)
    times = np.array([t for t in raw if lower < t < upper], dtype=float)

    if len(times) == 0:
        med = float(np.median(event_times))
        times = np.array([med], dtype=float)
    return np.unique(times)


def infer_n_splits(y_event: pd.Series, requested: int) -> int:
    class_counts = y_event.value_counts(dropna=False).to_dict()
    min_class = min(class_counts.values()) if class_counts else 0
    if min_class < 2:
        raise ValueError("Not enough samples in at least one event class for cross-validation.")
    return max(2, min(int(requested), int(min_class)))


def assert_unique_columns(df: pd.DataFrame, df_name: str = "DataFrame") -> None:
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"{df_name} has duplicated columns: {dupes}")


def get_stage_group_from_stage4(stage_is_4: object) -> str:
    try:
        return "Stage IV" if int(stage_is_4) == 1 else "Stage I-III"
    except Exception:
        return "Unknown"


@dataclass
class FeatureSets:
    clinical_num: list[str]
    clinical_cat: list[str]
    treatment_num: list[str]
    molecular_num: list[str]
    molecular_cat: list[str]
    kg_num: list[str]


class KGFeatureBuilder:
    def __init__(self, kg_csv: str):
        self.df = pd.read_csv(kg_csv, dtype=str).fillna("")
        self.g = nx.Graph()
        self.name_lookup: dict[tuple[str, str], tuple[str, str]] = {}
        self.compact_lookup: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self.names_by_type: dict[str, list[str]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        for _, row in self.df.iterrows():
            x = (norm_text(row["x_type"]), norm_text(row["x_name"]))
            y = (norm_text(row["y_type"]), norm_text(row["y_name"]))
            self.g.add_edge(
                x,
                y,
                relation=norm_text(row.get("relation", "")),
                display_relation=norm_text(row.get("display_relation", "")),
            )

        names_by_type: dict[str, set[str]] = {}
        compact_lookup: dict[tuple[str, str], list[tuple[str, str]]] = {}
        name_lookup: dict[tuple[str, str], tuple[str, str]] = {}

        for node_type, node_name in self.g.nodes():
            name_lookup[(node_type, node_name)] = (node_type, node_name)
            compact_lookup.setdefault((node_type, compact_text(node_name)), []).append((node_type, node_name))
            names_by_type.setdefault(node_type, set()).add(node_name)

        self.names_by_type = {k: sorted(v) for k, v in names_by_type.items()}
        self.compact_lookup = compact_lookup
        self.name_lookup = name_lookup

    @lru_cache(maxsize=100000)
    def resolve_nodes(self, query: str, allowed_types: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        q = norm_text(query)
        qc = compact_text(query)
        if not q:
            return tuple()

        hits: list[tuple[str, str]] = []

        for node_type in allowed_types:
            exact = self.name_lookup.get((node_type, q))
            if exact is not None:
                hits.append(exact)

            compact_hits = self.compact_lookup.get((node_type, qc), [])
            hits.extend(compact_hits)

            if hits:
                continue

            for cand in self.names_by_type.get(node_type, []):
                if q == cand:
                    hits.append((node_type, cand))
                elif len(q) >= 4 and (q in cand or cand in q):
                    hits.append((node_type, cand))
                if len(hits) >= 5:
                    break

        seen = set()
        uniq = []
        for item in hits:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        return tuple(uniq[:5])

    @lru_cache(maxsize=200000)
    def shortest_distance(self, a: tuple[str, str], b: tuple[str, str]) -> float:
        if a == b:
            return 0.0
        try:
            return float(nx.shortest_path_length(self.g, a, b))
        except nx.NetworkXNoPath:
            return math.inf

    def _pair_distance_stats(
        self,
        left_nodes: Iterable[tuple[str, str]],
        right_nodes: Iterable[tuple[str, str]],
    ) -> tuple[int, float, float]:
        dists = []
        left_nodes = list(left_nodes)
        right_nodes = list(right_nodes)
        if not left_nodes or not right_nodes:
            return 0, 99.0, 99.0

        for a in left_nodes:
            for b in right_nodes:
                d = self.shortest_distance(a, b)
                if math.isfinite(d):
                    dists.append(d)

        if not dists:
            return 0, 99.0, 99.0
        return len(dists), float(np.min(dists)), float(np.mean(dists))

    def _resolve_unique(self, queries: Iterable[str], allowed_types: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        nodes = []
        for q in queries:
            nodes.extend(self.resolve_nodes(q, allowed_types))
        return tuple(dict.fromkeys(nodes))

    def mapping_diagnostics(self, row: pd.Series) -> dict[str, object]:
        cancer_queries = [str(row.get("cancer_type", "")), str(row.get("cancer_type_detailed", ""))]
        site_queries = [str(row.get("primary_site", ""))]
        treatment_queries = parse_json_list(row.get("treatments", ""))
        gene_queries = parse_json_list(row.get("altered_genes", ""))

        disease_nodes = self._resolve_unique(cancer_queries, ("disease",))
        anatomy_nodes = self._resolve_unique(site_queries, ("anatomy",))

        mapped_treatments = [q for q in treatment_queries if len(self.resolve_nodes(q, ("drug",))) > 0]
        mapped_genes = [q for q in gene_queries if len(self.resolve_nodes(q, ("gene/protein",))) > 0]

        return {
            "kg_has_disease_mapping": int(len(disease_nodes) > 0),
            "kg_has_anatomy_mapping": int(len(anatomy_nodes) > 0),
            "kg_total_treatments": int(len(treatment_queries)),
            "kg_mapped_treatments": int(len(mapped_treatments)),
            "kg_treatment_mapping_rate": float(len(mapped_treatments) / len(treatment_queries)) if treatment_queries else np.nan,
            "kg_total_genes": int(len(gene_queries)),
            "kg_mapped_genes": int(len(mapped_genes)),
            "kg_gene_mapping_rate": float(len(mapped_genes) / len(gene_queries)) if gene_queries else np.nan,
            "diag_num_disease_nodes": float(len(disease_nodes)),
            "diag_num_anatomy_nodes": float(len(anatomy_nodes)),
            "diag_num_drug_nodes": float(len(self._resolve_unique(treatment_queries, ("drug",)))),
            "diag_num_gene_nodes": float(len(self._resolve_unique(gene_queries, ("gene/protein",)))),
        }

    def build_patient_features(self, row: pd.Series) -> dict[str, float]:
        cancer_queries = [
            str(row.get("cancer_type", "")),
            str(row.get("cancer_type_detailed", "")),
        ]
        site_queries = [str(row.get("primary_site", ""))]
        treatment_queries = parse_json_list(row.get("treatments", ""))
        gene_queries = parse_json_list(row.get("altered_genes", ""))

        disease_nodes = self._resolve_unique(cancer_queries, ("disease",))
        anatomy_nodes = self._resolve_unique(site_queries, ("anatomy",))
        drug_nodes = self._resolve_unique(treatment_queries, ("drug",))
        gene_nodes = self._resolve_unique(gene_queries, ("gene/protein",))

        c_t_count, c_t_min, c_t_mean = self._pair_distance_stats(disease_nodes, drug_nodes)
        c_g_count, c_g_min, c_g_mean = self._pair_distance_stats(disease_nodes, gene_nodes)
        t_g_count, t_g_min, t_g_mean = self._pair_distance_stats(drug_nodes, gene_nodes)
        a_g_count, a_g_min, a_g_mean = self._pair_distance_stats(anatomy_nodes, gene_nodes)

        any_disease_or_anatomy = list(disease_nodes) + list(anatomy_nodes)
        dga_count, dga_min, dga_mean = self._pair_distance_stats(any_disease_or_anatomy, gene_nodes)

        return {
            "kg_num_disease_nodes": float(len(disease_nodes)),
            "kg_num_anatomy_nodes": float(len(anatomy_nodes)),
            "kg_num_drug_nodes": float(len(drug_nodes)),
            "kg_num_gene_nodes": float(len(gene_nodes)),
            "kg_cancer_treatment_pairs": float(c_t_count),
            "kg_cancer_treatment_min_dist": float(c_t_min),
            "kg_cancer_treatment_mean_dist": float(c_t_mean),
            "kg_cancer_gene_pairs": float(c_g_count),
            "kg_cancer_gene_min_dist": float(c_g_min),
            "kg_cancer_gene_mean_dist": float(c_g_mean),
            "kg_treatment_gene_pairs": float(t_g_count),
            "kg_treatment_gene_min_dist": float(t_g_min),
            "kg_treatment_gene_mean_dist": float(t_g_mean),
            "kg_site_gene_pairs": float(a_g_count),
            "kg_site_gene_min_dist": float(a_g_min),
            "kg_site_gene_mean_dist": float(a_g_mean),
            "kg_disease_or_site_gene_pairs": float(dga_count),
            "kg_disease_or_site_gene_min_dist": float(dga_min),
            "kg_disease_or_site_gene_mean_dist": float(dga_mean),
        }

    def build_dataframe(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        rows = [self.build_patient_features(row) for _, row in cohort_df.iterrows()]
        out = pd.DataFrame(rows).fillna(99.0)
        assert_unique_columns(out, "kg_feat_df")
        return out

    def build_mapping_diagnostics_dataframe(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        rows = [self.mapping_diagnostics(row) for _, row in cohort_df.iterrows()]
        out = pd.DataFrame(rows)
        assert_unique_columns(out, "kg_diag_df")
        return out


class SurvivalFeatureEncoder:
    def __init__(
        self,
        feature_sets: FeatureSets,
        top_k_treatments: int = 120,
        top_k_genes: int = 300,
    ):
        self.feature_sets = feature_sets
        self.top_k_treatments = top_k_treatments
        self.top_k_genes = top_k_genes
        self.ohe = make_one_hot_encoder()
        self.treatment_mlb = MultiLabelBinarizer()
        self.gene_mlb = MultiLabelBinarizer()
        self.treatment_vocab: list[str] = []
        self.gene_vocab: list[str] = []

    def _top_items(self, series: pd.Series, k: int) -> list[str]:
        counter = Counter()
        for items in series:
            counter.update(items)
        return [name for name, _ in counter.most_common(k)]

    def fit(self, df: pd.DataFrame) -> None:
        all_cat_cols = self.feature_sets.clinical_cat + self.feature_sets.molecular_cat
        self.ohe.fit(df[all_cat_cols].fillna("Unknown"))

        self.treatment_vocab = self._top_items(df["treatments_list"], self.top_k_treatments)
        self.gene_vocab = self._top_items(df["altered_genes_list"], self.top_k_genes)

        self.treatment_mlb.fit([self.treatment_vocab])
        self.gene_mlb.fit([self.gene_vocab])

    def _encode_categories(self, df: pd.DataFrame, include_molecular: bool) -> tuple[np.ndarray, list[str]]:
        all_cat_cols = self.feature_sets.clinical_cat + self.feature_sets.molecular_cat
        arr = self.ohe.transform(df[all_cat_cols].fillna("Unknown"))
        all_names = list(self.ohe.get_feature_names_out(all_cat_cols))

        keep_cols = list(self.feature_sets.clinical_cat)
        if include_molecular:
            keep_cols += self.feature_sets.molecular_cat

        keep_prefixes = tuple(col + "_" for col in keep_cols)
        keep_idx = [i for i, name in enumerate(all_names) if name.startswith(keep_prefixes)]
        return arr[:, keep_idx], [all_names[i] for i in keep_idx]

    def _encode_multilabel(
        self,
        series: pd.Series,
        vocab: list[str],
        mlb: MultiLabelBinarizer,
        prefix: str,
    ) -> tuple[np.ndarray, list[str]]:
        filtered = [[x for x in items if x in vocab] for items in series]
        arr = mlb.transform(filtered)
        names = [f"{prefix}{x}" for x in mlb.classes_]
        return arr, names

    def transform(
        self,
        df: pd.DataFrame,
        include_treatment: bool,
        include_molecular: bool,
        include_kg: bool,
    ) -> tuple[np.ndarray, list[str]]:
        parts = []
        names = []

        clin_num = df[self.feature_sets.clinical_num].astype(float).fillna(0.0).to_numpy()
        parts.append(clin_num)
        names.extend(self.feature_sets.clinical_num)

        cat_arr, cat_names = self._encode_categories(df, include_molecular=include_molecular)
        parts.append(cat_arr)
        names.extend(cat_names)

        if include_treatment:
            tx_num = df[self.feature_sets.treatment_num].astype(float).fillna(0.0).to_numpy()
            parts.append(tx_num)
            names.extend(self.feature_sets.treatment_num)

            tr_arr, tr_names = self._encode_multilabel(
                df["treatments_list"],
                self.treatment_vocab,
                self.treatment_mlb,
                "tx__",
            )
            parts.append(tr_arr)
            names.extend(tr_names)

        if include_molecular:
            mol_num = df[self.feature_sets.molecular_num].astype(float).fillna(0.0).to_numpy()
            parts.append(mol_num)
            names.extend(self.feature_sets.molecular_num)

            ge_arr, ge_names = self._encode_multilabel(
                df["altered_genes_list"],
                self.gene_vocab,
                self.gene_mlb,
                "gene__",
            )
            parts.append(ge_arr)
            names.extend(ge_names)

        if include_kg:
            kg_num = df[self.feature_sets.kg_num].astype(float).fillna(99.0).to_numpy()
            parts.append(kg_num)
            names.extend(self.feature_sets.kg_num)

        x = np.concatenate(parts, axis=1).astype(np.float32)
        return x, names


def load_cohort(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    if "patient_id" not in df.columns:
        df["patient_id"] = [f"patient_{i}" for i in range(len(df))]
    df["patient_id"] = df["patient_id"].astype(str)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["stage_is_4"] = df["stage_is_4"].apply(safe_bool_int)
    df["prior_treatment_binary"] = df["prior_treatment_binary"].apply(safe_bool_int)
    df["sample_is_metastasis"] = df["sample_is_metastasis"].apply(safe_bool_int)
    df["missing_tmb"] = df["missing_tmb"].apply(safe_bool_int)
    df["missing_tumor_purity"] = df["missing_tumor_purity"].apply(safe_bool_int)
    df["os_time"] = pd.to_numeric(df["os_time"], errors="coerce")
    df["os_event"] = pd.to_numeric(df["os_event"], errors="coerce").fillna(0).astype(int)
    df["num_treatments"] = pd.to_numeric(df["num_treatments"], errors="coerce").fillna(0)
    df["num_altered_genes"] = pd.to_numeric(df["num_altered_genes"], errors="coerce").fillna(0)
    df["tmb"] = pd.to_numeric(df["tmb"], errors="coerce")
    df["tumor_purity"] = pd.to_numeric(df["tumor_purity"], errors="coerce")

    for col in [
        "sex",
        "race",
        "ethnicity",
        "stage",
        "sample_type",
        "cancer_type",
        "cancer_type_detailed",
        "primary_site",
        "smoking_history",
        "msi_type",
        "gene_panel",
    ]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": "Unknown", "nan": "Unknown", "NA": "Unknown"})

    df["treatments_list"] = df["treatments"].apply(parse_json_list)
    df["altered_genes_list"] = df["altered_genes"].apply(parse_json_list)
    df["stage_group"] = df["stage_is_4"].apply(get_stage_group_from_stage4)

    assert_unique_columns(df, "cohort_df")
    return df


def prepare_feature_sets(kg_cols: list[str]) -> FeatureSets:
    return FeatureSets(
        clinical_num=[
            "age",
            "stage_is_4",
            "prior_treatment_binary",
            "sample_is_metastasis",
        ],
        clinical_cat=[
            "sex",
            "race",
            "ethnicity",
            "stage",
            "sample_type",
            "cancer_type",
            "cancer_type_detailed",
            "primary_site",
            "smoking_history",
        ],
        treatment_num=[
            "num_treatments",
        ],
        molecular_num=[
            "tmb",
            "tumor_purity",
            "missing_tmb",
            "missing_tumor_purity",
            "num_altered_genes",
        ],
        molecular_cat=[
            "msi_type",
            "gene_panel",
        ],
        kg_num=kg_cols,
    )


def fill_train_medians(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for c in numeric_cols:
        train_col = train_df[c]
        test_col = test_df[c]

        if isinstance(train_col, pd.DataFrame) or isinstance(test_col, pd.DataFrame):
            raise ValueError(f"Column '{c}' is duplicated in dataset; expected a single numeric Series.")

        med = pd.to_numeric(train_col, errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        train_df[c] = pd.to_numeric(train_col, errors="coerce").fillna(med)
        test_df[c] = pd.to_numeric(test_col, errors="coerce").fillna(med)

    return train_df, test_df


def summarize_kg_diagnostics(diag_df: pd.DataFrame) -> dict[str, float]:
    out = {
        "n_patients": int(len(diag_df)),
        "disease_mapping_patient_rate": float(diag_df["kg_has_disease_mapping"].mean()) if len(diag_df) else np.nan,
        "anatomy_mapping_patient_rate": float(diag_df["kg_has_anatomy_mapping"].mean()) if len(diag_df) else np.nan,
        "treatment_mapping_rate_nonempty": float(diag_df["kg_treatment_mapping_rate"].dropna().mean()) if diag_df["kg_treatment_mapping_rate"].notna().any() else np.nan,
        "gene_mapping_rate_nonempty": float(diag_df["kg_gene_mapping_rate"].dropna().mean()) if diag_df["kg_gene_mapping_rate"].notna().any() else np.nan,
        "patients_with_any_treatment": int((diag_df["kg_total_treatments"] > 0).sum()),
        "patients_with_any_gene": int((diag_df["kg_total_genes"] > 0).sum()),
        "patients_with_unmapped_treatments": int((diag_df["kg_total_treatments"] > diag_df["kg_mapped_treatments"]).sum()),
        "patients_with_unmapped_genes": int((diag_df["kg_total_genes"] > diag_df["kg_mapped_genes"]).sum()),
    }
    return out


def summarize_feature_distribution(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            raise ValueError(f"Feature column '{c}' is duplicated in dataframe.")
        s = pd.to_numeric(col, errors="coerce")
        rows.append(
            {
                "feature": c,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "n_unique": int(s.nunique(dropna=True)),
                "pct_sentinel_99": float((s == 99).mean() * 100.0),
                "pct_zero": float((s == 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def can_run_subset(
    df: pd.DataFrame,
    min_rows: int = 20,
    min_class_count: int = 2,
) -> tuple[bool, str]:
    if len(df) < min_rows:
        return False, f"n={len(df)} < min_rows={min_rows}"

    if "os_event" not in df.columns:
        return False, "missing os_event"

    class_counts = df["os_event"].value_counts(dropna=False).to_dict()
    if len(class_counts) < 2:
        return False, f"os_event has only one class: {class_counts}"

    if min(class_counts.values()) < min_class_count:
        return False, f"min event-class count {min(class_counts.values())} < {min_class_count}"

    return True, "ok"


def build_dataset_specs(
    df: pd.DataFrame,
    cancer_types: list[str],
    include_stage_subsets: bool = True,
    stage_subset_cancers: list[str] | None = None,
    stage_groups: list[str] | None = None,
    min_rows_per_subset: int = 20,
    min_class_count: int = 2,
) -> list[tuple[str, pd.DataFrame]]:
    if stage_groups is None:
        stage_groups = ["Stage I-III", "Stage IV"]

    if stage_subset_cancers is None:
        stage_subset_cancers = cancer_types

    stage_subset_cancers = set(stage_subset_cancers)

    datasets: list[tuple[str, pd.DataFrame]] = [("all_cancers", df.copy())]

    for cancer_type in cancer_types:
        sub = df[df["cancer_type"] == cancer_type].copy()
        if len(sub) == 0:
            print(f"[WARN] No rows found for cancer type: {cancer_type}")
            continue

        ok_main, reason_main = can_run_subset(
            sub,
            min_rows=min_rows_per_subset,
            min_class_count=min_class_count,
        )
        if ok_main:
            datasets.append((cancer_type, sub))
        else:
            print(f"[WARN] Skip main dataset [{cancer_type}] because {reason_main}")

        if include_stage_subsets and cancer_type in stage_subset_cancers:
            for sg in stage_groups:
                sg_sub = sub[sub["stage_group"] == sg].copy()
                if len(sg_sub) == 0:
                    print(f"[WARN] No rows found for {cancer_type} / {sg}")
                    continue

                ok_sg, reason_sg = can_run_subset(
                    sg_sub,
                    min_rows=min_rows_per_subset,
                    min_class_count=min_class_count,
                )
                if ok_sg:
                    datasets.append((f"{cancer_type} - {sg}", sg_sub))
                else:
                    print(f"[WARN] Skip dataset [{cancer_type} - {sg}] because {reason_sg}")

    return datasets

def evaluate_one_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    out_dir: Path,
    n_splits: int,
    random_state: int,
    feature_sets: FeatureSets,
    top_k_treatments: int,
    top_k_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = out_dir / safe_slug(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    actual_splits = infer_n_splits(df["os_event"], n_splits)
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    models = {
        "clinical_only": dict(include_treatment=False, include_molecular=False, include_kg=False),
        "clinical_kg": dict(include_treatment=False, include_molecular=False, include_kg=True),
        "clinical_molecular": dict(include_treatment=False, include_molecular=True, include_kg=False),
        "clinical_molecular_kg": dict(include_treatment=False, include_molecular=True, include_kg=True),
        "clinical_treatment_molecular": dict(include_treatment=True, include_molecular=True, include_kg=False),
        "clinical_treatment_molecular_kg": dict(include_treatment=True, include_molecular=True, include_kg=True),
    }

    all_numeric_cols = (
        feature_sets.clinical_num
        + feature_sets.treatment_num
        + feature_sets.molecular_num
        + feature_sets.kg_num
    )

    fold_rows = []
    pred_rows = []

    export_cols = [
        "patient_id",
        "cancer_type",
        "cancer_type_detailed",
        "stage",
        "stage_is_4",
        "stage_group",
        "os_time",
        "os_event",
        "num_treatments",
        "num_altered_genes",
    ]
    export_cols = [c for c in export_cols if c in df.columns]

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(df, df["os_event"]), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_df, test_df = fill_train_medians(train_df, test_df, all_numeric_cols)

        encoder = SurvivalFeatureEncoder(
            feature_sets=feature_sets,
            top_k_treatments=top_k_treatments,
            top_k_genes=top_k_genes,
        )
        encoder.fit(train_df)

        y_train = make_surv_y(train_df)
        y_test = make_surv_y(test_df)
        eval_times = choose_eval_times(y_train, y_test)

        for model_name, flags in models.items():
            x_train, feat_names = encoder.transform(train_df, **flags)
            x_test, _ = encoder.transform(test_df, **flags)

            model = RandomSurvivalForest(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            )
            model.fit(x_train, y_train)

            risk = model.predict(x_test)
            cindex = float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
            auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, eval_times)

            row = {
                "dataset": dataset_name,
                "fold": fold_id,
                "model": model_name,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_features": int(x_train.shape[1]),
                "c_index": cindex,
                "td_auc_mean": float(mean_auc),
            }
            for i, t in enumerate(eval_times, start=1):
                row[f"td_auc_t{i}"] = float(auc_values[i - 1])
                row[f"eval_time_t{i}"] = float(t)

            fold_rows.append(row)

            pred_df_one = test_df[export_cols].copy()
            pred_df_one["dataset"] = dataset_name
            pred_df_one["fold"] = fold_id
            pred_df_one["model"] = model_name
            pred_df_one["risk_score"] = risk.astype(float)
            pred_df_one["n_features"] = int(x_train.shape[1])
            pred_rows.append(pred_df_one)

            print(
                f"[{dataset_name}] [Fold {fold_id}] {model_name}: "
                f"C-index={cindex:.4f}, mean td-AUC={float(mean_auc):.4f}, "
                f"n_features={x_train.shape[1]}"
            )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(dataset_dir / "os_ablation_fold_metrics.csv", index=False)

    summary_df = (
        folds_df.groupby("model")
        .agg(
            c_index_mean=("c_index", "mean"),
            c_index_std=("c_index", "std"),
            td_auc_mean=("td_auc_mean", "mean"),
            td_auc_std=("td_auc_mean", "std"),
            mean_n_features=("n_features", "mean"),
        )
        .reset_index()
        .sort_values("c_index_mean", ascending=False)
    )
    summary_df.insert(0, "dataset", dataset_name)
    summary_df.to_csv(dataset_dir / "os_ablation_summary.csv", index=False)

    oof_pred_df = pd.concat(pred_rows, ignore_index=True)
    oof_pred_df.to_csv(dataset_dir / "os_ablation_oof_predictions.csv", index=False)

    meta = {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "event_rate": float(df["os_event"].mean()),
        "median_os_time": float(df["os_time"].median()),
        "n_splits_requested": int(n_splits),
        "n_splits_used": int(actual_splits),
        "top_k_treatments": int(top_k_treatments),
        "top_k_genes": int(top_k_genes),
        "cancer_type_counts": df["cancer_type"].value_counts().to_dict(),
        "stage_group_counts": df["stage_group"].value_counts().to_dict() if "stage_group" in df.columns else {},
    }
    with open(dataset_dir / "os_ablation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return folds_df, summary_df, oof_pred_df

def export_stage_specific_oof_files(
    all_preds_df: pd.DataFrame,
    out_path: Path,
    cancer_types: list[str],
    stage_groups: list[str] | None = None,
) -> None:
    if stage_groups is None:
        stage_groups = ["Stage I-III", "Stage IV"]

    if len(all_preds_df) == 0:
        return

    for cancer_type in cancer_types:
        cancer_slug = safe_slug(cancer_type)

        cancer_all = all_preds_df[all_preds_df["cancer_type"] == cancer_type].copy()
        if len(cancer_all) == 0:
            print(f"[WARN] No OOF rows for cancer type: {cancer_type}")
            continue

        cancer_all.to_csv(
            out_path / f"{cancer_slug}_oof_predictions_all_models.csv",
            index=False,
        )

        cancer_main = all_preds_df[all_preds_df["dataset"] == cancer_type].copy()
        if len(cancer_main) > 0:
            cancer_main.to_csv(
                out_path / f"{cancer_slug}_main_dataset_oof_predictions.csv",
                index=False,
            )

        for sg in stage_groups:
            sg_slug = safe_slug(sg)

            cancer_stage_all = cancer_all[cancer_all["stage_group"] == sg].copy()
            if len(cancer_stage_all) > 0:
                cancer_stage_all.to_csv(
                    out_path / f"{cancer_slug}_{sg_slug}_oof_predictions_all_models.csv",
                    index=False,
                )

            cancer_stage_main = all_preds_df[
                all_preds_df["dataset"] == f"{cancer_type} - {sg}"
            ].copy()
            if len(cancer_stage_main) > 0:
                cancer_stage_main.to_csv(
                    out_path / f"{cancer_slug}_{sg_slug}_main_dataset_oof_predictions.csv",
                    index=False,
                )

    # 向后兼容：保留你原先 pancreatic 的老文件名
    pancreatic_slug = safe_slug("Pancreatic Cancer")
    legacy_map = {
        f"{pancreatic_slug}_main_dataset_oof_predictions.csv": "pancreatic_main_dataset_oof_predictions.csv",
        f"{pancreatic_slug}_stage_i_iii_main_dataset_oof_predictions.csv": "pancreatic_stage_i_iii_main_dataset_oof_predictions.csv",
        f"{pancreatic_slug}_stage_iv_main_dataset_oof_predictions.csv": "pancreatic_stage_iv_main_dataset_oof_predictions.csv",
        f"{pancreatic_slug}_oof_predictions_all_models.csv": "pancreatic_oof_predictions_all_models.csv",
        f"{pancreatic_slug}_stage_i_iii_oof_predictions_all_models.csv": "pancreatic_stage_i_iii_oof_predictions_all_models.csv",
        f"{pancreatic_slug}_stage_iv_oof_predictions_all_models.csv": "pancreatic_stage_iv_oof_predictions_all_models.csv",
    }

    for new_name, legacy_name in legacy_map.items():
        src = out_path / new_name
        dst = out_path / legacy_name
        if src.exists():
            src_df = pd.read_csv(src)
            src_df.to_csv(dst, index=False)

def run_ablation(
    cohort_csv: str,
    kg_csv: str,
    out_dir: str,
    cancer_types: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    top_k_treatments: int = 120,
    top_k_genes: int = 300,
    include_stage_subsets: bool = True,
    stage_subset_cancers: list[str] | None = None,
    min_rows_per_stage_subset: int = 20,
    min_event_class_count: int = 2,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_cohort(cohort_csv)

    print("Building KG prior features...")
    kg_builder = KGFeatureBuilder(kg_csv)
    kg_diag_df = kg_builder.build_mapping_diagnostics_dataframe(df)
    kg_feat_df = kg_builder.build_dataframe(df)

    overlap_cols = sorted(set(kg_diag_df.columns) & set(kg_feat_df.columns))
    if overlap_cols:
        raise ValueError(
            "kg_diag_df and kg_feat_df still have overlapping columns after renaming: "
            f"{overlap_cols}"
        )

    df = pd.concat(
        [df.reset_index(drop=True), kg_diag_df.reset_index(drop=True), kg_feat_df.reset_index(drop=True)],
        axis=1,
    )
    assert_unique_columns(df, "merged_cohort_with_kg")

    kg_diag_cols = list(kg_diag_df.columns)
    kg_feature_cols = list(kg_feat_df.columns)

    df.to_csv(out_path / "os_cohort_with_kg_features.csv", index=False)

    kg_diag_summary = summarize_kg_diagnostics(kg_diag_df)
    with open(out_path / "kg_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(kg_diag_summary, f, ensure_ascii=False, indent=2)

    kg_diag_export_cols = [
        "patient_id",
        "cancer_type",
        "stage_group",
        "os_time",
        "os_event",
        "num_treatments",
        "num_altered_genes",
    ] + kg_diag_cols + kg_feature_cols
    kg_diag_export_cols = [c for c in kg_diag_export_cols if c in df.columns]
    df[kg_diag_export_cols].to_csv(out_path / "kg_diagnostics_patient_level.csv", index=False)

    feat_dist_df = summarize_feature_distribution(df, kg_feature_cols)
    feat_dist_df.to_csv(out_path / "kg_feature_distribution.csv", index=False)

    feature_sets = prepare_feature_sets(kg_feature_cols)
    datasets = build_dataset_specs(
        df=df,
        cancer_types=cancer_types,
        include_stage_subsets=include_stage_subsets,
        stage_subset_cancers=stage_subset_cancers,
        min_rows_per_subset=min_rows_per_stage_subset,
        min_class_count=min_event_class_count,
    )

    all_fold_frames = []
    all_summary_frames = []
    all_pred_frames = []

    for dataset_name, subset_df in datasets:
        print(f"\n=== Running dataset: {dataset_name} (n={len(subset_df)}) ===")
        try:
            folds_df, summary_df, pred_df = evaluate_one_dataset(
                dataset_name=dataset_name,
                df=subset_df.reset_index(drop=True),
                out_dir=out_path,
                n_splits=n_splits,
                random_state=random_state,
                feature_sets=feature_sets,
                top_k_treatments=top_k_treatments,
                top_k_genes=top_k_genes,
            )
            all_fold_frames.append(folds_df)
            all_summary_frames.append(summary_df)
            all_pred_frames.append(pred_df)
        except Exception as e:
            print(f"[ERROR] Dataset {dataset_name} failed: {e}")

    if all_pred_frames:
        all_preds_df = pd.concat(all_pred_frames, ignore_index=True)
        all_preds_df.to_csv(out_path / "os_ablation_oof_predictions_all_datasets.csv"    , index=False)

        export_stage_specific_oof_files(
            all_preds_df=all_preds_df,
            out_path=out_path,
            cancer_types=cancer_types,
            stage_groups=["Stage I-III", "Stage IV"],
        )

    meta = {
        "cohort_rows": int(len(df)),
        "event_rate": float(df["os_event"].mean()),
        "median_os_time": float(df["os_time"].median()),
        "n_splits_requested": int(n_splits),
        "top_k_treatments": int(top_k_treatments),
        "top_k_genes": int(top_k_genes),
        "cancer_types_requested": cancer_types,
        "include_stage_subsets": bool(include_stage_subsets),
        "stage_subset_cancers": stage_subset_cancers if stage_subset_cancers is not None else cancer_types,
        "min_rows_per_stage_subset": int(min_rows_per_stage_subset),
        "min_event_class_count": int(min_event_class_count),
        "kg_feature_columns": kg_feature_cols,
        "kg_diagnostic_columns": kg_diag_cols,
        "datasets_run": [name for name, _ in datasets],
    }
    with open(out_path / "os_ablation_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all outputs to: {out_path.resolve()}")
    print("Key OOF files:")
    print(f"- {out_path / 'os_ablation_oof_predictions_all_datasets.csv'}")
    print(f"- {out_path / 'pancreatic_main_dataset_oof_predictions.csv'}")
    print(f"- {out_path / 'pancreatic_stage_i_iii_main_dataset_oof_predictions.csv'}")
    print(f"- {out_path / 'pancreatic_stage_iv_main_dataset_oof_predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_csv", required=True, help="Path to os_cohort_strict.csv")
    parser.add_argument("--kg_csv", required=True, help="Path to kg_merged.csv")
    parser.add_argument("--out_dir", default="os_ablation_results")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--top_k_treatments", type=int, default=120)
    parser.add_argument("--top_k_genes", type=int, default=300)
    parser.add_argument(
        "--cancer_types",
        nargs="*",
        default=DEFAULT_CANCER_TYPES,
        help="Cancer types for single-cancer analyses.",
    )
    parser.add_argument(
        "--no_stage_subsets",
        action="store_true",
        help="Disable automatic Stage I-III / Stage IV subgroup evaluation for the requested cancer types.",
    )
    parser.add_argument(
        "--stage_subset_cancers",
        nargs="*",
        default=None,
        help="Cancer types for which stage-specific subsets should be evaluated. Default: all requested cancer types.",
    )
    parser.add_argument(
        "--min_rows_per_stage_subset",
        type=int,
        default=20,
        help="Minimum number of rows required to run a stage-specific subset.",
    )
    parser.add_argument(
       "--min_event_class_count",
        type=int,
        default=2,
        help="Minimum count in each os_event class required to run a subset.",
    )
    args = parser.parse_args()

    run_ablation(
        cohort_csv=args.cohort_csv,
        kg_csv=args.kg_csv,
        out_dir=args.out_dir,
        cancer_types=args.cancer_types,
        n_splits=args.n_splits,
        random_state=args.random_state,
        top_k_treatments=args.top_k_treatments,
        top_k_genes=args.top_k_genes,
        include_stage_subsets=not args.no_stage_subsets,
        stage_subset_cancers=args.stage_subset_cancers,
        min_rows_per_stage_subset=args.min_rows_per_stage_subset,
        min_event_class_count=args.min_event_class_count,
    )