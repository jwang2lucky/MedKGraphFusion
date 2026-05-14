from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


KG_CACHE_VERSION = "v1"


PATHWAY_LIKE_TYPE_HINTS = {
    "pathway",
    "biological_process",
    "go_term",
    "molecular_function",
    "cellular_component",
    "process",
    "function",
    "signaling",
    "signalling",
    "hallmark",
}


ALLOWED_INTERMEDIATE_TYPES = {
    "biological_process",
    "gene/protein",
    "disease",
    "effect/phenotype",
    "anatomy",
    "molecular_function",
    "drug",
    "cellular_component",
    "exposure",
    "pathway",
    "procedure",
}


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
            return [str(v).strip() for v in obj if str(v).strip()]
    except Exception:
        pass
    return []


def parse_flexible_list(cell: object) -> list[str]:
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, (list, tuple, set)):
                return [str(v).strip() for v in obj if str(v).strip()]
        except Exception:
            pass

    parts = [p.strip() for p in re.split(r"[|;,]", text) if p.strip()]
    if parts:
        return parts

    return [text]


def parse_origin_list(cell: object) -> list[str]:
    return parse_flexible_list(cell)


def safe_bool_int(x: object) -> int:
    text = str(x).strip().lower()
    return 1 if text in {"1", "1.0", "true", "yes"} else 0


def node_to_id(node: tuple[str, str]) -> str:
    return f"{node[0]}|{node[1]}"


def pretty_name(node: tuple[str, str]) -> str:
    node_type, node_name = node
    if node_type == "gene/protein":
        return node_name.upper()
    return node_name.title()


def edge_key(u: tuple[str, str], v: tuple[str, str]) -> frozenset:
    return frozenset((u, v))


def is_pathway_like(node: tuple[str, str]) -> bool:
    node_type, node_name = node
    t = norm_text(node_type)
    n = norm_text(node_name)
    if any(h in t for h in PATHWAY_LIKE_TYPE_HINTS):
        return True
    if "pathway" in n or "signaling" in n or "signalling" in n or "process" in n:
        return True
    return False


def is_pancreas_related_name(name: str) -> bool:
    n = norm_text(name)
    return "pancrea" in n


def is_generic_disease_node(node: tuple[str, str]) -> bool:
    node_type, node_name = node
    if norm_text(node_type) != "disease":
        return False
    n = norm_text(node_name)
    return n in {"cancer", "neoplasm", "tumor", "tumour", "malignancy"}


def is_cross_cancer_disease_node(node: tuple[str, str]) -> bool:
    node_type, node_name = node
    if norm_text(node_type) != "disease":
        return False
    n = norm_text(node_name)
    if is_pancreas_related_name(n):
        return False
    if "cancer" in n or "adenocarcinoma" in n or "carcinoma" in n or "tumor" in n or "tumour" in n:
        return True
    return False


def is_allowed_intermediate_type(node: tuple[str, str]) -> bool:
    node_type, _ = node
    return norm_text(node_type) in ALLOWED_INTERMEDIATE_TYPES


def load_cohort(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    if "patient_id" not in df.columns:
        raise ValueError("cohort_csv must contain patient_id")

    df["patient_id"] = df["patient_id"].astype(str)

    if "stage_is_4" in df.columns:
        df["stage_is_4"] = df["stage_is_4"].apply(safe_bool_int)
    else:
        df["stage_is_4"] = 0

    if "os_time" in df.columns:
        df["os_time"] = pd.to_numeric(df["os_time"], errors="coerce")
    else:
        df["os_time"] = np.nan

    if "os_event" in df.columns:
        df["os_event"] = pd.to_numeric(df["os_event"], errors="coerce").fillna(0).astype(int)
    else:
        df["os_event"] = 0

    for col in ["cancer_type", "cancer_type_detailed", "primary_site", "stage"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    if "treatments" not in df.columns or "altered_genes" not in df.columns:
        raise ValueError("cohort_csv must contain treatments and altered_genes columns")

    df["treatments_list"] = df["treatments"].apply(parse_json_list)
    df["altered_genes_list"] = df["altered_genes"].apply(parse_json_list)
    return df


def load_oof(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["patient_id"] = df["patient_id"].astype(str)

    if "risk_score" not in df.columns:
        raise ValueError("oof_csv must contain risk_score")
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")

    if "stage_is_4" in df.columns:
        df["stage_is_4"] = pd.to_numeric(df["stage_is_4"], errors="coerce").fillna(0).astype(int)
    else:
        df["stage_is_4"] = 0

    if "os_time" in df.columns:
        df["os_time"] = pd.to_numeric(df["os_time"], errors="coerce")
    else:
        df["os_time"] = np.nan

    if "os_event" in df.columns:
        df["os_event"] = pd.to_numeric(df["os_event"], errors="coerce").fillna(0).astype(int)
    else:
        df["os_event"] = 0

    if "stage_group" not in df.columns:
        df["stage_group"] = np.where(df["stage_is_4"] == 1, "Stage IV", "Stage I-III")

    if "model" not in df.columns:
        raise ValueError("oof_csv must contain model")
    if "cancer_type" not in df.columns:
        df["cancer_type"] = ""

    df = df.dropna(subset=["patient_id", "risk_score"]).copy()
    return df


@dataclass
class GraphBundle:
    name: str
    g: nx.Graph
    edge_pairs: set[frozenset]
    edge_meta: dict[frozenset, dict[str, tuple[str, ...]]]


class KGResolver:
    def __init__(self, kg_csv: str, name: str):
        self.kg_csv = str(Path(kg_csv).resolve())
        self.name = name
        self.df = pd.read_csv(kg_csv, dtype=str).fillna("")
        self.g = nx.Graph()
        self.edge_pairs: set[frozenset] = set()
        self.edge_meta: dict[frozenset, dict[str, tuple[str, ...]]] = {}
        self.neighbor_index: dict[tuple[str, str], set[tuple[str, str]]] = {}
        self.name_lookup: dict[tuple[str, str], tuple[str, str]] = {}
        self.compact_lookup: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self.names_by_type: dict[str, list[str]] = {}
        self._build_graph()

    @staticmethod
    def _file_signature(path: str) -> str:
        p = Path(path).resolve()
        st = p.stat()
        raw = f"{KG_CACHE_VERSION}|{p}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_cache_or_build(
        cls,
        kg_csv: str,
        name: str,
        cache_dir: str = ".kg_cache",
        force_rebuild: bool = False,
    ) -> "KGResolver":
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        sig = cls._file_signature(kg_csv)
        cache_path = cache_root / f"{safe_slug(name)}__{Path(kg_csv).stem}__{sig}.pkl"

        if cache_path.exists() and not force_rebuild:
            print(f"[CACHE HIT] Loading {name} from {cache_path}")
            with open(cache_path, "rb") as f:
                state = pickle.load(f)

            obj = cls.__new__(cls)
            obj.kg_csv = state["kg_csv"]
            obj.name = state["name"]
            obj.df = state["df"]
            obj.g = state["g"]
            obj.edge_pairs = state["edge_pairs"]
            obj.edge_meta = state["edge_meta"]
            obj.neighbor_index = state["neighbor_index"]
            obj.name_lookup = state["name_lookup"]
            obj.compact_lookup = state["compact_lookup"]
            obj.names_by_type = state["names_by_type"]
            return obj

        print(f"[CACHE MISS] Building {name} from CSV: {kg_csv}")
        obj = cls(kg_csv=kg_csv, name=name)

        state = {
            "kg_csv": obj.kg_csv,
            "name": obj.name,
            "df": obj.df,
            "g": obj.g,
            "edge_pairs": obj.edge_pairs,
            "edge_meta": obj.edge_meta,
            "neighbor_index": obj.neighbor_index,
            "name_lookup": obj.name_lookup,
            "compact_lookup": obj.compact_lookup,
            "names_by_type": obj.names_by_type,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[CACHE SAVE] Saved {name} cache to {cache_path}")
        return obj

    def _build_graph(self) -> None:
        names_by_type: dict[str, set[str]] = {}
        compact_lookup: dict[tuple[str, str], list[tuple[str, str]]] = {}
        name_lookup: dict[tuple[str, str], tuple[str, str]] = {}
        edge_meta_acc: dict[frozenset, dict[str, set[str]]] = {}

        for _, row in self.df.iterrows():
            x = (norm_text(row.get("x_type", "")), norm_text(row.get("x_name", "")))
            y = (norm_text(row.get("y_type", "")), norm_text(row.get("y_name", "")))
            if not x[0] or not x[1] or not y[0] or not y[1]:
                continue

            rel = norm_text(row.get("relation", ""))
            disp = norm_text(row.get("display_relation", ""))
            origins_raw = row.get("_origins", row.get("origins", ""))
            origins = parse_origin_list(origins_raw)

            if not self.g.has_edge(x, y):
                self.g.add_edge(
                    x,
                    y,
                    relation=rel,
                    display_relation=disp,
                    relations=set([rel]) if rel else set(),
                    display_relations=set([disp]) if disp else set(),
                    origins=set(origins),
                )
            else:
                d = self.g.get_edge_data(x, y) or {}
                if rel:
                    d.setdefault("relations", set()).add(rel)
                if disp:
                    d.setdefault("display_relations", set()).add(disp)
                if origins:
                    d.setdefault("origins", set()).update(origins)
                if not d.get("relation") and rel:
                    d["relation"] = rel
                if not d.get("display_relation") and disp:
                    d["display_relation"] = disp

            pair = edge_key(x, y)
            self.edge_pairs.add(pair)

            meta = edge_meta_acc.setdefault(
                pair,
                {"relations": set(), "display_relations": set(), "origins": set()},
            )
            if rel:
                meta["relations"].add(rel)
            if disp:
                meta["display_relations"].add(disp)
            if origins:
                meta["origins"].update(origins)

        for node_type, node_name in self.g.nodes():
            name_lookup[(node_type, node_name)] = (node_type, node_name)
            compact_lookup.setdefault((node_type, compact_text(node_name)), []).append((node_type, node_name))
            names_by_type.setdefault(node_type, set()).add(node_name)

        self.name_lookup = name_lookup
        self.compact_lookup = compact_lookup
        self.names_by_type = {k: sorted(v) for k, v in names_by_type.items()}
        self.edge_meta = {
            k: {
                "relations": tuple(sorted(v["relations"])),
                "display_relations": tuple(sorted(v["display_relations"])),
                "origins": tuple(sorted(v["origins"])),
            }
            for k, v in edge_meta_acc.items()
        }
        self.neighbor_index = {node: set(self.g.neighbors(node)) for node in self.g.nodes()}

    @property
    def bundle(self) -> GraphBundle:
        return GraphBundle(
            name=self.name,
            g=self.g,
            edge_pairs=self.edge_pairs,
            edge_meta=self.edge_meta,
        )

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
        for x in hits:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return tuple(uniq[:5])

    def resolve_unique(self, queries: Iterable[str], allowed_types: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        nodes = []
        for q in queries:
            nodes.extend(self.resolve_nodes(q, allowed_types))
        return tuple(dict.fromkeys(nodes))


def choose_representative_patient(
    oof_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    resolver: KGResolver,
    model_name: str,
    cancer_type: str,
    stage_group: str,
) -> str:
    pred = oof_df.copy()
    pred = pred[pred["model"] == model_name].copy()
    pred = pred[pred["cancer_type"] == cancer_type].copy()
    pred = pred[pred["stage_group"] == stage_group].copy()

    pred = (
        pred.sort_values(["patient_id", "fold"] if "fold" in pred.columns else ["patient_id"])
        .drop_duplicates(subset=["patient_id", "model"], keep="first")
        .reset_index(drop=True)
    )

    if pred.empty:
        raise ValueError("No OOF rows matched the requested model/cancer_type/stage_group")

    pred["risk_q"] = pd.qcut(pred["risk_score"], q=4, labels=["q0", "q1", "q2", "q3"], duplicates="drop")
    high = pred[pred["risk_q"] == "q3"].copy()
    if high.empty:
        high = pred.copy()

    merged = high.merge(
        cohort_df[
            [
                "patient_id",
                "cancer_type",
                "cancer_type_detailed",
                "primary_site",
                "treatments_list",
                "altered_genes_list",
            ]
        ],
        on="patient_id",
        how="left",
    )

    merged["n_tx"] = merged["treatments_list"].apply(len)
    merged["n_gene"] = merged["altered_genes_list"].apply(len)
    merged = merged.sort_values(["risk_score", "n_tx", "n_gene"], ascending=[False, False, False])

    for _, row in merged.iterrows():
        disease_nodes = resolver.resolve_unique(
            [row.get("cancer_type", ""), row.get("cancer_type_detailed", "")],
            ("disease",),
        )
        drug_nodes = resolver.resolve_unique(row.get("treatments_list", []), ("drug",))
        gene_nodes = resolver.resolve_unique(row.get("altered_genes_list", []), ("gene/protein",))

        if len(disease_nodes) > 0 and len(drug_nodes) > 0 and len(gene_nodes) > 0:
            return str(row["patient_id"])

    return str(merged.iloc[0]["patient_id"])


def get_patient_row(cohort_df: pd.DataFrame, patient_id: str) -> pd.Series:
    sub = cohort_df[cohort_df["patient_id"] == patient_id].copy()
    if sub.empty:
        raise ValueError(f"patient_id {patient_id} not found in cohort_csv")
    return sub.iloc[0]


def get_oof_row(oof_df: pd.DataFrame, patient_id: str, model_name: str) -> pd.Series:
    sub = oof_df[(oof_df["patient_id"] == patient_id) & (oof_df["model"] == model_name)].copy()
    if sub.empty:
        raise ValueError(f"patient_id {patient_id} with model {model_name} not found in oof_csv")
    if "fold" in sub.columns:
        sub = sub.sort_values("fold")
    sub = sub.drop_duplicates(subset=["patient_id", "model"], keep="first")
    return sub.iloc[0]


def edge_relation(g: nx.Graph, u: tuple[str, str], v: tuple[str, str]) -> str:
    if not g.has_edge(u, v):
        return ""
    d = g.get_edge_data(u, v) or {}
    rel = d.get("display_relation") or d.get("relation") or ""
    return str(rel)


def edge_origins_from_meta(meta: dict[frozenset, dict[str, tuple[str, ...]]], u: tuple[str, str], v: tuple[str, str]) -> list[str]:
    vals = meta.get(edge_key(u, v), {}).get("origins", ())
    return [str(x) for x in vals if str(x).strip()]


def edge_origins_from_bundle(bundle: GraphBundle, u: tuple[str, str], v: tuple[str, str]) -> list[str]:
    return edge_origins_from_meta(bundle.edge_meta, u, v)


def edge_origins_from_resolver(resolver: KGResolver, u: tuple[str, str], v: tuple[str, str]) -> list[str]:
    return edge_origins_from_meta(resolver.edge_meta, u, v)


def path_exists_in_bundle(path_nodes: list[tuple[str, str]], bundle: GraphBundle) -> bool:
    return all(edge_key(path_nodes[i], path_nodes[i + 1]) in bundle.edge_pairs for i in range(len(path_nodes) - 1))


def path_to_text(path_nodes: list[tuple[str, str]], g: nx.Graph) -> str:
    parts = []
    for i, node in enumerate(path_nodes):
        parts.append(f"{node[0]}:{pretty_name(node)}")
        if i < len(path_nodes) - 1:
            rel = edge_relation(g, path_nodes[i], path_nodes[i + 1])
            parts.append(f"--{rel or 'related_to'}-->")
    return " ".join(parts)


def choose_anchor_nodes(
    disease_nodes: Iterable[tuple[str, str]],
    anatomy_nodes: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    disease_nodes = list(disease_nodes)
    anatomy_nodes = list(anatomy_nodes)

    preferred_disease_nodes = [
        n for n in disease_nodes
        if is_pancreas_related_name(n[1]) and not is_generic_disease_node(n)
    ]
    preferred_anatomy_nodes = [
        n for n in anatomy_nodes
        if is_pancreas_related_name(n[1])
    ]

    if preferred_disease_nodes:
        return preferred_disease_nodes
    if preferred_anatomy_nodes:
        return preferred_anatomy_nodes

    non_generic_disease = [n for n in disease_nodes if not is_generic_disease_node(n)]
    if non_generic_disease:
        return non_generic_disease

    return anatomy_nodes


def node_degree_penalty(resolver: KGResolver, node: tuple[str, str]) -> float:
    return float(np.log1p(max(1, resolver.g.degree(node))))


def has_repeated_nodes(path_nodes: list[tuple[str, str]]) -> bool:
    return len(path_nodes) != len(set(path_nodes))


def score_local_segment(
    path_nodes: list[tuple[str, str]],
    merged_resolver: KGResolver,
    new_bundle: GraphBundle,
) -> float:
    mid_nodes = path_nodes[1:-1]
    edge_pairs = [edge_key(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]

    n_new_edges = sum(1 for ek in edge_pairs if ek in new_bundle.edge_pairs)
    n_pathway_like = sum(1 for n in path_nodes if is_pathway_like(n))
    n_pancreas_nodes = sum(1 for n in path_nodes if is_pancreas_related_name(n[1]))
    n_disease_mid = sum(1 for n in mid_nodes if norm_text(n[0]) == "disease")
    hub_penalty = sum(node_degree_penalty(merged_resolver, n) for n in mid_nodes)

    score = (
        8 * n_new_edges
        + 7 * n_pathway_like
        + 6 * n_pancreas_nodes
        - 16 * n_disease_mid
        - 2.5 * hub_penalty
        - 1.2 * (len(path_nodes) - 1)
    )

    if len(path_nodes) == 2 and n_pathway_like == 0:
        score -= 4.0

    return float(score)


def enumerate_best_two_hop_segments(
    src: tuple[str, str],
    dst: tuple[str, str],
    merged_resolver: KGResolver,
    new_bundle: GraphBundle,
    top_k: int = 6,
    allow_direct: bool = True,
    allow_disease_intermediate: bool = False,
    allow_bad_intermediate: bool = False,
) -> list[dict]:
    if src == dst:
        return []

    g = merged_resolver.g
    if (src not in g) or (dst not in g):
        return []

    candidates = []

    if allow_direct and g.has_edge(src, dst):
        p = [src, dst]
        candidates.append(
            {
                "path": p,
                "segment_score": score_local_segment(
                    path_nodes=p,
                    merged_resolver=merged_resolver,
                    new_bundle=new_bundle,
                ),
            }
        )

    common = merged_resolver.neighbor_index.get(src, set()) & merged_resolver.neighbor_index.get(dst, set())
    if len(common) > 10000:
        common = set(list(common)[:10000])

    for mid in common:
        if mid in {src, dst}:
            continue

        mid_is_disease = norm_text(mid[0]) == "disease"
        mid_is_bad = (not is_allowed_intermediate_type(mid)) and (not is_pathway_like(mid))

        if mid_is_disease and not allow_disease_intermediate:
            continue
        if mid_is_bad and not allow_bad_intermediate:
            continue

        p = [src, mid, dst]
        candidates.append(
            {
                "path": p,
                "segment_score": score_local_segment(
                    path_nodes=p,
                    merged_resolver=merged_resolver,
                    new_bundle=new_bundle,
                ),
            }
        )

    candidates = sorted(
        candidates,
        key=lambda x: (x["segment_score"], -len(x["path"])),
        reverse=True,
    )

    out = []
    seen = set()
    for item in candidates:
        sig = tuple(item["path"])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
        if len(out) >= top_k:
            break

    return out


def path_mode_pass(
    path_nodes: list[tuple[str, str]],
    min_pathway_nodes: int,
    allow_disease_intermediate: bool,
    max_disease_intermediate: int,
    allow_bad_intermediate: bool,
    max_bad_intermediate: int,
    strict_cross_cancer: bool,
) -> bool:
    if len(path_nodes) < 3:
        return False

    if is_generic_disease_node(path_nodes[0]):
        return False

    intermediate_nodes = path_nodes[1:-1]
    n_pathway_like = sum(1 for n in path_nodes if is_pathway_like(n))
    n_disease_mid = sum(1 for n in intermediate_nodes if norm_text(n[0]) == "disease")
    n_bad_mid = sum(
        1 for n in intermediate_nodes
        if (not is_allowed_intermediate_type(n)) and (not is_pathway_like(n))
    )
    n_cross_cancer = sum(1 for n in intermediate_nodes if is_cross_cancer_disease_node(n))

    if n_pathway_like < min_pathway_nodes:
        return False
    if (not allow_disease_intermediate) and n_disease_mid > 0:
        return False
    if n_disease_mid > max_disease_intermediate:
        return False
    if (not allow_bad_intermediate) and n_bad_mid > 0:
        return False
    if n_bad_mid > max_bad_intermediate:
        return False
    if strict_cross_cancer and n_cross_cancer > 0:
        return False
    if (not strict_cross_cancer) and n_cross_cancer > 1:
        return False

    return True


def score_explanation_path(
    path_nodes: list[tuple[str, str]],
    patient_gene: tuple[str, str],
    merged_resolver: KGResolver,
    prime_bundle: GraphBundle,
    new_bundle: GraphBundle,
    mode_bonus: float = 0.0,
) -> tuple[float, list[dict], list[str]]:
    path_len = len(path_nodes) - 1
    intermediate_nodes = path_nodes[1:-1]

    edge_details = []
    n_new_edges = 0
    new_origin_tags = set()

    for i in range(path_len):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        pair = edge_key(u, v)

        merged_origins = edge_origins_from_resolver(merged_resolver, u, v)
        new_origins = edge_origins_from_bundle(new_bundle, u, v)

        in_prime = int(pair in prime_bundle.edge_pairs)
        in_new = int(pair in new_bundle.edge_pairs)
        if in_new:
            n_new_edges += 1
            new_origin_tags.update(new_origins)

        edge_details.append(
            {
                "source_type": u[0],
                "source_name": u[1],
                "target_type": v[0],
                "target_name": v[1],
                "relation": edge_relation(merged_resolver.g, u, v),
                "in_primekg": in_prime,
                "in_new_triples": in_new,
                "merged_origins": merged_origins,
                "new_origins": new_origins,
            }
        )

    in_prime = path_exists_in_bundle(path_nodes, prime_bundle)
    novel_vs_prime = int((not in_prime) and n_new_edges > 0)

    n_pathway_like = int(sum(1 for n in path_nodes if is_pathway_like(n)))
    n_pancreas_nodes = int(sum(1 for n in path_nodes if is_pancreas_related_name(n[1])))
    n_cross_cancer_nodes = int(sum(1 for n in intermediate_nodes if is_cross_cancer_disease_node(n)))
    n_generic_nodes = int(sum(1 for n in path_nodes if is_generic_disease_node(n)))
    n_disease_mid = int(sum(1 for n in intermediate_nodes if norm_text(n[0]) == "disease"))
    n_bad_mid = int(
        sum(1 for n in intermediate_nodes if (not is_allowed_intermediate_type(n)) and (not is_pathway_like(n)))
    )

    direct_association_like = int(
        (path_len <= 2)
        and (n_pathway_like == 0)
        and (norm_text(path_nodes[0][0]) in {"disease", "anatomy"})
    )

    hub_penalty = (
        1.2 * node_degree_penalty(merged_resolver, patient_gene)
        + sum(node_degree_penalty(merged_resolver, n) for n in intermediate_nodes if n != patient_gene)
    )

    score = (
        30 * novel_vs_prime
        + 10 * n_new_edges
        + 14 * n_pathway_like
        + 10 * n_pancreas_nodes
        + 4 * len(new_origin_tags)
        - 20 * n_cross_cancer_nodes
        - 16 * n_generic_nodes
        - 14 * n_disease_mid
        - 8 * n_bad_mid
        - 10 * direct_association_like
        - 3.2 * hub_penalty
        - 2.0 * path_len
        + mode_bonus
    )

    if n_pathway_like == 0:
        score -= 10.0
    else:
        score += 6.0

    if novel_vs_prime == 1 and n_pathway_like > 0:
        score += 8.0

    if n_pancreas_nodes == 0:
        score -= 4.0

    return float(score), edge_details, sorted(new_origin_tags)


def select_diverse_top_paths(
    paths_df: pd.DataFrame,
    top_k: int,
    max_paths_per_gene: int = 2,
) -> pd.DataFrame:
    if paths_df.empty:
        return paths_df.copy()

    remaining = paths_df.copy()
    selected = []
    gene_counts: dict[str, int] = defaultdict(int)
    selected_node_sets: list[set[tuple[str, str]]] = []

    while len(selected) < top_k and not remaining.empty:
        best_idx = None
        best_adj_score = -1e18

        for idx, row in remaining.iterrows():
            gene_name = row["patient_gene_name"]
            if gene_counts[gene_name] >= max_paths_per_gene:
                continue

            node_set = {
                (norm_text(n["type"]), norm_text(n["name"]))
                for n in json.loads(row["path_nodes_json"])
            }

            overlap_pen = 0.0
            if selected_node_sets:
                overlap_pen = max(
                    len(node_set & s) / max(1, len(node_set | s))
                    for s in selected_node_sets
                )

            adj_score = float(row["score"]) - 18.0 * gene_counts[gene_name] - 12.0 * overlap_pen
            if adj_score > best_adj_score:
                best_adj_score = adj_score
                best_idx = idx

        if best_idx is None:
            break

        chosen = remaining.loc[[best_idx]].copy()
        chosen["diverse_rank"] = len(selected) + 1
        selected.append(chosen)

        chosen_row = chosen.iloc[0]
        gene_counts[chosen_row["patient_gene_name"]] += 1
        selected_node_sets.append(
            {
                (norm_text(n["type"]), norm_text(n["name"]))
                for n in json.loads(chosen_row["path_nodes_json"])
            }
        )

        remaining = remaining.drop(index=best_idx)

    if not selected:
        out = paths_df.head(top_k).copy()
        out["diverse_rank"] = np.arange(1, len(out) + 1)
        return out

    return pd.concat(selected, ignore_index=True)


def enumerate_patient_paths(
    patient_row: pd.Series,
    merged_resolver: KGResolver,
    prime_bundle: GraphBundle,
    new_bundle: GraphBundle,
    require_pathway: bool = False,
    max_path_len: int = 4,
    per_pair_top_k: int = 6,
    max_paths_per_gene_pair: int = 8,
    min_pathway_nodes: int = 1,
    mode_fill_target: int = 50,
) -> tuple[pd.DataFrame, dict[str, list[tuple[str, str]]], dict]:
    cancer_queries = [
        str(patient_row.get("cancer_type", "")),
        str(patient_row.get("cancer_type_detailed", "")),
    ]
    site_queries = [str(patient_row.get("primary_site", ""))]
    treatment_queries = list(patient_row.get("treatments_list", []))
    gene_queries = list(patient_row.get("altered_genes_list", []))

    disease_nodes = merged_resolver.resolve_unique(cancer_queries, ("disease",))
    anatomy_nodes = merged_resolver.resolve_unique(site_queries, ("anatomy",))
    drug_nodes = merged_resolver.resolve_unique(treatment_queries, ("drug",))
    gene_nodes = merged_resolver.resolve_unique(gene_queries, ("gene/protein",))

    primary_anchors = choose_anchor_nodes(disease_nodes, anatomy_nodes)
    fallback_anchors = [
        n for n in list(disease_nodes) + list(anatomy_nodes)
        if not is_generic_disease_node(n)
    ]
    anchors = list(dict.fromkeys(primary_anchors + fallback_anchors))

    if len(anchors) == 0:
        raise ValueError("No disease/site anchor node could be resolved for this patient.")
    if len(drug_nodes) == 0:
        raise ValueError("No treatment nodes could be resolved for this patient.")
    if len(gene_nodes) == 0:
        raise ValueError("No gene nodes could be resolved for this patient.")

    requested_min_pathway = max(min_pathway_nodes, 1) if require_pathway else max(min_pathway_nodes, 0)

    search_modes = [
        {
            "name": "strict_pathway",
            "min_pathway_nodes": max(1, requested_min_pathway),
            "allow_disease_intermediate": False,
            "max_disease_intermediate": 0,
            "allow_bad_intermediate": False,
            "max_bad_intermediate": 0,
            "strict_cross_cancer": True,
            "mode_bonus": 14.0,
        },
        {
            "name": "strict_no_disease",
            "min_pathway_nodes": 0,
            "allow_disease_intermediate": False,
            "max_disease_intermediate": 0,
            "allow_bad_intermediate": False,
            "max_bad_intermediate": 0,
            "strict_cross_cancer": False,
            "mode_bonus": 8.0,
        },
        {
            "name": "relaxed_one_disease",
            "min_pathway_nodes": 0,
            "allow_disease_intermediate": True,
            "max_disease_intermediate": 1,
            "allow_bad_intermediate": False,
            "max_bad_intermediate": 0,
            "strict_cross_cancer": False,
            "mode_bonus": 0.0,
        },
        {
            "name": "relaxed_one_disease_one_bad",
            "min_pathway_nodes": 0,
            "allow_disease_intermediate": True,
            "max_disease_intermediate": 1,
            "allow_bad_intermediate": True,
            "max_bad_intermediate": 1,
            "strict_cross_cancer": False,
            "mode_bonus": -6.0,
        },
    ]

    all_records_by_sig: dict[tuple[tuple[str, str], ...], dict] = {}
    mode_stats = []
    segment_cache: dict[tuple[tuple[str, str], tuple[str, str], str], list[dict]] = {}

    print(f"[INFO] Resolved anchors={len(anchors)}, genes={len(gene_nodes)}, drugs={len(drug_nodes)}, diseases={len(disease_nodes)}, anatomy={len(anatomy_nodes)}")

    for mode in search_modes:
        mode_name = mode["name"]
        mode_added = 0
        mode_candidate_count = 0

        for anchor in anchors:
            for gene in gene_nodes:
                ag_key = (anchor, gene, f"ag::{mode_name}")
                if ag_key not in segment_cache:
                    segment_cache[ag_key] = enumerate_best_two_hop_segments(
                        src=anchor,
                        dst=gene,
                        merged_resolver=merged_resolver,
                        new_bundle=new_bundle,
                        top_k=per_pair_top_k,
                        allow_direct=True,
                        allow_disease_intermediate=mode["allow_disease_intermediate"],
                        allow_bad_intermediate=mode["allow_bad_intermediate"],
                    )

        for gene in gene_nodes:
            for drug in drug_nodes:
                gd_key = (gene, drug, f"gd::{mode_name}")
                if gd_key not in segment_cache:
                    segment_cache[gd_key] = enumerate_best_two_hop_segments(
                        src=gene,
                        dst=drug,
                        merged_resolver=merged_resolver,
                        new_bundle=new_bundle,
                        top_k=per_pair_top_k,
                        allow_direct=True,
                        allow_disease_intermediate=mode["allow_disease_intermediate"],
                        allow_bad_intermediate=mode["allow_bad_intermediate"],
                    )

        for anchor in anchors:
            for gene in gene_nodes:
                ag_segments = segment_cache.get((anchor, gene, f"ag::{mode_name}"), [])
                if not ag_segments:
                    continue

                for drug in drug_nodes:
                    gd_segments = segment_cache.get((gene, drug, f"gd::{mode_name}"), [])
                    if not gd_segments:
                        continue

                    combo_candidates = []
                    for seg1 in ag_segments:
                        for seg2 in gd_segments:
                            full = seg1["path"] + seg2["path"][1:]
                            if len(full) < 3:
                                continue
                            if (len(full) - 1) > max_path_len:
                                continue
                            if has_repeated_nodes(full):
                                continue
                            if not path_mode_pass(
                                path_nodes=full,
                                min_pathway_nodes=mode["min_pathway_nodes"],
                                allow_disease_intermediate=mode["allow_disease_intermediate"],
                                max_disease_intermediate=mode["max_disease_intermediate"],
                                allow_bad_intermediate=mode["allow_bad_intermediate"],
                                max_bad_intermediate=mode["max_bad_intermediate"],
                                strict_cross_cancer=mode["strict_cross_cancer"],
                            ):
                                continue

                            score, edge_details, new_origin_tags = score_explanation_path(
                                path_nodes=full,
                                patient_gene=gene,
                                merged_resolver=merged_resolver,
                                prime_bundle=prime_bundle,
                                new_bundle=new_bundle,
                                mode_bonus=mode["mode_bonus"],
                            )
                            combo_score = float(score + 0.5 * seg1["segment_score"] + 0.5 * seg2["segment_score"])
                            combo_candidates.append((combo_score, full, edge_details, new_origin_tags))

                    if not combo_candidates:
                        continue

                    combo_candidates = sorted(combo_candidates, key=lambda x: x[0], reverse=True)
                    mode_candidate_count += len(combo_candidates)

                    for combo_score, full, edge_details, new_origin_tags in combo_candidates[:max_paths_per_gene_pair]:
                        sig = tuple(full)

                        path_len = len(full) - 1
                        in_prime = path_exists_in_bundle(full, prime_bundle)
                        n_new_edges = int(sum(1 for i in range(path_len) if edge_key(full[i], full[i + 1]) in new_bundle.edge_pairs))
                        novel_vs_prime = int((not in_prime) and n_new_edges > 0)
                        n_pathway_like = int(sum(1 for n in full if is_pathway_like(n)))
                        n_pancreas_nodes = int(sum(1 for n in full if is_pancreas_related_name(n[1])))
                        n_cross_cancer_nodes = int(sum(1 for n in full[1:-1] if is_cross_cancer_disease_node(n)))
                        n_generic_nodes = int(sum(1 for n in full if is_generic_disease_node(n)))

                        if n_pathway_like > 0 and novel_vs_prime == 1:
                            mechanism_tier = "mechanistic_novel"
                        elif n_pathway_like > 0:
                            mechanism_tier = "mechanistic_supported"
                        elif novel_vs_prime == 1:
                            mechanism_tier = "novel_association"
                        else:
                            mechanism_tier = "association_like"

                        rels = [edge_relation(merged_resolver.g, full[i], full[i + 1]) for i in range(path_len)]

                        record = {
                            "anchor_node_type": anchor[0],
                            "anchor_node_name": anchor[1],
                            "patient_gene_type": gene[0],
                            "patient_gene_name": gene[1],
                            "patient_drug_type": drug[0],
                            "patient_drug_name": drug[1],
                            "path_len": int(path_len),
                            "n_new_edges": int(n_new_edges),
                            "uses_new_edges": int(n_new_edges > 0),
                            "path_exists_in_primekg": int(in_prime),
                            "novel_vs_primekg": int(novel_vs_prime),
                            "n_pathway_like_nodes": int(n_pathway_like),
                            "n_pancreas_nodes": int(n_pancreas_nodes),
                            "n_cross_cancer_nodes": int(n_cross_cancer_nodes),
                            "n_generic_nodes": int(n_generic_nodes),
                            "n_new_origins": int(len(new_origin_tags)),
                            "mechanism_tier": mechanism_tier,
                            "search_mode": mode_name,
                            "patient_gene_degree": int(merged_resolver.g.degree(gene)),
                            "score": float(combo_score),
                            "path_text": path_to_text(full, merged_resolver.g),
                            "path_nodes_json": json.dumps(
                                [{"type": n[0], "name": n[1]} for n in full],
                                ensure_ascii=False,
                            ),
                            "path_relations_json": json.dumps(rels, ensure_ascii=False),
                            "path_edge_details_json": json.dumps(edge_details, ensure_ascii=False),
                            "new_edge_origin_tags": "|".join(new_origin_tags),
                            "new_edge_origins_json": json.dumps(new_origin_tags, ensure_ascii=False),
                        }

                        prev = all_records_by_sig.get(sig)
                        if prev is None or float(record["score"]) > float(prev["score"]):
                            if prev is None:
                                mode_added += 1
                            all_records_by_sig[sig] = record

        mode_stats.append(
            {
                "mode": mode_name,
                "unique_paths_after_mode": len(all_records_by_sig),
                "mode_new_unique_paths": mode_added,
                "raw_mode_candidates": mode_candidate_count,
            }
        )
        print(f"[INFO] mode={mode_name} new_unique={mode_added} total_unique={len(all_records_by_sig)} raw_candidates={mode_candidate_count}")

        if len(all_records_by_sig) >= mode_fill_target:
            break

    out = pd.DataFrame(list(all_records_by_sig.values()))
    diagnostics = {
        "requested_min_pathway_nodes": requested_min_pathway,
        "mode_stats": mode_stats,
        "resolved_anchor_count": len(anchors),
        "resolved_gene_count": len(gene_nodes),
        "resolved_drug_count": len(drug_nodes),
        "resolved_disease_count": len(disease_nodes),
        "resolved_anatomy_count": len(anatomy_nodes),
    }

    if out.empty:
        return out, {
            "anchors": list(anchors),
            "genes": list(gene_nodes),
            "drugs": list(drug_nodes),
            "anatomy": list(anatomy_nodes),
            "disease": list(disease_nodes),
        }, diagnostics

    out = out.sort_values(
        [
            "novel_vs_primekg",
            "n_pathway_like_nodes",
            "n_pancreas_nodes",
            "n_new_origins",
            "score",
            "path_len",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    resolved = {
        "anchors": list(anchors),
        "genes": list(gene_nodes),
        "drugs": list(drug_nodes),
        "anatomy": list(anatomy_nodes),
        "disease": list(disease_nodes),
    }
    return out, resolved, diagnostics


def build_subgraph_exports(
    top_paths_df: pd.DataFrame,
    merged_resolver: KGResolver,
    prime_bundle: GraphBundle,
    new_bundle: GraphBundle,
    patient_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = []
    edge_rows = []
    seen_nodes = set()
    seen_edges = set()

    patient_gene_set = {norm_text(g) for g in patient_row["altered_genes_list"]}
    patient_tx_set = {norm_text(t) for t in patient_row["treatments_list"]}
    patient_anchor_names = {
        norm_text(str(patient_row.get("cancer_type", ""))),
        norm_text(str(patient_row.get("cancer_type_detailed", ""))),
        norm_text(str(patient_row.get("primary_site", ""))),
    }

    for _, row in top_paths_df.iterrows():
        nodes = json.loads(row["path_nodes_json"])
        path_nodes = [(norm_text(n["type"]), norm_text(n["name"])) for n in nodes]

        for n in path_nodes:
            nid = node_to_id(n)
            if nid not in seen_nodes:
                seen_nodes.add(nid)
                node_rows.append(
                    {
                        "node_id": nid,
                        "node_type": n[0],
                        "node_name": n[1],
                        "pretty_name": pretty_name(n),
                        "is_patient_gene": int(n[0] == "gene/protein" and n[1] in patient_gene_set),
                        "is_patient_drug": int(n[0] == "drug" and n[1] in patient_tx_set),
                        "is_patient_anchor": int(n[1] in patient_anchor_names or is_pancreas_related_name(n[1])),
                        "is_pathway_like": int(is_pathway_like(n)),
                        "is_pancreas_related": int(is_pancreas_related_name(n[1])),
                    }
                )

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            ek = (node_to_id(u), node_to_id(v))
            ek_sorted = tuple(sorted(ek))
            if ek_sorted in seen_edges:
                continue
            seen_edges.add(ek_sorted)

            pair = edge_key(u, v)
            merged_origins = edge_origins_from_resolver(merged_resolver, u, v)
            new_origins = edge_origins_from_bundle(new_bundle, u, v)

            edge_rows.append(
                {
                    "source_id": node_to_id(u),
                    "target_id": node_to_id(v),
                    "source_name": pretty_name(u),
                    "target_name": pretty_name(v),
                    "source_type": u[0],
                    "target_type": v[0],
                    "relation": edge_relation(merged_resolver.g, u, v),
                    "in_primekg": int(pair in prime_bundle.edge_pairs),
                    "in_new_triples": int(pair in new_bundle.edge_pairs),
                    "novel_vs_primekg": int(pair not in prime_bundle.edge_pairs and pair in new_bundle.edge_pairs),
                    "origin_tags": "|".join(merged_origins),
                    "origins_json": json.dumps(merged_origins, ensure_ascii=False),
                    "new_origin_tags": "|".join(new_origins),
                    "new_origins_json": json.dumps(new_origins, ensure_ascii=False),
                }
            )

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    return nodes_df, edges_df


def draw_explanation_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    out_png: str,
) -> None:
    if nodes_df.empty or edges_df.empty:
        print("[WARN] No nodes/edges available for plotting.")
        return

    g = nx.Graph()
    for _, row in nodes_df.iterrows():
        g.add_node(
            row["node_id"],
            label=row["pretty_name"],
            node_type=row["node_type"],
            is_patient_gene=int(row["is_patient_gene"]),
            is_patient_drug=int(row["is_patient_drug"]),
            is_patient_anchor=int(row["is_patient_anchor"]),
            is_pathway_like=int(row["is_pathway_like"]),
            is_pancreas_related=int(row["is_pancreas_related"]),
        )

    for _, row in edges_df.iterrows():
        g.add_edge(
            row["source_id"],
            row["target_id"],
            relation=row["relation"],
            novel_vs_primekg=int(row["novel_vs_primekg"]),
        )

    pos = nx.spring_layout(g, seed=42, k=1.1)

    plt.figure(figsize=(11, 8))

    node_colors = []
    node_sizes = []
    for _, d in g.nodes(data=True):
        if d["is_patient_anchor"] == 1:
            node_colors.append("#d62728")
            node_sizes.append(1600)
        elif d["is_patient_gene"] == 1:
            node_colors.append("#2ca02c")
            node_sizes.append(1300)
        elif d["is_patient_drug"] == 1:
            node_colors.append("#1f77b4")
            node_sizes.append(1300)
        elif d["is_pathway_like"] == 1:
            node_colors.append("#9467bd")
            node_sizes.append(1100)
        elif d["is_pancreas_related"] == 1:
            node_colors.append("#ff9896")
            node_sizes.append(1000)
        elif d["node_type"] == "gene/protein":
            node_colors.append("#98df8a")
            node_sizes.append(900)
        elif d["node_type"] == "drug":
            node_colors.append("#9ecae1")
            node_sizes.append(900)
        elif d["node_type"] == "disease":
            node_colors.append("#f7b6b2")
            node_sizes.append(950)
        else:
            node_colors.append("#d9d9d9")
            node_sizes.append(760)

    edge_colors = []
    edge_widths = []
    for _, _, d in g.edges(data=True):
        if int(d.get("novel_vs_primekg", 0)) == 1:
            edge_colors.append("#ff7f0e")
            edge_widths.append(3.0)
        else:
            edge_colors.append("#999999")
            edge_widths.append(1.8)

    nx.draw_networkx_edges(g, pos, edge_color=edge_colors, width=edge_widths, alpha=0.9)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.9, edgecolors="black")

    labels = {n: d["label"] for n, d in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9)

    edge_labels = {}
    for u, v, d in g.edges(data=True):
        rel = str(d.get("relation", "")).strip()
        if rel:
            edge_labels[(u, v)] = rel[:18]
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7)

    plt.title("MedGraphusion patient-specific explanation network", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved network figure to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_csv", required=True)
    parser.add_argument("--oof_csv", required=True)
    parser.add_argument("--primekg_csv", required=True)
    parser.add_argument("--new_triples_csv", required=True)
    parser.add_argument("--merged_kg_csv", required=True)
    parser.add_argument("--out_dir", default="medgraphusion_explanation_case")

    parser.add_argument("--patient_id", default=None)
    parser.add_argument("--model", default="clinical_treatment_molecular_kg")
    parser.add_argument("--cancer_type", default="Pancreatic Cancer")
    parser.add_argument("--stage_group", default="Stage IV")
    parser.add_argument("--top_k_paths", type=int, default=8)
    parser.add_argument("--require_pathway", action="store_true")
    parser.add_argument("--max_path_len", type=int, default=4)

    parser.add_argument("--segment_top_k", type=int, default=6)
    parser.add_argument("--max_paths_per_gene_pair", type=int, default=8)
    parser.add_argument("--min_pathway_nodes", type=int, default=1)
    parser.add_argument("--max_paths_per_gene", type=int, default=2)
    parser.add_argument("--mode_fill_target", type=int, default=50)

    parser.add_argument("--cache_dir", default=".kg_cache")
    parser.add_argument("--force_rebuild_cache", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cohort_df = load_cohort(args.cohort_csv)
    oof_df = load_oof(args.oof_csv)

    print("Loading/building graphs with cache...")
    prime_resolver = KGResolver.from_cache_or_build(
        kg_csv=args.primekg_csv,
        name="primekg",
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )
    new_resolver = KGResolver.from_cache_or_build(
        kg_csv=args.new_triples_csv,
        name="new_triples",
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )
    merged_resolver = KGResolver.from_cache_or_build(
        kg_csv=args.merged_kg_csv,
        name="merged",
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )

    patient_id = args.patient_id
    if patient_id is None:
        patient_id = choose_representative_patient(
            oof_df=oof_df,
            cohort_df=cohort_df,
            resolver=merged_resolver,
            model_name=args.model,
            cancer_type=args.cancer_type,
            stage_group=args.stage_group,
        )
        print(f"[INFO] Auto-selected representative patient: {patient_id}")
    else:
        print(f"[INFO] Using user-specified patient: {patient_id}")

    patient_row = get_patient_row(cohort_df, patient_id)
    pred_row = get_oof_row(oof_df, patient_id, args.model)

    paths_df, resolved, diagnostics = enumerate_patient_paths(
        patient_row=patient_row,
        merged_resolver=merged_resolver,
        prime_bundle=prime_resolver.bundle,
        new_bundle=new_resolver.bundle,
        require_pathway=args.require_pathway,
        max_path_len=args.max_path_len,
        per_pair_top_k=args.segment_top_k,
        max_paths_per_gene_pair=args.max_paths_per_gene_pair,
        min_pathway_nodes=args.min_pathway_nodes,
        mode_fill_target=args.mode_fill_target,
    )

    if paths_df.empty:
        print("[WARN] No explanation paths found under current constraints and fallback modes.")
        summary = {
            "patient_id": patient_id,
            "model": args.model,
            "risk_score": float(pred_row["risk_score"]),
            "cancer_type": str(patient_row.get("cancer_type", "")),
            "cancer_type_detailed": str(patient_row.get("cancer_type_detailed", "")),
            "stage": str(patient_row.get("stage", "")),
            "stage_group": str(pred_row.get("stage_group", "")),
            "treatments_list": patient_row["treatments_list"],
            "altered_genes_list": patient_row["altered_genes_list"],
            "resolved_anchor_nodes": [node_to_id(n) for n in resolved["anchors"]],
            "resolved_disease_nodes": [node_to_id(n) for n in resolved["disease"]],
            "resolved_anatomy_nodes": [node_to_id(n) for n in resolved["anatomy"]],
            "resolved_gene_nodes": [node_to_id(n) for n in resolved["genes"]],
            "resolved_drug_nodes": [node_to_id(n) for n in resolved["drugs"]],
            "n_paths_found": 0,
            "search_diagnostics": diagnostics,
        }
        with open(out_dir / "selected_patient_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    top_paths_df = select_diverse_top_paths(
        paths_df=paths_df,
        top_k=args.top_k_paths,
        max_paths_per_gene=args.max_paths_per_gene,
    ).copy()

    if "diverse_rank" in top_paths_df.columns:
        top_paths_df = top_paths_df.sort_values("diverse_rank").reset_index(drop=True)

    nodes_df, edges_df = build_subgraph_exports(
        top_paths_df=top_paths_df,
        merged_resolver=merged_resolver,
        prime_bundle=prime_resolver.bundle,
        new_bundle=new_resolver.bundle,
        patient_row=patient_row,
    )

    summary = {
        "patient_id": patient_id,
        "model": args.model,
        "risk_score": float(pred_row["risk_score"]),
        "fold": int(pred_row["fold"]) if "fold" in pred_row.index and pd.notna(pred_row["fold"]) else None,
        "cancer_type": str(patient_row.get("cancer_type", "")),
        "cancer_type_detailed": str(patient_row.get("cancer_type_detailed", "")),
        "stage": str(patient_row.get("stage", "")),
        "stage_group": str(pred_row.get("stage_group", "")),
        "os_time": float(pred_row["os_time"]) if pd.notna(pred_row["os_time"]) else None,
        "os_event": int(pred_row["os_event"]) if pd.notna(pred_row["os_event"]) else None,
        "treatments_list": patient_row["treatments_list"],
        "altered_genes_list": patient_row["altered_genes_list"],
        "resolved_anchor_nodes": [node_to_id(n) for n in resolved["anchors"]],
        "resolved_disease_nodes": [node_to_id(n) for n in resolved["disease"]],
        "resolved_anatomy_nodes": [node_to_id(n) for n in resolved["anatomy"]],
        "resolved_gene_nodes": [node_to_id(n) for n in resolved["genes"]],
        "resolved_drug_nodes": [node_to_id(n) for n in resolved["drugs"]],
        "n_paths_found": int(len(paths_df)),
        "n_top_paths_exported": int(len(top_paths_df)),
        "n_top_paths_novel_vs_primekg": int(top_paths_df["novel_vs_primekg"].sum()),
        "n_top_paths_using_new_edges": int(top_paths_df["uses_new_edges"].sum()) if "uses_new_edges" in top_paths_df.columns else None,
        "top_mechanism_tiers": top_paths_df["mechanism_tier"].value_counts().to_dict() if "mechanism_tier" in top_paths_df.columns else {},
        "search_diagnostics": diagnostics,
    }

    with open(out_dir / "selected_patient_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    paths_df.to_csv(out_dir / "explanation_paths_ranked.csv", index=False)
    top_paths_df.to_csv(out_dir / "explanation_paths_top.csv", index=False)
    nodes_df.to_csv(out_dir / "explanation_nodes.csv", index=False)
    edges_df.to_csv(out_dir / "explanation_edges.csv", index=False)

    draw_explanation_network(
        nodes_df=nodes_df,
        edges_df=edges_df,
        out_png=str(out_dir / "explanation_network.png"),
    )

    print("[OK] Saved outputs:")
    print(f"- {out_dir / 'selected_patient_summary.json'}")
    print(f"- {out_dir / 'explanation_paths_ranked.csv'}")
    print(f"- {out_dir / 'explanation_paths_top.csv'}")
    print(f"- {out_dir / 'explanation_nodes.csv'}")
    print(f"- {out_dir / 'explanation_edges.csv'}")
    print(f"- {out_dir / 'explanation_network.png'}")


if __name__ == "__main__":
    main()