from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================
# Global plotting style
# =====================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "red": "#E45756",
    "green": "#54A24B",
    "teal": "#72B7B2",
    "purple": "#B279A2",
    "gray": "#9D9D9D",
    "dark": "#333333",
    "light_blue": "#AEC7E8",
    "light_orange": "#FFBB78",
}


# =====================
# Utilities
# =====================
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_entropy(counter_dict: dict[str, int]) -> float:
    vals = np.array(list(counter_dict.values()), dtype=float)
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def normalize_str(x) -> str:
    """
    Normalize missing or empty strings to 'none'.

    Note:
    - 'none' is treated as an unmapped placeholder entity type.
    - It can later be removed through --exclude-types none.
    """
    if pd.isna(x):
        return "none"
    s = str(x).strip()
    return s if s else "none"


def read_kg_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = [
        "relation", "display_relation",
        "x_id", "x_type", "x_name",
        "y_id", "y_type", "y_name"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    for col in [
        "relation", "display_relation",
        "x_id", "x_type", "x_name",
        "y_id", "y_type", "y_name"
    ]:
        df[col] = df[col].map(normalize_str)

    if "_origins" in df.columns:
        df["_origins"] = df["_origins"].fillna("").astype(str)
    else:
        df["_origins"] = ""

    if "_input_predicate" in df.columns:
        df["_input_predicate"] = df["_input_predicate"].fillna("").astype(str)
    else:
        df["_input_predicate"] = ""

    return df


def extract_nodes(df: pd.DataFrame) -> pd.DataFrame:
    left = df[["x_id", "x_type", "x_name"]].copy()
    left.columns = ["node_id", "node_type", "node_name"]

    right = df[["y_id", "y_type", "y_name"]].copy()
    right.columns = ["node_id", "node_type", "node_name"]

    nodes = pd.concat([left, right], ignore_index=True)

    # 一个 node_id 可能理论上对应多个 type，这里沿用原逻辑：
    # 以 node_id + node_type 去重。
    nodes = nodes.drop_duplicates(subset=["node_id", "node_type"])
    return nodes


def node_type_counts(df: pd.DataFrame) -> dict[str, int]:
    nodes = extract_nodes(df)
    counts = nodes["node_type"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def display_relation_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["display_relation"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def raw_relation_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["relation"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def get_triple_set(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(zip(df["x_id"], df["display_relation"], df["y_id"]))


def get_entity_pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["x_id"], df["y_id"]))


def get_type_pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["x_type"], df["y_type"]))


def estimate_2hop_paths(df: pd.DataFrame) -> int:
    """
    Estimate number of directed 2-hop paths x -> mid -> y.

    For each intermediate node v:
        number of 2-hop paths through v = indeg(v) * outdeg(v)
    """
    out_deg = Counter(df["x_id"].tolist())
    in_deg = Counter(df["y_id"].tolist())

    total = 0
    for nid in set(out_deg.keys()) | set(in_deg.keys()):
        total += out_deg.get(nid, 0) * in_deg.get(nid, 0)

    return int(total)


def classify_edge_source(row: pd.Series) -> str:
    """
    基于 merged_kg.csv 的字段判断边来源：
    - 如果 _origins 非空且不是 []，视为 literature
    - 否则视为 database/base

    目前主分析未使用此函数，但保留以支持后续 provenance 分析。
    """
    origins = str(row.get("_origins", "")).strip()
    if origins and origins != "[]" and origins.lower() != "nan":
        return "literature"
    return "database"


# =====================
# Filtering
# =====================
def find_low_frequency_types(
    merged_df: pd.DataFrame,
    min_nodes_per_type: int = 10,
    min_type_ratio: float = 0.0
) -> tuple[set[str], pd.DataFrame]:
    """
    Detect entity types with low support in the merged graph.

    A type is considered low-frequency if:
    - node_count < min_nodes_per_type, or
    - node_count / total_nodes < min_type_ratio
    """
    nodes = extract_nodes(merged_df)

    counts = (
        nodes["node_type"]
        .value_counts()
        .rename_axis("node_type")
        .reset_index(name="node_count")
    )

    total_nodes = counts["node_count"].sum()
    counts["ratio"] = counts["node_count"] / total_nodes if total_nodes > 0 else 0.0

    low_types = set(
        counts.loc[
            (counts["node_count"] < min_nodes_per_type) |
            (counts["ratio"] < min_type_ratio),
            "node_type"
        ].astype(str).tolist()
    )

    return low_types, counts


def normalize_type_set(types: list[str] | set[str] | tuple[str, ...] | None) -> set[str]:
    """
    Normalize a collection of entity type labels.

    This keeps exact labels after strip(), because KG type labels may be case-sensitive.
    """
    if not types:
        return set()
    return {str(t).strip() for t in types if str(t).strip()}


def add_forced_excluded_types(
    low_types: set[str],
    force_exclude_types: set[str] | None = None
) -> set[str]:
    """
    Add forced excluded entity types to the low-frequency type set.

    Example:
    - low_types from frequency filtering
    - force_exclude_types = {'none'}

    Final remove set = low_types ∪ force_exclude_types
    """
    low_types = set(low_types)
    force_exclude_types = normalize_type_set(force_exclude_types)
    low_types.update(force_exclude_types)
    return low_types


def filter_graph_by_entity_types(
    df: pd.DataFrame,
    remove_types: set[str]
) -> pd.DataFrame:
    """
    Remove any triple whose subject or object type belongs to remove_types.
    """
    remove_types = normalize_type_set(remove_types)
    if not remove_types:
        return df.copy()

    keep_mask = (~df["x_type"].isin(remove_types)) & (~df["y_type"].isin(remove_types))
    return df.loc[keep_mask].copy()


def build_filtering_summary(
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    low_frequency_types: set[str],
    forced_excluded_types: set[str],
    final_removed_types: set[str]
) -> pd.DataFrame:
    """
    Build a clear summary table for filtering.
    """
    low_frequency_types = normalize_type_set(low_frequency_types)
    forced_excluded_types = normalize_type_set(forced_excluded_types)
    final_removed_types = normalize_type_set(final_removed_types)

    original_nodes = extract_nodes(original_df)
    filtered_nodes = extract_nodes(filtered_df)

    original_node_set = set(zip(original_nodes["node_id"], original_nodes["node_type"]))
    filtered_node_set = set(zip(filtered_nodes["node_id"], filtered_nodes["node_type"]))

    removed_node_count = len(original_node_set - filtered_node_set)

    rows = [
        {
            "item": "original_merged_triples",
            "value": len(original_df)
        },
        {
            "item": "filtered_merged_triples",
            "value": len(filtered_df)
        },
        {
            "item": "removed_triples",
            "value": len(original_df) - len(filtered_df)
        },
        {
            "item": "original_merged_nodes",
            "value": len(original_node_set)
        },
        {
            "item": "filtered_merged_nodes",
            "value": len(filtered_node_set)
        },
        {
            "item": "removed_nodes",
            "value": removed_node_count
        },
        {
            "item": "low_frequency_entity_type_count",
            "value": len(low_frequency_types)
        },
        {
            "item": "low_frequency_entity_types",
            "value": "; ".join(sorted(low_frequency_types))
        },
        {
            "item": "forced_excluded_entity_type_count",
            "value": len(forced_excluded_types)
        },
        {
            "item": "forced_excluded_entity_types",
            "value": "; ".join(sorted(forced_excluded_types))
        },
        {
            "item": "final_removed_entity_type_count",
            "value": len(final_removed_types)
        },
        {
            "item": "final_removed_entity_types",
            "value": "; ".join(sorted(final_removed_types))
        },
    ]

    return pd.DataFrame(rows)


# =====================
# Quantitative analysis
# =====================
def build_quant_report(
    base_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    base_name: str = "PrimeKG"
) -> dict:
    base_nodes = extract_nodes(base_df)
    merged_nodes = extract_nodes(merged_df)

    base_node_set = set(zip(base_nodes["node_id"], base_nodes["node_type"]))
    merged_node_set = set(zip(merged_nodes["node_id"], merged_nodes["node_type"]))

    new_node_set = merged_node_set - base_node_set

    base_triples = get_triple_set(base_df)
    merged_triples = get_triple_set(merged_df)
    new_triples = merged_triples - base_triples

    base_pairs = get_entity_pair_set(base_df)
    merged_pairs = get_entity_pair_set(merged_df)
    new_pairs = merged_pairs - base_pairs

    base_type_pairs = get_type_pair_set(base_df)
    merged_type_pairs = get_type_pair_set(merged_df)
    new_type_pairs = merged_type_pairs - base_type_pairs

    base_rel_raw = raw_relation_counts(base_df)
    merged_rel_raw = raw_relation_counts(merged_df)

    base_rel_disp = display_relation_counts(base_df)
    merged_rel_disp = display_relation_counts(merged_df)

    base_nodes_by_type = node_type_counts(base_df)
    merged_nodes_by_type = node_type_counts(merged_df)

    new_nodes_by_type = {}
    for node_id, node_type in new_node_set:
        new_nodes_by_type[node_type] = new_nodes_by_type.get(node_type, 0) + 1

    new_nodes_by_type = dict(
        sorted(new_nodes_by_type.items(), key=lambda x: x[1], reverse=True)
    )

    base_2hop = estimate_2hop_paths(base_df)
    merged_2hop = estimate_2hop_paths(merged_df)

    new_relation_types = set(merged_rel_raw.keys()) - set(base_rel_raw.keys())
    new_display_relation_types = set(merged_rel_disp.keys()) - set(base_rel_disp.keys())

    # 新增边集合层面的 novelty
    # 由于 new_triples 本身定义为 merged - base，因此 novel triple rate 恒为 100%，
    # 这里保留该指标主要用于汇报。
    novel_triple_rate = 100.0 if len(new_triples) > 0 else 0.0

    # 新增 entity pair 同理。
    novel_entity_pair_rate = 100.0 if len(new_pairs) > 0 else 0.0

    report = {
        "base_name": base_name,

        "base_node_count": int(len(base_node_set)),
        "merged_node_count": int(len(merged_node_set)),
        "new_node_count": int(len(new_node_set)),
        "node_coverage_lift_pct": round(
            (len(new_node_set) / len(base_node_set) * 100) if len(base_node_set) else 0,
            4
        ),

        "base_triple_count": int(len(base_df)),
        "merged_triple_count": int(len(merged_df)),
        "new_triple_count": int(len(new_triples)),
        "triple_growth_pct": round(
            ((len(merged_df) - len(base_df)) / len(base_df) * 100) if len(base_df) else 0,
            4
        ),

        "base_entity_type_count": int(len(base_nodes_by_type)),
        "merged_entity_type_count": int(len(merged_nodes_by_type)),
        "new_entity_type_count": int(
            len(set(merged_nodes_by_type.keys()) - set(base_nodes_by_type.keys()))
        ),

        "base_relation_count": int(len(base_rel_raw)),
        "merged_relation_count": int(len(merged_rel_raw)),
        "new_relation_count": int(len(new_relation_types)),
        "relation_diversity_lift_pct": round(
            ((len(merged_rel_raw) - len(base_rel_raw)) / len(base_rel_raw) * 100)
            if len(base_rel_raw) else 0,
            4
        ),

        "base_display_relation_count": int(len(base_rel_disp)),
        "merged_display_relation_count": int(len(merged_rel_disp)),
        "new_display_relation_count": int(len(new_display_relation_types)),

        "base_relation_entropy": round(safe_entropy(base_rel_raw), 6),
        "merged_relation_entropy": round(safe_entropy(merged_rel_raw), 6),
        "base_display_relation_entropy": round(safe_entropy(base_rel_disp), 6),
        "merged_display_relation_entropy": round(safe_entropy(merged_rel_disp), 6),

        "base_2hop_paths": int(base_2hop),
        "merged_2hop_paths": int(merged_2hop),
        "estimated_new_2hop_paths": int(merged_2hop - base_2hop),

        "new_entity_pair_count": int(len(new_pairs)),
        "new_type_pair_count": int(len(new_type_pairs)),

        "novel_triple_rate_pct": round(novel_triple_rate, 4),
        "novel_entity_pair_rate_pct": round(novel_entity_pair_rate, 4),
        "novel_relation_usage_rate_pct": round(
            (len(new_relation_types) / len(merged_rel_raw) * 100) if len(merged_rel_raw) else 0,
            4
        ),
        "novel_type_pair_rate_pct": round(
            (len(new_type_pairs) / len(merged_type_pairs) * 100) if len(merged_type_pairs) else 0,
            4
        ),

        "base_nodes_by_type": dict(
            sorted(base_nodes_by_type.items(), key=lambda x: x[1], reverse=True)
        ),
        "merged_nodes_by_type": dict(
            sorted(merged_nodes_by_type.items(), key=lambda x: x[1], reverse=True)
        ),
        "new_nodes_by_type": new_nodes_by_type,

        "base_relations_by_type": dict(
            sorted(base_rel_disp.items(), key=lambda x: x[1], reverse=True)
        ),
        "merged_relations_by_type": dict(
            sorted(merged_rel_disp.items(), key=lambda x: x[1], reverse=True)
        ),
    }

    return report


# =====================
# Export helpers
# =====================
def save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def export_summary_table(report: dict, out_dir: Path) -> pd.DataFrame:
    rows = [
        ["Base node count", report["base_node_count"]],
        ["Merged node count", report["merged_node_count"]],
        ["New node count", report["new_node_count"]],
        ["Node coverage lift (%)", report["node_coverage_lift_pct"]],

        ["Base triple count", report["base_triple_count"]],
        ["Merged triple count", report["merged_triple_count"]],
        ["New triple count", report["new_triple_count"]],
        ["Triple growth (%)", report["triple_growth_pct"]],

        ["Base entity type count", report["base_entity_type_count"]],
        ["Merged entity type count", report["merged_entity_type_count"]],
        ["New entity type count", report["new_entity_type_count"]],

        ["Base relation count", report["base_relation_count"]],
        ["Merged relation count", report["merged_relation_count"]],
        ["New relation count", report["new_relation_count"]],
        ["Relation diversity lift (%)", report["relation_diversity_lift_pct"]],

        ["Base display relation count", report["base_display_relation_count"]],
        ["Merged display relation count", report["merged_display_relation_count"]],
        ["New display relation count", report["new_display_relation_count"]],

        ["Base relation entropy", report["base_relation_entropy"]],
        ["Merged relation entropy", report["merged_relation_entropy"]],
        ["Base display relation entropy", report["base_display_relation_entropy"]],
        ["Merged display relation entropy", report["merged_display_relation_entropy"]],

        ["Base 2-hop paths", report["base_2hop_paths"]],
        ["Merged 2-hop paths", report["merged_2hop_paths"]],
        ["Estimated new 2-hop paths", report["estimated_new_2hop_paths"]],

        ["New entity pair count", report["new_entity_pair_count"]],
        ["New type pair count", report["new_type_pair_count"]],

        ["Novel triple rate (%)", report["novel_triple_rate_pct"]],
        ["Novel entity-pair rate (%)", report["novel_entity_pair_rate_pct"]],
        ["Novel relation usage rate (%)", report["novel_relation_usage_rate_pct"]],
        ["Novel type-pair rate (%)", report["novel_type_pair_rate_pct"]],
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df.to_csv(
        out_dir / "table_quantitative_summary.filtered.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return df


# =====================
# Plots
# =====================
def format_count_for_bar(x: int | float) -> str:
    """
    More compact labels for very large counts.
    """
    x = float(x)
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{int(x):,}"


def plot_graph_overview(report: dict, out_dir: Path):
    labels = [report["base_name"], "Filtered Integrated KG"]

    panel_data = [
        (
            "a",
            "Node Count",
            [report["base_node_count"], report["merged_node_count"]],
            COLORS["blue"],
            COLORS["light_blue"]
        ),
        (
            "b",
            "Triplet Count",
            [report["base_triple_count"], report["merged_triple_count"]],
            COLORS["orange"],
            COLORS["light_orange"]
        ),
        (
            "c",
            "Entity Types",
            [report["base_entity_type_count"], report["merged_entity_type_count"]],
            COLORS["teal"],
            "#b2dfdb"
        ),
        (
            "d",
            "Relation Types",
            [report["base_relation_count"], report["merged_relation_count"]],
            COLORS["purple"],
            "#e1bee7"
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

    for ax, (panel, ylabel, vals, c1, c0) in zip(axes, panel_data):
        bars = ax.bar(
            labels,
            vals,
            color=[c0, c1],
            edgecolor="black",
            linewidth=0.8,
            width=0.55
        )

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.02,
                format_count_for_bar(h),
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_ylabel(ylabel)
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.18)

        ax.tick_params(axis="x", labelrotation=20)

    fig.suptitle(
        "Graph-Level Overview After Entity-Type Filtering",
        fontsize=12,
        y=1.02
    )
    fig.tight_layout()

    fig.savefig(
        out_dir / "figure_graph_overview.filtered.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        out_dir / "figure_graph_overview.filtered.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_node_type_distribution(report: dict, out_dir: Path):
    base = report["base_nodes_by_type"]
    merged = report["merged_nodes_by_type"]

    all_types = sorted(
        set(base.keys()) | set(merged.keys()),
        key=lambda t: -merged.get(t, 0)
    )

    base_vals = [base.get(t, 0) for t in all_types]
    merged_vals = [merged.get(t, 0) for t in all_types]

    x = np.arange(len(all_types))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(all_types) * 1.1), 5))

    ax.bar(
        x - width / 2,
        base_vals,
        width=width,
        label=report["base_name"],
        color=COLORS["light_blue"],
        edgecolor="black",
        linewidth=0.7
    )
    ax.bar(
        x + width / 2,
        merged_vals,
        width=width,
        label="Filtered Integrated KG",
        color=COLORS["blue"],
        edgecolor="black",
        linewidth=0.7
    )

    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=30, ha="right")
    ax.set_ylabel("Node Count")
    ax.set_title("Entity Type Distribution After Entity-Type Filtering")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        out_dir / "figure_node_type_distribution.filtered.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        out_dir / "figure_node_type_distribution.filtered.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_new_nodes_by_type(report: dict, out_dir: Path):
    """
    Optional but useful figure:
    show only newly added nodes by type.
    """
    data = report["new_nodes_by_type"]
    if not data:
        return

    types = list(data.keys())
    vals = list(data.values())

    y = np.arange(len(types))

    fig, ax = plt.subplots(figsize=(8, max(4.5, len(types) * 0.38)))
    bars = ax.barh(
        y,
        vals,
        color=COLORS["green"],
        edgecolor="black",
        linewidth=0.7,
        height=0.65
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}",
            va="center",
            ha="left",
            fontsize=9
        )

    ax.set_yticks(y)
    ax.set_yticklabels(types)
    ax.set_xlabel("New Node Count")
    ax.set_title("New Nodes by Entity Type After Entity-Type Filtering")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.15)

    fig.tight_layout()

    fig.savefig(
        out_dir / "figure_new_nodes_by_type.filtered.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        out_dir / "figure_new_nodes_by_type.filtered.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_novelty_metrics(report: dict, out_dir: Path):
    metrics = [
        ("Novel triple rate", report["novel_triple_rate_pct"]),
        ("Novel entity-pair rate", report["novel_entity_pair_rate_pct"]),
        ("Novel relation usage rate", report["novel_relation_usage_rate_pct"]),
        ("Novel type-pair rate", report["novel_type_pair_rate_pct"]),
    ]

    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [
        COLORS["blue"],
        COLORS["teal"],
        COLORS["orange"],
        COLORS["purple"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(names))

    bars = ax.barh(
        y,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        height=0.65
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}%",
            va="center",
            ha="left",
            fontsize=10
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 110)
    ax.set_title("Novelty of Newly Added Knowledge After Entity-Type Filtering")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        out_dir / "figure_novelty_metrics.filtered.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        out_dir / "figure_novelty_metrics.filtered.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter low-frequency and forced-excluded entity types "
            "and re-run KG quantitative analysis."
        )
    )

    parser.add_argument(
        "--base-csv",
        type=str,
        required=True,
        help="Path to base KG csv, e.g. kg.csv"
    )
    parser.add_argument(
        "--merged-csv",
        type=str,
        required=True,
        help="Path to merged KG csv, e.g. merged_kg.csv"
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="PrimeKG",
        help="Base KG name"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="quant_figures_filtered",
        help="Output directory"
    )
    parser.add_argument(
        "--min-nodes-per-type",
        type=int,
        default=10,
        help="Minimum unique node count required for an entity type"
    )
    parser.add_argument(
        "--min-type-ratio",
        type=float,
        default=0.0,
        help="Minimum node ratio required for an entity type"
    )
    parser.add_argument(
        "--exclude-types",
        nargs="*",
        default=["none"],
        help=(
            "Entity types to always exclude regardless of frequency. "
            "Default: none. Example: --exclude-types none unknown other"
        )
    )
    parser.add_argument(
        "--filter-base-too",
        action="store_true",
        help=(
            "Whether to apply the same entity-type filter to the base KG. "
            "Default is False. Use only if base KG also contains placeholder "
            "or low-frequency types that should be removed from both graphs."
        )
    )
    parser.add_argument(
        "--save-filtered-csv",
        action="store_true",
        help="Whether to save filtered merged kg csv"
    )

    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    print("Loading graphs...")
    base_df = read_kg_csv(args.base_csv)
    merged_df = read_kg_csv(args.merged_csv)

    forced_excluded_types = normalize_type_set(args.exclude_types)

    print("Detecting low-frequency entity types in merged graph...")
    low_frequency_types, type_count_df = find_low_frequency_types(
        merged_df,
        min_nodes_per_type=args.min_nodes_per_type,
        min_type_ratio=args.min_type_ratio
    )

    final_removed_types = add_forced_excluded_types(
        low_frequency_types,
        force_exclude_types=forced_excluded_types
    )

    print(f"Low-frequency types to remove ({len(low_frequency_types)}): {sorted(low_frequency_types)}")
    print(f"Forced excluded types ({len(forced_excluded_types)}): {sorted(forced_excluded_types)}")
    print(f"Final entity types to remove ({len(final_removed_types)}): {sorted(final_removed_types)}")

    print("Filtering merged graph...")
    filtered_merged_df = filter_graph_by_entity_types(
        merged_df,
        final_removed_types
    )

    if args.filter_base_too:
        print("Filtering base graph with the same entity-type filter...")
        filtered_base_df = filter_graph_by_entity_types(
            base_df,
            final_removed_types
        )
    else:
        filtered_base_df = base_df.copy()

    if args.save_filtered_csv:
        filtered_csv_path = out_dir / "merged_kg.filtered.csv"
        filtered_merged_df.to_csv(
            filtered_csv_path,
            index=False,
            encoding="utf-8-sig"
        )
        print(f"Saved filtered merged KG to: {filtered_csv_path.resolve()}")

        if args.filter_base_too:
            filtered_base_csv_path = out_dir / "base_kg.filtered.csv"
            filtered_base_df.to_csv(
                filtered_base_csv_path,
                index=False,
                encoding="utf-8-sig"
            )
            print(f"Saved filtered base KG to: {filtered_base_csv_path.resolve()}")

    # Save entity type counts before filtering
    type_count_df.to_csv(
        out_dir / "table_entity_type_counts_before_filtering.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Save filtering summary
    filtering_summary_df = build_filtering_summary(
        original_df=merged_df,
        filtered_df=filtered_merged_df,
        low_frequency_types=low_frequency_types,
        forced_excluded_types=forced_excluded_types,
        final_removed_types=final_removed_types
    )
    filtering_summary_df.to_csv(
        out_dir / "table_filtering_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Filtering summary:")
    print(filtering_summary_df.to_string(index=False))

    print("Recomputing quantitative report...")
    report = build_quant_report(
        filtered_base_df,
        filtered_merged_df,
        base_name=args.base_name
    )

    save_json(
        report,
        out_dir / "quant_eval_report.filtered.json"
    )

    export_summary_table(report, out_dir)
    plot_graph_overview(report, out_dir)
    plot_node_type_distribution(report, out_dir)
    plot_new_nodes_by_type(report, out_dir)
    plot_novelty_metrics(report, out_dir)

    print(f"Done. All outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()