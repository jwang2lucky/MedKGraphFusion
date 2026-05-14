from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge


# =====================
# Global style
# =====================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FIG_BG = "#ffffff"
EDGE_COLOR = "white"
TEXT_DARK = "#222222"
TEXT_MID = "#555555"


# =====================
# Fixed configuration
# =====================
PANEL_A_RELATIONS = [
    "associated with",
    "used for",
    "part of",
    "phenotype present",
    "parent-child",
    "indication",
    "causes",
    "affects expression",
    "located in",
    "expressed in",
]

PANEL_B_ENTITY_TYPES = [
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
]

RELATION_COLOR_MAP = {
    "associated with": "#b07aa1",
    "used for": "#bab0ab",
    "part of": "#edc948",
    "phenotype present": "#9c755f",
    "parent-child": "#8cd17d",
    "indication": "#e7969c",
    "causes": "#4e79a7",
    "affects expression": "#59a14f",
    "located in": "#76b7b2",
    "expressed in": "#f28e2b",
    "Other": "#c7c7c7",
}

ENTITY_COLOR_MAP = {
    "biological_process": "#4e79a7",
    "gene/protein": "#59a14f",
    "disease": "#e15759",
    "effect/phenotype": "#f28e2b",
    "anatomy": "#76b7b2",
    "molecular_function": "#edc948",
    "drug": "#b07aa1",
    "cellular_component": "#ff9da7",
    "exposure": "#9c755f",
    "pathway": "#bab0ab",
    "procedure": "#8cd17d",
    "Other": "#c7c7c7",
}


# =====================
# Utilities
# =====================
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(normalize_str)
    return df


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_int(n: int | float) -> str:
    return f"{int(round(n)):,}"


def normalize_entity_type(s: str) -> str:
    s = normalize_str(s)
    if not s:
        return ""

    s = s.strip().lower()
    s = s.replace(" ", "_")

    mapping = {
        "gene": "gene/protein",
        "protein": "gene/protein",
        "gene_protein": "gene/protein",
        "gene/protein": "gene/protein",
        "effect": "effect/phenotype",
        "phenotype": "effect/phenotype",
        "effect_phenotype": "effect/phenotype",
        "effect/phenotype": "effect/phenotype",
        "biological_process": "biological_process",
        "molecular_function": "molecular_function",
        "cellular_component": "cellular_component",
    }
    return mapping.get(s, s)


def pretty_type_label(s: str) -> str:
    mapping = {
        "biological_process": "Biological process",
        "gene/protein": "Gene/protein",
        "disease": "Disease",
        "effect/phenotype": "Effect/phenotype",
        "anatomy": "Anatomy",
        "molecular_function": "Molecular function",
        "drug": "Drug",
        "cellular_component": "Cellular component",
        "exposure": "Exposure",
        "pathway": "Pathway",
        "procedure": "Procedure",
        "none": "None",
    }
    s = normalize_str(s)
    return mapping.get(s, s.replace("_", " "))


def pretty_pair_label(x_type: str, y_type: str) -> str:
    return f"{pretty_type_label(x_type)} -> {pretty_type_label(y_type)}"


def clean_relation(s: str) -> str:
    return normalize_str(s)


def mix_with_white(color: str, strength: float) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    mixed = rgb * (1 - strength) + white * strength
    return mcolors.to_hex(mixed)


def radial_text(
    ax,
    angle_deg: float,
    radius: float,
    text: str,
    fontsize: float = 9,
    fontweight: str = "normal",
    color: str = TEXT_DARK,
):
    angle_rad = np.deg2rad(angle_deg)
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)

    rotation = angle_deg
    ha = "left"

    if 90 < angle_deg < 270:
        rotation = angle_deg + 180
        ha = "right"

    ax.text(
        x,
        y,
        text,
        rotation=rotation,
        rotation_mode="anchor",
        ha=ha,
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
    )


def get_entity_type_order(counter: Counter | pd.Series | dict) -> list[str]:
    if isinstance(counter, pd.Series):
        keys = set(counter.index.tolist())
    elif isinstance(counter, dict):
        keys = set(counter.keys())
    else:
        keys = set(counter.keys())

    ordered = [t for t in PANEL_B_ENTITY_TYPES if t in keys]
    extras = sorted([t for t in keys if t not in set(PANEL_B_ENTITY_TYPES) and t != ""])
    ordered.extend(extras)
    return ordered


def add_inner_labels(ax, wedges, labels, values, radius, min_pct_total=4.0, fontsize=10):
    total = float(sum(values))
    for w, lab, v in zip(wedges, labels, values):
        pct = (v / total * 100) if total > 0 else 0
        if pct < min_pct_total:
            continue
        ang = (w.theta1 + w.theta2) / 2
        radial_text(ax, ang, radius, lab, fontsize=fontsize, fontweight="medium")


def add_selected_outer_labels(
    ax,
    wedges,
    labels,
    values,
    parent_ids,
    radius,
    min_pct_total=2.5,
    top_k_per_parent=1,
    fontsize=8,
):
    total = float(sum(values))
    by_parent = defaultdict(list)

    for i, (p, v) in enumerate(zip(parent_ids, values)):
        by_parent[p].append((i, v))

    show = [False] * len(values)

    for p, arr in by_parent.items():
        arr = sorted(arr, key=lambda x: x[1], reverse=True)
        for i, _ in arr[:top_k_per_parent]:
            show[i] = True
        for i, v in arr:
            if total > 0 and (v / total * 100) >= min_pct_total:
                show[i] = True

    for i, (w, lab) in enumerate(zip(wedges, labels)):
        if not show[i]:
            continue
        ang = (w.theta1 + w.theta2) / 2
        radial_text(ax, ang, radius, lab, fontsize=fontsize)


# =====================
# Panel A logic
# =====================
def choose_panel_a_pairs(
    rel_df: pd.DataFrame,
    relation_name: str,
    top_k_default: int = 6,
    min_pct: float = 0.05,
) -> list[tuple[str, int]]:
    pair_counter = (
        rel_df.groupby("entity_pair_type")
        .size()
        .sort_values(ascending=False)
    )

    items = list(pair_counter.items())
    rel_total = int(pair_counter.sum())

    if relation_name == "parent-child":
        if len(items) <= 10:
            return items

    keep = []
    other_sum = 0

    for i, (pair, cnt) in enumerate(items):
        pct = cnt / rel_total if rel_total > 0 else 0.0
        if i < top_k_default or pct >= min_pct:
            keep.append((pair, int(cnt)))
        else:
            other_sum += int(cnt)

    if other_sum > 0:
        keep.append(("Other", int(other_sum)))

    return keep


def build_panel_a(text_df: pd.DataFrame, out_dir: Path):
    req = ["display_relation", "x_type", "y_type"]
    missing = [c for c in req if c not in text_df.columns]
    if missing:
        raise ValueError(f"text-csv missing columns: {missing}")

    df = text_df.copy()
    df["display_relation"] = df["display_relation"].map(clean_relation)
    df["x_type"] = df["x_type"].map(normalize_entity_type)
    df["y_type"] = df["y_type"].map(normalize_entity_type)

    df = df[
        (df["x_type"] != "") &
        (df["y_type"] != "") &
        (df["x_type"].str.lower() != "none") &
        (df["y_type"].str.lower() != "none")
    ].copy()

    df["relation_bucket"] = df["display_relation"].apply(
        lambda x: x if x in PANEL_A_RELATIONS else "Other"
    )

    df["entity_pair_type"] = df.apply(
        lambda r: pretty_pair_label(r["x_type"], r["y_type"]),
        axis=1
    )

    inner_counter = (
        df.groupby("relation_bucket")
        .size()
        .reindex(PANEL_A_RELATIONS + ["Other"], fill_value=0)
    )

    inner_labels = []
    inner_values = []
    inner_colors = []

    outer_labels = []
    outer_values = []
    outer_colors = []
    outer_parent_ids = []

    panel_a_inner_rows = []
    panel_a_outer_rows = []

    for rel in PANEL_A_RELATIONS + ["Other"]:
        rel_total = int(inner_counter.loc[rel])
        if rel_total <= 0:
            continue

        parent_idx = len(inner_labels)

        inner_labels.append(rel)
        inner_values.append(rel_total)
        inner_colors.append(RELATION_COLOR_MAP.get(rel, RELATION_COLOR_MAP["Other"]))

        panel_a_inner_rows.append({
            "relation": rel,
            "relation_total": rel_total
        })

        rel_df = df[df["relation_bucket"] == rel].copy()
        kept_pairs = choose_panel_a_pairs(rel_df, rel)

        n = max(len(kept_pairs), 1)
        for j, (pair_label, cnt) in enumerate(kept_pairs):
            outer_labels.append(pair_label)
            outer_values.append(int(cnt))
            outer_parent_ids.append(parent_idx)

            strength = 0.12 + 0.48 * (j / max(n - 1, 1))
            outer_colors.append(
                mix_with_white(
                    RELATION_COLOR_MAP.get(rel, RELATION_COLOR_MAP["Other"]),
                    strength
                )
            )

            panel_a_outer_rows.append({
                "relation": rel,
                "relation_total": rel_total,
                "entity_pair_type": pair_label,
                "pair_count": int(cnt)
            })

    pd.DataFrame(panel_a_inner_rows).to_csv(
        out_dir / "panel_a_inner_relation_counts.final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(panel_a_outer_rows).to_csv(
        out_dir / "panel_a_relation_entity_pair_counts.final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "inner_labels": inner_labels,
        "inner_values": inner_values,
        "inner_colors": inner_colors,
        "outer_labels": outer_labels,
        "outer_values": outer_values,
        "outer_colors": outer_colors,
        "outer_parent_ids": outer_parent_ids,
        "center_title": "Text-based\nMedKG",
        "center_subtitle": (
            f"{fmt_int(len(df))} triples\n"
            "Inner: selected relations\n"
            "Outer: entity-type pairs"
        ),
        "panel_type": "panel_a",
    }


# =====================
# Panel B2 logic
# =====================
def build_panel_b_entity_novelty_from_summary(summary: dict, out_dir: Path):
    if "merged_nodes_by_type" not in summary:
        raise ValueError("summary missing key: merged_nodes_by_type")
    if "new_nodes_by_type" not in summary:
        raise ValueError("summary missing key: new_nodes_by_type")

    merged_nodes = {
        normalize_entity_type(k): int(v)
        for k, v in summary["merged_nodes_by_type"].items()
    }

    new_nodes = {
        normalize_entity_type(k): int(v)
        for k, v in summary["new_nodes_by_type"].items()
    }

    entity_counter = Counter(merged_nodes)
    entity_order = get_entity_type_order(entity_counter)

    inner_labels = []
    inner_values = []
    inner_colors = []
    outer_by_relation = []

    panel_b_inner_rows = []
    panel_b_outer_rows = []

    for ent_type in entity_order:
        merged_total = int(merged_nodes.get(ent_type, 0))
        if merged_total <= 0:
            continue

        new_total = int(new_nodes.get(ent_type, 0))
        if new_total < 0:
            new_total = 0
        if new_total > merged_total:
            new_total = merged_total

        base_existing = merged_total - new_total

        label = pretty_type_label(ent_type)

        inner_labels.append(label)
        inner_values.append(merged_total)
        inner_colors.append(ENTITY_COLOR_MAP.get(ent_type, ENTITY_COLOR_MAP["Other"]))

        text_added_fraction = new_total / merged_total if merged_total > 0 else 0.0

        panel_b_inner_rows.append({
            "entity_type": ent_type,
            "entity_type_label": label,
            "merged_node_total": merged_total,
            "base_existing_nodes": base_existing,
            "text_added_nodes": new_total,
            "text_added_fraction": text_added_fraction,
        })

        items = []

        if base_existing > 0:
            items.append({
                "label": "Base-existing",
                "value": int(base_existing),
                "color": mix_with_white(
                    ENTITY_COLOR_MAP.get(ent_type, ENTITY_COLOR_MAP["Other"]),
                    0.60
                ),
            })

        if new_total > 0:
            items.append({
                "label": "Text-added",
                "value": int(new_total),
                "color": ENTITY_COLOR_MAP.get(ent_type, ENTITY_COLOR_MAP["Other"]),
            })

        for item in items:
            panel_b_outer_rows.append({
                "entity_type": ent_type,
                "entity_type_label": label,
                "merged_node_total": merged_total,
                "novelty_group": item["label"],
                "node_count": int(item["value"]),
            })

        outer_by_relation.append({
            "relation": label,
            "entity_type": ent_type,
            "relation_total": merged_total,
            "items": items,
        })

    pd.DataFrame(panel_b_inner_rows).to_csv(
        out_dir / "panel_b2_entity_novelty_inner_counts.final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(panel_b_outer_rows).to_csv(
        out_dir / "panel_b2_entity_novelty_outer_counts.final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "inner_labels": inner_labels,
        "inner_values": inner_values,
        "inner_colors": inner_colors,
        "outer_by_relation": outer_by_relation,
        "center_title": "Integrated\nMedKG",
        "center_subtitle": (
            f"{fmt_int(sum(inner_values))} merged nodes\n"
            "Inner: entity types\n"
            "Outer: base-existing vs text-added"
        ),
        "panel_type": "panel_b_entity_novelty",
    }


# =====================
# Drawing
# =====================
def draw_two_ring_donut(
    ax,
    data: dict,
    labeled: bool = True,
    panel_title: str = "",
    panel_subtitle: str = "",
):
    ax.set_aspect("equal")
    ax.set_facecolor(FIG_BG)

    startangle = 90
    counterclock = False

    inner_radius = 1.02
    inner_width = 0.34
    outer_radius = 1.38
    outer_width = 0.28

    inner_wedges, _ = ax.pie(
        data["inner_values"],
        radius=inner_radius,
        startangle=startangle,
        counterclock=counterclock,
        labels=None,
        colors=data["inner_colors"],
        wedgeprops=dict(width=inner_width, edgecolor=EDGE_COLOR, linewidth=1.4),
    )

    outer_wedges, _ = ax.pie(
        data["outer_values"],
        radius=outer_radius,
        startangle=startangle,
        counterclock=counterclock,
        labels=None,
        colors=data["outer_colors"],
        wedgeprops=dict(width=outer_width, edgecolor=EDGE_COLOR, linewidth=1.0),
    )

    if labeled:
        ax.text(
            0, 0.08, data["center_title"],
            ha="center", va="center",
            fontsize=18, fontweight="bold", color=TEXT_DARK,
        )

        ax.text(
            0, -0.14, data["center_subtitle"],
            ha="center", va="center",
            fontsize=10.5, color=TEXT_MID, linespacing=1.25,
        )

        add_inner_labels(
            ax=ax,
            wedges=inner_wedges,
            labels=data["inner_labels"],
            values=data["inner_values"],
            radius=inner_radius - inner_width / 2,
            min_pct_total=4.0,
            fontsize=10,
        )

        add_selected_outer_labels(
            ax=ax,
            wedges=outer_wedges,
            labels=data["outer_labels"],
            values=data["outer_values"],
            parent_ids=data["outer_parent_ids"],
            radius=outer_radius + 0.09,
            min_pct_total=2.5,
            top_k_per_parent=1,
            fontsize=8,
        )

        ax.text(
            0, 1.63, panel_title,
            ha="center", va="bottom",
            fontsize=18, fontweight="bold", color="#111111",
        )

        ax.text(
            0, 1.49, panel_subtitle,
            ha="center", va="bottom",
            fontsize=10.5, color=TEXT_MID,
        )

    ax.set_xlim(-1.73, 1.73)
    ax.set_ylim(-1.62, 1.72)


def draw_panel_b_donut(
    ax,
    data: dict,
    labeled: bool = True,
    panel_title: str = "",
    panel_subtitle: str = "",
):
    ax.set_aspect("equal")
    ax.set_facecolor(FIG_BG)

    startangle = 90
    counterclock = False

    inner_radius = 1.02
    inner_width = 0.34
    outer_radius = 1.38
    outer_width = 0.28

    inner_wedges, _ = ax.pie(
        data["inner_values"],
        radius=inner_radius,
        startangle=startangle,
        counterclock=counterclock,
        labels=None,
        colors=data["inner_colors"],
        wedgeprops=dict(width=inner_width, edgecolor=EDGE_COLOR, linewidth=1.4),
    )

    outer_label_records = []

    for inner_wedge, rel_block in zip(inner_wedges, data["outer_by_relation"]):
        theta1 = inner_wedge.theta1
        theta2 = inner_wedge.theta2
        items = rel_block["items"]
        total = sum(item["value"] for item in items)

        if total <= 0:
            continue

        current = theta1
        span = theta2 - theta1

        for item in items:
            frac = item["value"] / total
            delta = span * frac
            next_theta = current + delta

            wedge = Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=current,
                theta2=next_theta,
                width=outer_width,
                facecolor=item["color"],
                edgecolor=EDGE_COLOR,
                linewidth=1.0,
            )
            ax.add_patch(wedge)

            outer_label_records.append({
                "theta1": current,
                "theta2": next_theta,
                "label": item["label"],
                "value": item["value"],
                "relation": rel_block["relation"],
            })

            current = next_theta

    if labeled:
        ax.text(
            0, 0.08, data["center_title"],
            ha="center", va="center",
            fontsize=18, fontweight="bold", color=TEXT_DARK,
        )

        ax.text(
            0, -0.14, data["center_subtitle"],
            ha="center", va="center",
            fontsize=10.5, color=TEXT_MID, linespacing=1.25,
        )

        add_inner_labels(
            ax=ax,
            wedges=inner_wedges,
            labels=data["inner_labels"],
            values=data["inner_values"],
            radius=inner_radius - inner_width / 2,
            min_pct_total=4.0,
            fontsize=10,
        )

        total_outer = sum(rec["value"] for rec in outer_label_records)
        by_relation = defaultdict(list)

        for i, rec in enumerate(outer_label_records):
            by_relation[rec["relation"]].append((i, rec["value"], rec["label"]))

        show = [False] * len(outer_label_records)

        for parent, arr in by_relation.items():
            arr = sorted(arr, key=lambda x: x[1], reverse=True)
            for idx, v, lab in arr:
                if total_outer > 0 and (v / total_outer * 100) >= 0.6:
                    show[idx] = True
            if arr and not any(show[idx] for idx, _, _ in arr):
                show[arr[0][0]] = True

        for i, rec in enumerate(outer_label_records):
            if not show[i]:
                continue
            ang = (rec["theta1"] + rec["theta2"]) / 2
            radial_text(
                ax,
                ang,
                outer_radius + 0.09,
                rec["label"],
                fontsize=8,
            )

        ax.text(
            0, 1.63, panel_title,
            ha="center", va="bottom",
            fontsize=18, fontweight="bold", color="#111111",
        )

        ax.text(
            0, 1.49, panel_subtitle,
            ha="center", va="bottom",
            fontsize=10.5, color=TEXT_MID,
        )

    ax.set_xlim(-1.73, 1.73)
    ax.set_ylim(-1.62, 1.72)


# =====================
# Plot wrapper
# =====================
def build_panel_b_entity_novelty(summary: dict, out_dir: Path):
    return build_panel_b_entity_novelty_from_summary(summary, out_dir)


def plot_figure(
    text_df: pd.DataFrame,
    summary: dict,
    out_dir: Path,
    labeled: bool = True,
    suffix: str = "labeled",
):
    panel_a = build_panel_a(text_df, out_dir)
    panel_b = build_panel_b_entity_novelty(summary, out_dir)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.2),
        facecolor=FIG_BG,
        constrained_layout=True,
    )

    draw_two_ring_donut(
        ax=axes[0],
        data=panel_a,
        labeled=labeled,
        panel_title="a  Text-based MedKG: relation -> entity-pair composition",
        panel_subtitle="Inner ring: selected relations; outer ring: entity-type pairs.",
    )

    draw_panel_b_donut(
        ax=axes[1],
        data=panel_b,
        labeled=labeled,
        panel_title="b  Integrated MedKG: entity type -> novelty composition",
        panel_subtitle="Inner ring: merged entity types; outer ring: base-existing vs text-added.",
    )

    for ax in axes:
        ax.axis("off")

    fig.savefig(
        out_dir / f"overview_graph.entity_novelty.{suffix}.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir / f"overview_graph.entity_novelty.{suffix}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================
# CLI
# =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw overview graph using Panel A and Panel B2 "
            "(entity type -> base-existing vs text-added). "
            "Automatically output both labeled and no-label versions."
        )
    )

    parser.add_argument(
        "--text-csv",
        required=True,
        help="Text-based triples CSV. Required columns: display_relation, x_type, y_type.",
    )

    parser.add_argument(
        "--summary-json",
        required=True,
        help="Summary JSON with merged_nodes_by_type and new_nodes_by_type.",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = ensure_dir(args.out_dir)
    text_df = read_csv(args.text_csv)
    summary = load_json(args.summary_json)

    # 输出带标签版本
    plot_figure(
        text_df=text_df,
        summary=summary,
        out_dir=out_dir,
        labeled=True,
        suffix="labeled",
    )

    # 输出无标签版本
    plot_figure(
        text_df=text_df,
        summary=summary,
        out_dir=out_dir,
        labeled=False,
        suffix="nolabel",
    )

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()