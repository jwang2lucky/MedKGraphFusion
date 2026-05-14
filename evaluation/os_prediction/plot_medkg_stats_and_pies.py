from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FIG_BG = "#ffffff"
TEXT_DARK = "#222222"
TEXT_MID = "#555555"
EDGE_COLOR = "white"

RELATION_BASE_PALETTE = [
    "#b07aa1",  # muted purple
    "#bab0ab",  # warm gray
    "#edc948",  # muted yellow
    "#9c755f",  # brown
    "#8cd17d",  # muted green
    "#e7969c",  # muted pink
    "#4e79a7",  # blue
    "#59a14f",  # green
    "#e15759",  # red
    "#f28e2b",  # orange
    "#76b7b2",  # teal
    "#ff9da7",  # pink
    "#9d9d9d",  # gray
]

ENTITY_BASE_PALETTE = [
    "#4e79a7",  # blue
    "#59a14f",  # green
    "#e15759",  # red
    "#f28e2b",  # orange
    "#76b7b2",  # teal
    "#edc948",  # yellow
    "#b07aa1",  # purple
    "#ff9da7",  # pink
    "#9c755f",  # brown
    "#bab0ab",  # gray
    "#8cd17d",  # light green
    "#e7969c",  # muted pink
    "#9d9d9d",  # gray 2
]


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


def parse_source_field(v) -> list[str]:
    s = normalize_str(v)
    if not s:
        return []
    s = s.strip("[](){}")
    s = s.replace('"', "").replace("'", "")
    parts = re.split(r"[;,|]", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def normalize_source_name(s: str) -> str:
    s = normalize_str(s)
    if not s:
        return "Unknown"
    s_upper = s.upper()
    if s_upper == "CUSTOM":
        return "text-based"
    mapping = {
        "DRUGBANK": "DrugBank",
        "NCBI": "NCBI",
        "GO": "GO",
        "MONDO": "MONDO",
        "MONDO_GROUPED": "MONDO_grouped",
        "HPO": "HPO",
        "CTD": "CTD",
        "REACTOME": "REACTOME",
        "UBERON": "UBERON",
        "MESH": "MeSH",
        "OMIM": "OMIM",
        "UMLS": "UMLS",
    }
    return mapping.get(s_upper, s)


def serialize_counter(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def sorted_counter_df(counter: Counter, key_col: str, value_col: str) -> pd.DataFrame:
    items = sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))
    return pd.DataFrame(items, columns=[key_col, value_col])


def get_relation_counts(df: pd.DataFrame) -> Counter:
    if "display_relation" not in df.columns:
        raise ValueError("Missing column: display_relation")
    rels = [normalize_str(x) for x in df["display_relation"].tolist()]
    rels = [x for x in rels if x and x.lower() != "none"]
    return Counter(rels)


def get_entity_counts(df: pd.DataFrame) -> Counter:
    required = ["x_id", "x_type", "y_id", "y_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for entity stats: {missing}")

    node_seen = set()
    counter = Counter()
    for _, row in df.iterrows():
        for node_id_col, node_type_col in [("x_id", "x_type"), ("y_id", "y_type")]:
            node_id = normalize_str(row[node_id_col])
            node_type = normalize_str(row[node_type_col])
            if not node_id or not node_type or node_type.lower() == "none":
                continue
            key = (node_id, node_type)
            if key in node_seen:
                continue
            node_seen.add(key)
            counter[node_type] += 1
    return counter


def get_panel_a_relation_pair_counts(text_df: pd.DataFrame) -> dict[str, Counter]:
    required = ["display_relation", "x_type", "y_type"]
    missing = [c for c in required if c not in text_df.columns]
    if missing:
        raise ValueError(f"Missing columns for Panel A: {missing}")

    rel_pair = defaultdict(Counter)
    for _, row in text_df.iterrows():
        rel = normalize_str(row["display_relation"])
        x_type = normalize_str(row["x_type"])
        y_type = normalize_str(row["y_type"])
        if not rel or not x_type or not y_type:
            continue
        if x_type.lower() == "none" or y_type.lower() == "none" or rel.lower() == "none":
            continue
        pair = pretty_pair_label(x_type, y_type)
        rel_pair[rel][pair] += 1
    return dict(rel_pair)


def get_panel_b_entity_resource_counts(integrated_df: pd.DataFrame) -> tuple[Counter, dict[str, Counter]]:
    required = ["x_id", "x_type", "x_source", "y_id", "y_type", "y_source"]
    missing = [c for c in required if c not in integrated_df.columns]
    if missing:
        raise ValueError(f"Missing columns for Panel B: {missing}")

    node_resources = defaultdict(set)

    for _, row in integrated_df.iterrows():
        x_id = normalize_str(row["x_id"])
        x_type = normalize_str(row["x_type"])
        y_id = normalize_str(row["y_id"])
        y_type = normalize_str(row["y_type"])

        if x_id and x_type and x_type.lower() != "none":
            for src in parse_source_field(row.get("x_source", "")):
                node_resources[(x_id, x_type)].add(normalize_source_name(src))

        if y_id and y_type and y_type.lower() != "none":
            for src in parse_source_field(row.get("y_source", "")):
                node_resources[(y_id, y_type)].add(normalize_source_name(src))

    entity_counter = Counter()
    entity_resource_counter = defaultdict(Counter)

    for (node_id, node_type), resources in node_resources.items():
        entity_counter[node_type] += 1
        clean_resources = {r for r in resources if r and r != "Unknown"}
        if not clean_resources:
            entity_resource_counter[node_type]["Unknown"] += 1
        else:
            for r in sorted(clean_resources):
                entity_resource_counter[node_type][r] += 1

    return entity_counter, dict(entity_resource_counter)


def save_panel_a_outputs(rel_pair_counts: dict[str, Counter], out_dir: Path):
    rel_summary = Counter({rel: sum(pair_counter.values()) for rel, pair_counter in rel_pair_counts.items()})
    rel_df = sorted_counter_df(rel_summary, "relation", "count")
    rel_df.to_csv(out_dir / "panel_a_inner_relation_counts.csv", index=False, encoding="utf-8-sig")

    rows = []
    for rel, pair_counter in sorted(rel_pair_counts.items(), key=lambda x: (-sum(x[1].values()), x[0])):
        total = sum(pair_counter.values())
        for pair, cnt in sorted(pair_counter.items(), key=lambda x: (-x[1], x[0])):
            rows.append({
                "relation": rel,
                "relation_total": total,
                "entity_pair_type": pair,
                "pair_count": cnt,
            })
    pd.DataFrame(rows).to_csv(out_dir / "panel_a_relation_entity_pair_counts.csv", index=False, encoding="utf-8-sig")


def save_panel_b_outputs(entity_counter: Counter, entity_resource_counter: dict[str, Counter], out_dir: Path):
    entity_df = sorted_counter_df(entity_counter, "entity_type", "node_count")
    entity_df.to_csv(out_dir / "panel_b_inner_entity_counts.csv", index=False, encoding="utf-8-sig")

    rows = []
    for etype, cnt in sorted(entity_counter.items(), key=lambda x: (-x[1], x[0])):
        resource_counter = entity_resource_counter.get(etype, Counter())
        for resource, rcnt in sorted(resource_counter.items(), key=lambda x: (-x[1], x[0])):
            rows.append({
                "entity_type": etype,
                "entity_total": cnt,
                "resource": resource,
                "resource_count": rcnt,
            })
    pd.DataFrame(rows).to_csv(out_dir / "panel_b_entity_resource_counts.csv", index=False, encoding="utf-8-sig")


def choose_top_plus_other(counter: Counter, top_n: int = 8) -> tuple[list[str], list[int]]:
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if len(items) <= top_n:
        labels = [k for k, _ in items]
        values = [int(v) for _, v in items]
        return labels, values

    top_items = items[:top_n]
    other_sum = sum(v for _, v in items[top_n:])
    labels = [k for k, _ in top_items] + ["Other"]
    values = [int(v) for _, v in top_items] + [int(other_sum)]
    return labels, values


def get_color_map(labels: list[str], palette: list[str]) -> dict[str, str]:
    unique_non_other = [lab for lab in labels if lab != "Other"]
    cmap = {}
    for i, lab in enumerate(unique_non_other):
        cmap[lab] = palette[i % len(palette)]
    cmap["Other"] = "#d0d0d0"
    return cmap


def pie_autopct(values: list[int], min_pct: float = 3.0):
    total = float(sum(values))
    def _fmt(pct: float) -> str:
        if pct < min_pct:
            return ""
        value = round(pct * total / 100.0)
        return f"{pct:.1f}%\n({int(value):,})"
    return _fmt


def plot_pie(ax, counter: Counter, title: str, palette: list[str], top_n: int = 8):
    labels, values = choose_top_plus_other(counter, top_n=top_n)
    color_map = get_color_map(labels, palette)
    colors = [color_map[l] for l in labels]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor=EDGE_COLOR, linewidth=1.0),
        labeldistance=1.08,
        pctdistance=0.68,
        autopct=pie_autopct(values, min_pct=4.0),
        textprops=dict(color=TEXT_DARK, fontsize=10),
    )
    for t in autotexts:
        t.set_color("#1f1f1f")
        t.set_fontsize(9)
    ax.set_title(title, fontsize=14, pad=12)
    ax.axis("equal")


def make_2x2_pie_figure(
    integrated_relation: Counter,
    integrated_entity: Counter,
    text_relation: Counter,
    text_entity: Counter,
    out_dir: Path,
    basename: str = "figure_simple_pies_medkg",
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=FIG_BG)

    plot_pie(axes[0, 0], text_relation, "Text-based MedKG: relation distribution", RELATION_BASE_PALETTE, top_n=8)
    plot_pie(axes[0, 1], text_entity, "Text-based MedKG: entity distribution", ENTITY_BASE_PALETTE, top_n=8)
    plot_pie(axes[1, 0], integrated_relation, "Integrated MedKG: relation distribution", RELATION_BASE_PALETTE, top_n=8)
    plot_pie(axes[1, 1], integrated_entity, "Integrated MedKG: entity distribution", ENTITY_BASE_PALETTE, top_n=8)

    fig.suptitle("Simple pie-chart overview of relation and entity composition", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_path = out_dir / f"{basename}.png"
    pdf_path = out_dir / f"{basename}.pdf"
    svg_path = out_dir / f"{basename}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=FIG_BG)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=FIG_BG)
    fig.savefig(svg_path, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)

    return png_path, pdf_path, svg_path


def save_basic_stats(
    text_relation: Counter,
    text_entity: Counter,
    integrated_relation: Counter,
    integrated_entity: Counter,
    out_dir: Path,
):
    sorted_counter_df(text_relation, "relation", "count").to_csv(
        out_dir / "text_medkg_relation_counts.csv", index=False, encoding="utf-8-sig"
    )
    sorted_counter_df(text_entity, "entity_type", "count").to_csv(
        out_dir / "text_medkg_entity_counts.csv", index=False, encoding="utf-8-sig"
    )
    sorted_counter_df(integrated_relation, "relation", "count").to_csv(
        out_dir / "integrated_medkg_relation_counts.csv", index=False, encoding="utf-8-sig"
    )
    sorted_counter_df(integrated_entity, "entity_type", "count").to_csv(
        out_dir / "integrated_medkg_entity_counts.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "text_medkg_relation_counts": serialize_counter(text_relation),
        "text_medkg_entity_counts": serialize_counter(text_entity),
        "integrated_medkg_relation_counts": serialize_counter(integrated_relation),
        "integrated_medkg_entity_counts": serialize_counter(integrated_entity),
    }
    with open(out_dir / "medkg_basic_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_console_summary(
    text_relation: Counter,
    text_entity: Counter,
    panel_a_rel_pairs: dict[str, Counter],
    panel_b_entity_counter: Counter,
    panel_b_entity_resource_counter: dict[str, Counter],
):
    print("\n=== kg_merged_new_triples.csv: relation distribution ===")
    for k, v in sorted(text_relation.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}\t{v}")

    print("\n=== kg_merged_new_triples.csv: entity distribution ===")
    for k, v in sorted(text_entity.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}\t{v}")

    print("\n=== Panel A: relation -> entity-pair counts ===")
    for rel, pair_counter in sorted(panel_a_rel_pairs.items(), key=lambda x: (-sum(x[1].values()), x[0])):
        print(f"\n[{rel}] total={sum(pair_counter.values())}")
        for pair, cnt in sorted(pair_counter.items(), key=lambda x: (-x[1], x[0]))[:15]:
            print(f"  {pair}\t{cnt}")

    print("\n=== Panel B: entity type -> resource counts ===")
    for etype, cnt in sorted(panel_b_entity_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"\n[{etype}] total_nodes={cnt}")
        for resource, rcnt in sorted(panel_b_entity_resource_counter.get(etype, Counter()).items(), key=lambda x: (-x[1], x[0])):
            print(f"  {resource}\t{rcnt}")


def main():
    parser = argparse.ArgumentParser(
        description="Output MedKG relation/entity stats and generate four simple pie charts."
    )
    parser.add_argument("--integrated-csv", type=str, required=True, help="Path to merged_kg.filtered.csv")
    parser.add_argument("--text-csv", type=str, required=True, help="Path to kg_merged_new_triples.csv")
    parser.add_argument("--out-dir", type=str, default="figures_pies_and_stats", help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    integrated_df = read_csv(args.integrated_csv)
    text_df = read_csv(args.text_csv)

    text_relation = get_relation_counts(text_df)
    text_entity = get_entity_counts(text_df)
    integrated_relation = get_relation_counts(integrated_df)
    integrated_entity = get_entity_counts(integrated_df)

    panel_a_rel_pairs = get_panel_a_relation_pair_counts(text_df)
    panel_b_entity_counter, panel_b_entity_resource_counter = get_panel_b_entity_resource_counts(integrated_df)

    save_basic_stats(
        text_relation=text_relation,
        text_entity=text_entity,
        integrated_relation=integrated_relation,
        integrated_entity=integrated_entity,
        out_dir=out_dir,
    )
    save_panel_a_outputs(panel_a_rel_pairs, out_dir)
    save_panel_b_outputs(panel_b_entity_counter, panel_b_entity_resource_counter, out_dir)

    png_path, pdf_path, svg_path = make_2x2_pie_figure(
        integrated_relation=integrated_relation,
        integrated_entity=integrated_entity,
        text_relation=text_relation,
        text_entity=text_entity,
        out_dir=out_dir,
    )

    print_console_summary(
        text_relation=text_relation,
        text_entity=text_entity,
        panel_a_rel_pairs=panel_a_rel_pairs,
        panel_b_entity_counter=panel_b_entity_counter,
        panel_b_entity_resource_counter=panel_b_entity_resource_counter,
    )

    print("\nSaved outputs:")
    print(f"- {out_dir / 'text_medkg_relation_counts.csv'}")
    print(f"- {out_dir / 'text_medkg_entity_counts.csv'}")
    print(f"- {out_dir / 'integrated_medkg_relation_counts.csv'}")
    print(f"- {out_dir / 'integrated_medkg_entity_counts.csv'}")
    print(f"- {out_dir / 'panel_a_inner_relation_counts.csv'}")
    print(f"- {out_dir / 'panel_a_relation_entity_pair_counts.csv'}")
    print(f"- {out_dir / 'panel_b_inner_entity_counts.csv'}")
    print(f"- {out_dir / 'panel_b_entity_resource_counts.csv'}")
    print(f"- {out_dir / 'medkg_basic_stats.json'}")
    print(f"- {png_path}")
    print(f"- {pdf_path}")
    print(f"- {svg_path}")


if __name__ == "__main__":
    main()
