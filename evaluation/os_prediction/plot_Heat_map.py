import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

BASE_CSV = "/mnt/gpu04_data/wangjie/KGC/data/kg/kg.csv"
TEXT_CSV = "../kg_merged_new_triples.csv"

OUTPUT_ALL_TEXT_PAIR_CSV = "./figures_donut_0422/all_text_type_pair_links_with_base_count.csv"
OUTPUT_NEW_PAIR_CSV = "./figures_donut_0422/new_type_pair_links.csv"

OUTPUT_PNG = "./figures_donut_0422/panel3_new_typepair_heatmap.png"
OUTPUT_PDF = "./figures_donut_0422/panel3_new_typepair_heatmap.pdf"

# 修改成你实际的列名
SOURCE_TYPE_COL = "x_type"
TARGET_TYPE_COL = "y_type"

# 如果你想保留方向，设 True。
# 如果只想比较两个 type 是否连接过，不管方向，设 False。
DIRECTED = False

# 主图只展示 text count >= 10 的新 type-pair
MIN_TEXT_TRIPLE_COUNT = 0

# 是否使用 log10(count + 1) 作为颜色
USE_LOG10 = True

# 是否在格子里标注原始 count
ANNOTATE_COUNTS =False
ANNOTATE_MIN_COUNT = 0

# 图像参数
FIG_WIDTH = 7.5
FIG_HEIGHT = 6.5
DPI = 600
CMAP = "YlOrRd"


# ============================================================
# Core entity type schema
# ============================================================

CORE_ENTITY_TYPES = [
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


# ============================================================
# Helper functions
# ============================================================

def normalize_type(x):
    """
    Normalize entity type strings.
    根据你的实际数据可以继续扩展 mapping。
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    x = x.lower()
    x = x.replace(" ", "_")

    mapping = {
        # gene / protein
        "gene": "gene/protein",
        "protein": "gene/protein",
        "gene_protein": "gene/protein",
        "genes/proteins": "gene/protein",
        "gene/proteins": "gene/protein",
        "genes": "gene/protein",
        "proteins": "gene/protein",

        # phenotype / effect
        "phenotype": "effect/phenotype",
        "phenotypes": "effect/phenotype",
        "effect": "effect/phenotype",
        "effects": "effect/phenotype",
        "effect_phenotype": "effect/phenotype",
        "effect/phenotypes": "effect/phenotype",

        # biological process
        "biological process": "biological_process",
        "biological_processes": "biological_process",
        "bp": "biological_process",

        # molecular function
        "molecular function": "molecular_function",
        "molecular_functions": "molecular_function",
        "mf": "molecular_function",

        # cellular component
        "cellular component": "cellular_component",
        "cellular_components": "cellular_component",
        "cc": "cellular_component",

        # disease
        "diseases": "disease",

        # anatomy
        "anatomical_entity": "anatomy",
        "anatomical_structure": "anatomy",

        # drug
        "chemical": "drug",
        "compound": "drug",
        "drug/compound": "drug",
        "drugs": "drug",

        # exposure
        "environmental_exposure": "exposure",
        "environment": "exposure",

        # procedure
        "medical_procedure": "procedure",
        "procedures": "procedure",

        # none-like labels
        "": "none",
        "na": "none",
        "nan": "none",
        "null": "none",
        "unknown": "none",
        "other": "none",
    }

    return mapping.get(x, x)


def load_and_prepare(csv_path, source_col, target_col):
    df = pd.read_csv(csv_path)

    missing = [c for c in [source_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[source_col] = df[source_col].apply(normalize_type)
    df[target_col] = df[target_col].apply(normalize_type)

    df = df.dropna(subset=[source_col, target_col])

    if not DIRECTED:
        tmp = df[[source_col, target_col]].apply(
            lambda r: sorted([r[source_col], r[target_col]]),
            axis=1,
            result_type="expand"
        )
        df[source_col] = tmp[0]
        df[target_col] = tmp[1]

    return df


def filter_core_types(df, source_col, target_col):
    """
    只保留 11 个核心类型。
    自动去除 none 和其他非核心类型。
    """
    core_set = set(CORE_ENTITY_TYPES)

    df = df[
        df[source_col].isin(core_set) &
        df[target_col].isin(core_set)
    ].copy()

    return df


def count_type_pairs(df, source_col, target_col, count_name):
    out = (
        df.groupby([source_col, target_col])
        .size()
        .reset_index(name=count_name)
    )
    return out


# ============================================================
# Main
# ============================================================

def main():
    # ----------------------------
    # 1. Load
    # ----------------------------
    base_df = load_and_prepare(BASE_CSV, SOURCE_TYPE_COL, TARGET_TYPE_COL)
    text_df = load_and_prepare(TEXT_CSV, SOURCE_TYPE_COL, TARGET_TYPE_COL)

    print("Before core-type filtering:")
    print(f"  Base triples: {len(base_df):,}")
    print(f"  Text triples: {len(text_df):,}")
    print(f"  Base unique types: {sorted(set(base_df[SOURCE_TYPE_COL]).union(set(base_df[TARGET_TYPE_COL])))}")
    print(f"  Text unique types: {sorted(set(text_df[SOURCE_TYPE_COL]).union(set(text_df[TARGET_TYPE_COL])))}")

    # ----------------------------
    # 2. Keep only core 11 entity types
    # ----------------------------
    base_df = filter_core_types(base_df, SOURCE_TYPE_COL, TARGET_TYPE_COL)
    text_df = filter_core_types(text_df, SOURCE_TYPE_COL, TARGET_TYPE_COL)

    print("\nAfter core-type filtering:")
    print(f"  Base triples: {len(base_df):,}")
    print(f"  Text triples: {len(text_df):,}")

    # ----------------------------
    # 3. Count type-pairs
    # ----------------------------
    base_counts = count_type_pairs(
        base_df,
        SOURCE_TYPE_COL,
        TARGET_TYPE_COL,
        "base_triple_count"
    )

    text_counts = count_type_pairs(
        text_df,
        SOURCE_TYPE_COL,
        TARGET_TYPE_COL,
        "text_triple_count"
    )

    print(f"\nBase type-pairs after filtering: {len(base_counts):,}")
    print(f"Text type-pairs after filtering: {len(text_counts):,}")

    # ----------------------------
    # 4. Merge text pairs with base counts
    # ----------------------------
    all_text_pairs = text_counts.merge(
        base_counts,
        on=[SOURCE_TYPE_COL, TARGET_TYPE_COL],
        how="left"
    )

    all_text_pairs["base_triple_count"] = (
        all_text_pairs["base_triple_count"]
        .fillna(0)
        .astype(int)
    )

    all_text_pairs = all_text_pairs.sort_values(
        ["base_triple_count", "text_triple_count"],
        ascending=[True, False]
    ).reset_index(drop=True)

    all_text_pairs.to_csv(OUTPUT_ALL_TEXT_PAIR_CSV, index=False)
    print(f"\nSaved diagnostic file: {OUTPUT_ALL_TEXT_PAIR_CSV}")

    # 检查：如果 all_text_pairs 里 base count 全为 0，说明 base/text type-pair 很可能没有匹配上
    matched_n = (all_text_pairs["base_triple_count"] > 0).sum()
    print(f"Text type-pairs matched in base: {matched_n} / {len(all_text_pairs)}")

    if matched_n == 0:
        print("\nWARNING:")
        print("  No text type-pairs matched base type-pairs.")
        print("  This may be expected only if base and text have completely distinct schemas.")
        print("  Otherwise, check:")
        print("    1. whether SOURCE_TYPE_COL and TARGET_TYPE_COL are correct for both files;")
        print("    2. whether type names are normalized consistently;")
        print("    3. whether direction should be ignored by setting DIRECTED = False;")
        print("    4. whether base.csv contains the full structured KG.")

    # ----------------------------
    # 5. Extract new type-pairs introduced by text
    # ----------------------------
    new_pairs = all_text_pairs[
        all_text_pairs["base_triple_count"] == 0
    ].copy()

    # Filter by text count threshold
    new_pairs = new_pairs[
        new_pairs["text_triple_count"] >= MIN_TEXT_TRIPLE_COUNT
    ].copy()

    new_pairs = new_pairs.sort_values(
        "text_triple_count",
        ascending=False
    ).reset_index(drop=True)

    new_pairs["rank"] = np.arange(1, len(new_pairs) + 1)
    new_pairs["log10_count"] = np.log10(new_pairs["text_triple_count"] + 1)

    new_pairs.to_csv(OUTPUT_NEW_PAIR_CSV, index=False)
    print(f"Saved new type-pair file: {OUTPUT_NEW_PAIR_CSV}")
    print(f"New type-pairs retained for heatmap: {len(new_pairs):,}")

    print("\nTop new type-pairs:")
    print(new_pairs.head(30))

    # ----------------------------
    # 6. Build heatmap matrix
    # ----------------------------
    value_col = "log10_count" if USE_LOG10 else "text_triple_count"

    heatmap_df = new_pairs.pivot_table(
        index=SOURCE_TYPE_COL,
        columns=TARGET_TYPE_COL,
        values=value_col,
        aggfunc="sum",
        fill_value=0
    )

    raw_count_df = new_pairs.pivot_table(
        index=SOURCE_TYPE_COL,
        columns=TARGET_TYPE_COL,
        values="text_triple_count",
        aggfunc="sum",
        fill_value=0
    )

    # 强制只显示 11 x 11
    heatmap_df = heatmap_df.reindex(
        index=CORE_ENTITY_TYPES,
        columns=CORE_ENTITY_TYPES,
        fill_value=0
    )

    raw_count_df = raw_count_df.reindex(
        index=CORE_ENTITY_TYPES,
        columns=CORE_ENTITY_TYPES,
        fill_value=0
    )

    mask = heatmap_df == 0

    # ----------------------------
    # 7. Plot heatmap
    # ----------------------------
    from matplotlib.colors import LinearSegmentedColormap

    sns.set_theme(style="white", font_scale=0.95)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # 自定义 colormap：避免低值太接近纯白
    # 你也可以换成 "magma" / "rocket_r"
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_orange_red",
        ["#fee8c8", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"]
    )

    plot_df = heatmap_df.copy()

    # 若全为0则防止报错
    positive_vals = plot_df.values[plot_df.values > 0]
    if len(positive_vals) > 0:
        vmin = positive_vals.min()   # 让最小非零值也有明显颜色
        vmax = positive_vals.max()
    else:
        vmin, vmax = 0, 1

    if ANNOTATE_COUNTS:
        annot_matrix = raw_count_df.copy().astype(object)
        for i in annot_matrix.index:
            for j in annot_matrix.columns:
                val = raw_count_df.loc[i, j]
                if val >= ANNOTATE_MIN_COUNT:
                    annot_matrix.loc[i, j] = f"{int(val):,}"
                else:
                    annot_matrix.loc[i, j] = ""
        annot = annot_matrix
        fmt = ""
    else:
        annot = False
        fmt = ""

    cbar_label = (
        r"$\log_{10}$(text-derived triple count + 1)"
        if USE_LOG10 else
        "Text-derived triple count"
    )

    # 先画热图
    hm = sns.heatmap(
        plot_df,
        mask=(plot_df == 0),          # 只隐藏 truly-zero 的格子
        cmap=custom_cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=1.0,
        linecolor="#d9d9d9",
        square=True,
        annot=annot,
        fmt=fmt,
        annot_kws={"fontsize": 7, "color": "black"},
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        ax=ax
    )

    # 手动画整个 11x11 网格线，让空白区域也有边框感
    n_rows, n_cols = plot_df.shape
    for x in range(n_cols + 1):
        ax.axvline(x, color="#cfcfcf", lw=0.8, zorder=3)
    for y in range(n_rows + 1):
        ax.axhline(y, color="#cfcfcf", lw=0.8, zorder=3)

    # 整个 heatmap 外框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    ax.set_title(
        "New entity type-pair links introduced by text mining",
        fontsize=13,
        pad=14
)

    ax.set_xlabel("Target entity type", fontsize=11)
    ax.set_ylabel("Source entity type", fontsize=11)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0
    )

    note = (
        "Cells indicate source-target entity type pairs observed in the text-derived KG "
        "but absent from the baseline structured KG. Only core entity types and "
        f"pairs with text-derived triple count >= {MIN_TEXT_TRIPLE_COUNT} are shown."
)

    fig.text(
        0.5, 0.01,
        note,
        ha="center",
        va="bottom",
        fontsize=8
)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
