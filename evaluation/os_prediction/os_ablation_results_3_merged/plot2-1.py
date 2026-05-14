from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel



# ─── 实验组定义与标签 ────────────────────────────────────────────────────────
MODEL_ORDER = [
    "clinical_only",
    "clinical_kg",
    "clinical_molecular",
    "clinical_molecular_kg",
    "clinical_treatment_molecular",
    "clinical_treatment_molecular_kg",
]

MODEL_LABELS = {
    "clinical_only": "Clinical",
    "clinical_kg": "Clinical\n+ KG",
    "clinical_molecular": "Clinical\n+ Molecular",
    "clinical_molecular_kg": "Clinical\n+ Molecular\n+ KG",
    "clinical_treatment_molecular": "Clinical\n+ Treatment\n+ Molecular",
    "clinical_treatment_molecular_kg": "Clinical\n+ Treatment\n+ Molecular\n+ KG",
}

# 规划显著性检验的配对：(组A, 组B)
COMPARISON_PAIRS = [
    ("clinical_only", "clinical_kg"),
    ("clinical_molecular", "clinical_molecular_kg"),
    ("clinical_treatment_molecular", "clinical_treatment_molecular_kg"),
]

CANCER_TYPES = [
    "Non-Small Cell Lung Cancer",
    "Colorectal Cancer",
    "Breast Cancer",
    "Prostate Cancer",
    "Pancreatic Cancer",
]

STAGE_GROUPS = ["Stage I-III", "Stage IV"]


# ─── 统计与绘图辅助函数 ────────────────────────────────────────────────────────
def _get_p_stars(p: float) -> str:
    """根据p值返回星号"""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "" # ns (不显著的话留空，保持图面整洁)


def _draw_bracket(ax, x1, x2, y, h=0.01, text="", fontsize=11):
    """画统计检验的括号和星号"""
    # 竖线与横线
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="black")
    # 星号位置
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", color="black", fontsize=fontsize, fontweight="bold")
    return y + h + 0.035 # 返回新的高度，防止重叠


def _summarize_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # returns columns: dataset, model, mean, std
    out = (
        df.groupby(["dataset", "model"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    return out


def _filter_present_datasets(df: pd.DataFrame, datasets: list[str]) -> list[str]:
    present = []
    missing = []
    for d in datasets:
        if (df["dataset"] == d).any():
            present.append(d)
        else:
            missing.append(d)
    if missing:
        print(f"[WARN] Missing datasets in fold-metrics CSV, will skip: {missing}")
    return present


# ─── 核心画图逻辑 ────────────────────────────────────────────────────────────
def _plot_bargrid(
    df: pd.DataFrame,
    datasets: list[str],
    titles: dict[str, str],
    metric: str,
    out_png: str,
    out_pdf: str,
    suptitle: str,
    ncols: int,
    figsize: tuple[float, float],
):
    if df.empty or not datasets:
        raise ValueError("No rows/datasets to plot.")

    summary = _summarize_metric(df, metric)

    n_panels = len(datasets)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    # 动态确定 y 轴范围，给顶部的括号留出空间
    all_vals = df[metric].dropna().to_numpy()
    if len(all_vals) == 0:
        y_min, y_max = 0.5, 1.0
    else:
        y_min = max(0.4, np.floor((all_vals.min() - 0.05) * 20) / 20)
        y_max = min(1.05, np.ceil((all_vals.max() + 0.12) * 20) / 20) # 顶部多留 0.12 空间

    x = np.arange(len(MODEL_ORDER))

    for idx, dataset_name in enumerate(datasets):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        sub_folds = df[df["dataset"] == dataset_name].copy()
        sub_sum = summary[summary["dataset"] == dataset_name].copy()

        means = []
        for m in MODEL_ORDER:
            row = sub_sum[sub_sum["model"] == m]
            means.append(float(row["mean"].iloc[0]) if len(row) else np.nan)

        ax.bar(
            x,
            means,
            width=0.72,
            color="#d9d9d9",
            edgecolor="black",
            linewidth=0.8,
            zorder=1,
        )

        # 叠加每个 fold 的散点
        for i, model_name in enumerate(MODEL_ORDER):
            vals = sub_folds[sub_folds["model"] == model_name][metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            jitter = np.array([0.0]) if len(vals) == 1 else np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), i, dtype=float) + jitter,
                vals,
                s=18,
                color="black",
                zorder=3,
                linewidths=0.0,
            )

        # ─── 显著性检验与画线 ──────────────────────────────
        for pair in COMPARISON_PAIRS:
            if pair[0] in MODEL_ORDER and pair[1] in MODEL_ORDER:
                idx1 = MODEL_ORDER.index(pair[0])
                idx2 = MODEL_ORDER.index(pair[1])
                
                vals1 = sub_folds[sub_folds["model"] == pair[0]][metric].dropna().to_numpy()
                vals2 = sub_folds[sub_folds["model"] == pair[1]][metric].dropna().to_numpy()
                
                # 只有折数相同且数量大于1才能做配对t检验
                if len(vals1) > 1 and len(vals1) == len(vals2):
                    # 判断数据是否完全一样（避免方差为0报错）
                    if not np.allclose(vals1, vals2):
                        _, p_val = ttest_rel(vals1, vals2)
                        stars = _get_p_stars(p_val)
                        
                        if stars: # 只有显著才画
                            # 确定括号的起始高度（当前两个柱子中最高的散点或柱高 + offset）
                            max_val1 = vals1.max() if len(vals1) > 0 else means[idx1]
                            max_val2 = vals2.max() if len(vals2) > 0 else means[idx2]
                            base_height = max(max_val1, max_val2) + 0.015
                            
                            _draw_bracket(ax, idx1, idx2, base_height, h=0.01, text=stars)

        # 设置坐标轴外观
        ax.set_title(titles.get(dataset_name, dataset_name), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=45, ha="right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18, linewidth=0.7, zorder=0)
        ax.set_ylim(y_min, y_max)

    # 关闭不需要的子图边框
    for idx in range(n_panels, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    axes[0, 0].set_ylabel("C-index" if metric == "c_index" else "Mean td-AUC", fontsize=12)
    fig.suptitle(suptitle, fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig(out_png, dpi=500, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_png}")
    print(f"[OK] Saved {out_pdf}")


def plot_all_cancers_and_stages(
    fold_metrics_csv: str,
    metric: str = "c_index",
    out_prefix: str = "os_stage_barplots",
):
    df = pd.read_csv(fold_metrics_csv).copy()

    # keep only needed models
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    if df.empty:
        raise ValueError("No rows found after filtering models. Check MODEL_ORDER vs CSV.")

    # -------------------------
    # (A) Main datasets: all_cancers + 5 cancers
    # -------------------------
    main_datasets = ["all_cancers"] + CANCER_TYPES
    main_datasets = _filter_present_datasets(df, main_datasets)

    main_titles = {"all_cancers": "All cancers"}
    for ct in CANCER_TYPES:
        main_titles[ct] = ct

    df_main = df[df["dataset"].isin(main_datasets)].copy()
    if len(main_datasets) > 0 and not df_main.empty:
        _plot_bargrid(
            df=df_main,
            datasets=main_datasets,
            titles=main_titles,
            metric=metric,
            out_png=f"{out_prefix}_main.png",
            out_pdf=f"{out_prefix}_main.pdf",
            suptitle="OS prediction (main cohorts)",
            ncols=3,
            figsize=(15.5, 9.0),
        )
    else:
        print("[WARN] No main-cohort datasets found to plot.")

    # -------------------------
    # (B) Stage datasets: 5 cancers × (Stage I-III, Stage IV)
    # -------------------------
    stage_datasets = []
    stage_titles = {}
    for ct in CANCER_TYPES:
        for sg in STAGE_GROUPS:
            name = f"{ct} - {sg}"
            stage_datasets.append(name)
            # nicer titles per panel
            stage_titles[name] = f"{ct}\n{sg.replace('Stage ', 'Stage ')}"

    stage_datasets = _filter_present_datasets(df, stage_datasets)
    df_stage = df[df["dataset"].isin(stage_datasets)].copy()

    if len(stage_datasets) > 0 and not df_stage.empty:
        # 固定 2 列：左 I–III，右 IV；共 5 行
        # 为了保持顺序：每个癌种按 I-III, IV 排列
        ordered_stage = []
        for ct in CANCER_TYPES:
            for sg in STAGE_GROUPS:
                name = f"{ct} - {sg}"
                if name in stage_datasets:
                    ordered_stage.append(name)

        _plot_bargrid(
            df=df_stage,
            datasets=ordered_stage,
            titles=stage_titles,
            metric=metric,
            out_png=f"{out_prefix}_stage.png",
            out_pdf=f"{out_prefix}_stage.pdf",
            suptitle="OS prediction (stage-stratified cohorts)",
            ncols=2,
            figsize=(14.5, 18.5), # 微调画布高度以适配星号
        )
    else:
        print("[WARN] No stage-stratified datasets found to plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold_metrics_csv",
        required=True,
        help="Path to os_ablation_fold_metrics_all_datasets.csv",
    )
    parser.add_argument(
        "--metric",
        default="c_index",
        choices=["c_index", "td_auc_mean"],
        help="Metric to plot.",
    )
    parser.add_argument(
        "--out_prefix",
        default="os_stage_barplots",
        help="Output file prefix (will produce *_main and *_stage).",
    )
    args = parser.parse_args()

    plot_all_cancers_and_stages(
        fold_metrics_csv=args.fold_metrics_csv,
        metric=args.metric,
        out_prefix=args.out_prefix,
    )