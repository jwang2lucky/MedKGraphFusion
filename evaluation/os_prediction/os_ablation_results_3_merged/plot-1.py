from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import pandas as pd
from pathlib import Path
# 定义要寻找的目标文件名和输出的汇总文件名
TARGET_FILENAME = "os_ablation_fold_metrics.csv"
OUTPUT_FILENAME = "os_ablation_fold_metrics_all_datasets.csv"

# 【关键配置】将子文件夹名字自动映射为 plot2.py 需要的标准 dataset 名称
NAME_MAPPING = {
    "all_cancers": "all_cancers",
    
    "breast_cancer": "Breast Cancer",
    "breast_cancer_stage_i_iii": "Breast Cancer - Stage I-III",
    "breast_cancer_stage_iv": "Breast Cancer - Stage IV",
    
    "colorectal_cancer": "Colorectal Cancer",
    "colorectal_cancer_stage_i_iii": "Colorectal Cancer - Stage I-III",
    "colorectal_cancer_stage_iv": "Colorectal Cancer - Stage IV",
    
    "non_small_cell_lung_cancer": "Non-Small Cell Lung Cancer",
    "non_small_cell_lung_cancer_stage_i_iii": "Non-Small Cell Lung Cancer - Stage I-III",
    "non_small_cell_lung_cancer_stage_iv": "Non-Small Cell Lung Cancer - Stage IV",
    
    "pancreatic_cancer": "Pancreatic Cancer",
    "pancreatic_cancer_stage_i_iii": "Pancreatic Cancer - Stage I-III",
    "pancreatic_cancer_stage_iv": "Pancreatic Cancer - Stage IV",
    
    "prostate_cancer": "Prostate Cancer",
    "prostate_cancer_stage_i_iii": "Prostate Cancer - Stage I-III",
    "prostate_cancer_stage_iv": "Prostate Cancer - Stage IV",
}

def csv_merge():
    # 1. 递归查找当前目录及所有子目录下名为 os_ablation_fold_metrics.csv 的文件
    base_dir = Path(".")
    csv_files = list(base_dir.rglob(TARGET_FILENAME))

    if not csv_files:
        print(f"❌ 没有找到任何 '{TARGET_FILENAME}' 文件，请确认脚本是否放在正确的根目录下！")
        return

    print(f"🔍 找到了 {len(csv_files)} 个 CSV 文件，开始读取并合并...\n")

    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            folder_name = file.parent.name  # 获取该 csv 所在的上一级文件夹名

            # 2. 自动修正 dataset 列的名字，以匹配 plot2.py 的画图要求
            if folder_name in NAME_MAPPING:
                df["dataset"] = NAME_MAPPING[folder_name]
            else:
                print(f"  [警告] 文件夹名 '{folder_name}' 不在映射字典中，将保留原 dataset 名称。")

            df_list.append(df)
            print(f"  ✅ 成功读取: {folder_name}/{TARGET_FILENAME}")
            
        except Exception as e:
            print(f"  ❌ 读取 {file} 时出错: {e}")

    # 3. 将所有 DataFrame 上下拼接合并
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)

        # 4. 导出为总的 CSV
        merged_df.to_csv(OUTPUT_FILENAME, index=False)
        print("\n🎉 合并完成！")
        print(f"📊 参与合并的文件数: {len(df_list)} 个")
        print(f"📈 合并后的总数据行数: {len(merged_df)} 行")
        print(f"💾 总文件已保存至: ./{OUTPUT_FILENAME}")
    else:
        print("❌ 没有有效的数据可以合并。")
# ─── Nature/Cell/Science 风格配色 ───────────────────────────────────────────
# 参考 Nature 系列常用色板
NATURE_COLORS = {
    "blue":   "#4878CF",   # Stage I-III
    "red":    "#D65F5F",   # Stage IV
    # KM quartile 颜色：从低风险→高风险，蓝→红渐变
    "q0":     "#4878CF",   # Q1 lowest risk  蓝
    "q1":     "#6AB187",   # Q2              青绿
    "q2":     "#E7A34F",   # Q3              琥珀橙
    "q3":     "#C94040",   # Q4 highest risk 深红
}


def load_and_prepare(csv_path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df = df[df["model"] == model_name].copy()

    df["os_time"]   = pd.to_numeric(df["os_time"],   errors="coerce")
    df["os_event"]  = pd.to_numeric(df["os_event"],  errors="coerce").fillna(0).astype(int)
    df["risk_score"]= pd.to_numeric(df["risk_score"],errors="coerce")
    df["stage_is_4"]= pd.to_numeric(df["stage_is_4"],errors="coerce").fillna(0).astype(int)
    df["stage_group"]= np.where(df["stage_is_4"] == 1, "Stage IV", "Stage I–III")

    df = df.dropna(subset=["patient_id", "risk_score", "os_time", "os_event"]).copy()
    df = (
        df.sort_values(["patient_id", "fold"])
          .drop_duplicates(subset=["patient_id", "model"], keep="first")
          .reset_index(drop=True)
    )
    return df


def assign_stage_iv_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    late = df[df["stage_is_4"] == 1].copy()
    if len(late) < 8:
        raise ValueError("Stage IV patients are too few to create quartiles reliably.")

    late["risk_quartile"] = pd.qcut(
        late["risk_score"],
        q=4,
        labels=["q0", "q1", "q2", "q3"],
        duplicates="drop",
    )
    n_groups = late["risk_quartile"].nunique(dropna=True)
    if n_groups < 4:
        print(f"[WARN] Only {n_groups} quartile groups were created.")
    return late


def _format_model_name(model_name: str) -> str:
    """把 clinical_treatment_molecular_kg → Clinical + Treatment + Molecular + KG"""
    mapping = {
        "clinical":  "Clinical",
        "treatment": "Treatment",
        "molecular": "Molecular",
        "kg":        "KG",
    }
    parts = model_name.split("_")
    pretty = []
    for p in parts:
        pretty.append(mapping.get(p.lower(), p.capitalize()))
    return " + ".join(pretty)


def plot_hist_and_km(
    df: pd.DataFrame,
    model_name: str,
    out_prefix: str,
    cancer_title: str = "Pancreatic cancer",
) -> None:

    early = df[df["stage_is_4"] == 0].copy()
    late  = assign_stage_iv_quartiles(df)

    # ── 全局字体 / 风格 ──────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.linewidth":   0.9,
        "axes.labelsize":   12,
        "axes.titlesize":   13,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "pdf.fonttype":     42,   # editable text in PDF
        "ps.fonttype":      42,
    })

    fig = plt.figure(figsize=(7.2, 6.2))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.38)

    # =========================================================
    # Panel A — Histogram (no overlap, small gap between bars)
    # =========================================================
    ax1 = fig.add_subplot(gs[0])

    all_scores = df["risk_score"].dropna()
    n_bins = 16
    bins   = np.linspace(all_scores.min(), all_scores.max(), n_bins + 1)
    width  = (bins[1] - bins[0])

    # 计算每组在每个 bin 的频数
    early_counts, _ = np.histogram(early["risk_score"].dropna(), bins=bins)
    late_counts,  _ = np.histogram(late["risk_score"].dropna(),  bins=bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    half_gap    = width * 0.03          # 两组 bar 之间的视觉间隔

    bar_w   = width / 2 - half_gap     # 每条 bar 宽度
    offset  = width / 4                # 偏移量，让两组并排

    ax1.bar(
        bin_centers - offset,
        early_counts,
        width=bar_w,
        color=NATURE_COLORS["blue"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
        label="Stage I–III",
    )
    ax1.bar(
        bin_centers + offset,
        late_counts,
        width=bar_w,
        color=NATURE_COLORS["red"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
        label="Stage IV",
    )

    pretty_model = _format_model_name(model_name)
    ax1.set_title(f"{cancer_title}  ·  {pretty_model}", pad=8, fontweight="bold")
    ax1.set_xlabel("Risk score")
    ax1.set_ylabel("Number of patients")
    ax1.legend(frameon=False, loc="upper right", handlelength=1.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    # =========================================================
    # Panel B — KM curves (Stage IV risk quartiles)
    # =========================================================
    ax2 = fig.add_subplot(gs[1])
    kmf  = KaplanMeierFitter()

    quartile_order  = ["q0", "q1", "q2", "q3"]
    quartile_labels = {
        "q0": "Q1 – lowest risk",
        "q1": "Q2",
        "q2": "Q3",
        "q3": "Q4 – highest risk",
    }

    for q in quartile_order:
        sub = late[late["risk_quartile"] == q].copy()
        if len(sub) == 0:
            continue
        kmf.fit(
            durations=sub["os_time"],
            event_observed=sub["os_event"],
            label=quartile_labels[q],
        )
        kmf.plot_survival_function(
            ax=ax2,
            ci_show=False,
            linewidth=2.2,
            color=NATURE_COLORS[q],
        )

    ax2.set_xlabel("Overall survival time (months)")
    ax2.set_ylabel("Overall survival probability")
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    # legend 放左下，不遮挡曲线
    ax2.legend(frameon=False, loc="upper right", handlelength=2.0)

    # annotation box ─ 右上角
    logrank = multivariate_logrank_test(
        event_durations=late["os_time"],
        groups=late["risk_quartile"],
        event_observed=late["os_event"],
    )
    n_stage4 = len(late)
    n_early  = len(early)

    pval = logrank.p_value
    pval_str = f"{pval:.2e}" if pval >= 1e-300 else "< 1×10⁻³⁰⁰"

    annotation = (
        f"Stage IV  |  RSF quartiles\n"
        f"n = {n_stage4}  "
        f"  log-rank p = {pval_str}"
    )
    ax2.text(
        0.98, 0.96,
        annotation,
        transform=ax2.transAxes,
        ha="right", va="top",
        fontsize=9.5,
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=0.8,
            alpha=0.95,
        ),
    )

    # ── 保存 ─────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=500, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf",           bbox_inches="tight")
    plt.close()

    print(f"[OK]   {out_prefix}.png / .pdf")
    print(f"[INFO] Model       : {model_name}")
    print(f"[INFO] Stage I-III : n = {n_early}")
    print(f"[INFO] Stage IV    : n = {n_stage4}")
    print(f"[INFO] log-rank p  : {pval:.4e}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--out_prefix", default="pancreatic_case_study")
    args = parser.parse_args()

    df = load_and_prepare(args.csv, args.model)
    plot_hist_and_km(
        df=df,
        model_name=args.model,
        out_prefix=args.out_prefix,
        cancer_title="Pancreatic cancer",
    )


if __name__ == "__main__":
    csv_merge()
    main()