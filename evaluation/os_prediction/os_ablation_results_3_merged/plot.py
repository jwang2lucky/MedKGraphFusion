from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


def load_and_prepare(csv_path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    # 只保留指定模型
    df = df[df["model"] == model_name].copy()

    # 基本清洗
    df["os_time"] = pd.to_numeric(df["os_time"], errors="coerce")
    df["os_event"] = pd.to_numeric(df["os_event"], errors="coerce").fillna(0).astype(int)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df["stage_is_4"] = pd.to_numeric(df["stage_is_4"], errors="coerce").fillna(0).astype(int)

    # stage_group 有时文件里已存在，这里保险起见重建一遍
    df["stage_group"] = np.where(df["stage_is_4"] == 1, "Stage IV", "Stage I-III")

    # 去掉关键列缺失
    df = df.dropna(subset=["patient_id", "risk_score", "os_time", "os_event"]).copy()

    # 有些文件里可能有重复行，这里按 patient_id + model 去重
    # 理论上 OOF 下同一 dataset+model 每个病人只应保留 1 行
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

    # qcut 自动按四分位分组
    late["risk_quartile"] = pd.qcut(
        late["risk_score"],
        q=4,
        labels=["q0", "q1", "q2", "q3"],
        duplicates="drop",
    )

    n_groups = late["risk_quartile"].nunique(dropna=True)
    if n_groups < 4:
        print(f"[WARN] Only {n_groups} quartile groups were created (ties in risk scores may cause this).")

    return late


def plot_hist_and_km(
    df: pd.DataFrame,
    model_name: str,
    out_prefix: str,
    cancer_title: str = "Pancreatic cancer",
) -> None:
    early = df[df["stage_is_4"] == 0].copy()
    late = assign_stage_iv_quartiles(df)

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.30)

    # -----------------------------
    # Top panel: histogram
    # -----------------------------
    ax1 = fig.add_subplot(gs[0])

    all_scores = df["risk_score"].dropna()
    bins = np.linspace(all_scores.min(), all_scores.max(), 22)

    ax1.hist(
        early["risk_score"].dropna(),
        bins=bins,
        alpha=0.9,
        label="Stage I-III",
    )
    ax1.hist(
        late["risk_score"].dropna(),
        bins=bins,
        alpha=0.9,
        label="Stage IV",
    )

    ax1.set_title(f"{cancer_title}\n{model_name}", fontsize=13)
    ax1.set_xlabel("Risk score", fontsize=11)
    ax1.set_ylabel("Number of patients", fontsize=11)
    ax1.legend(frameon=False, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # -----------------------------
    # Bottom panel: KM for Stage IV quartiles
    # -----------------------------
    ax2 = fig.add_subplot(gs[1])
    kmf = KaplanMeierFitter()

    quartile_order = ["q0", "q1", "q2", "q3"]

    for q in quartile_order:
        sub = late[late["risk_quartile"] == q].copy()
        if len(sub) == 0:
            continue

        kmf.fit(
            durations=sub["os_time"],
            event_observed=sub["os_event"],
            label=q,
        )
        kmf.plot_survival_function(
            ax=ax2,
            ci_show=False,
            linewidth=2,
        )

    ax2.set_xlabel("OS time", fontsize=11)
    ax2.set_ylabel("OS", fontsize=11)
    ax2.set_ylim(0, 1.02)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    logrank = multivariate_logrank_test(
        event_durations=late["os_time"],
        groups=late["risk_quartile"],
        event_observed=late["os_event"],
    )

    n_stage4 = len(late)
    n_early = len(early)
    txt = (
        f"Stage IV\nRSF quartiles\n"
        f"n={n_stage4}\n"
        f"log-rank p={logrank.p_value:.2e}"
    )
    ax2.text(
        0.98,
        0.96,
        txt,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {out_prefix}.png")
    print(f"[OK] Saved: {out_prefix}.pdf")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Stage I-III n = {n_early}")
    print(f"[INFO] Stage IV n = {n_stage4}")
    print(f"[INFO] Stage IV log-rank p = {logrank.p_value:.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to pancreatic_main_dataset_oof_predictions.csv",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g. clinical_treatment_molecular_kg",
    )
    parser.add_argument(
        "--out_prefix",
        default="pancreatic_case_study",
        help="Output prefix for figure files",
    )
    args = parser.parse_args()

    df = load_and_prepare(args.csv, args.model)
    plot_hist_and_km(
        df=df,
        model_name=args.model,
        out_prefix=args.out_prefix,
        cancer_title="Pancreatic cancer",
    )


if __name__ == "__main__":
    main()