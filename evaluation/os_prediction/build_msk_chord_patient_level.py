from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_cbio_tsv(path: str) -> pd.DataFrame:
    # cBioPortal 下载的 txt 前面有很多 # 注释行，直接跳过
    return pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")


def clean_text(x):
    return str(x).strip()


def normalize_os_event(x: str) -> int:
    x = clean_text(x)
    return 1 if x.startswith("1:DECEASED") else 0


def to_float_safe(x):
    try:
        return float(x)
    except Exception:
        return None


def choose_index_sample(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    每个患者选一个 index sample。
    优先规则：
    1. Primary 优于 Metastasis / 其他
    2. SAMPLE_ID 字典序最小
    """
    df = sample_df.copy()

    def sample_rank(x: str) -> int:
        x = clean_text(x).lower()
        if x == "primary":
            return 0
        if x == "metastasis":
            return 1
        return 2

    df["_sample_rank"] = df["SAMPLE_TYPE"].apply(sample_rank)
    df = df.sort_values(["PATIENT_ID", "_sample_rank", "SAMPLE_ID"])
    idx_df = df.groupby("PATIENT_ID", as_index=False).first()
    return idx_df.drop(columns=["_sample_rank"], errors="ignore")


def aggregate_treatments(treatment_df: pd.DataFrame) -> pd.DataFrame:
    """
    聚合每个患者的治疗药物列表
    """
    df = treatment_df.copy()
    df["START_DATE_NUM"] = pd.to_numeric(df["START_DATE"], errors="coerce")
    df["STOP_DATE_NUM"] = pd.to_numeric(df["STOP_DATE"], errors="coerce")
    df["AGENT"] = df["AGENT"].map(clean_text)

    df = df[df["AGENT"] != ""]
    df = df.sort_values(["PATIENT_ID", "START_DATE_NUM", "STOP_DATE_NUM", "AGENT"])

    agg = (
        df.groupby("PATIENT_ID")["AGENT"]
        .apply(lambda s: json.dumps(sorted(set([x for x in s if x])), ensure_ascii=False))
        .reset_index()
        .rename(columns={"AGENT": "treatments"})
    )
    return agg


def aggregate_mutations(mutation_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    通过 Tumor_Sample_Barcode -> SAMPLE_ID -> PATIENT_ID
    聚合每个患者的 altered genes
    """
    muts = mutation_df.copy()
    muts["Tumor_Sample_Barcode"] = muts["Tumor_Sample_Barcode"].map(clean_text)
    muts["Hugo_Symbol"] = muts["Hugo_Symbol"].map(clean_text)

    samp = sample_df[["SAMPLE_ID", "PATIENT_ID"]].copy()
    samp["SAMPLE_ID"] = samp["SAMPLE_ID"].map(clean_text)
    samp["PATIENT_ID"] = samp["PATIENT_ID"].map(clean_text)

    merged = muts.merge(
        samp,
        left_on="Tumor_Sample_Barcode",
        right_on="SAMPLE_ID",
        how="inner"
    )
    merged = merged[merged["Hugo_Symbol"] != ""]

    agg = (
        merged.groupby("PATIENT_ID")["Hugo_Symbol"]
        .apply(lambda s: json.dumps(sorted(set([x for x in s if x])), ensure_ascii=False))
        .reset_index()
        .rename(columns={"Hugo_Symbol": "altered_genes"})
    )
    return agg


def main(data_dir: str, out_csv: str):
    data_dir = Path(data_dir)

    patient_path = data_dir / "data_clinical_patient.txt"
    sample_path = data_dir / "data_clinical_sample.txt"
    mutation_path = data_dir / "data_mutations.txt"
    treatment_path = data_dir / "data_timeline_treatment.txt"

    patient_df = read_cbio_tsv(str(patient_path))
    sample_df = read_cbio_tsv(str(sample_path))
    mutation_df = read_cbio_tsv(str(mutation_path))
    treatment_df = read_cbio_tsv(str(treatment_path))

    index_sample_df = choose_index_sample(sample_df)
    treat_agg_df = aggregate_treatments(treatment_df)
    mut_agg_df = aggregate_mutations(mutation_df, sample_df)

    # patient-level 主表
    out = patient_df.copy()

    # OS 字段
    out["os_time"] = out["OS_MONTHS"].apply(to_float_safe)
    out["os_event"] = out["OS_STATUS"].apply(normalize_os_event)

    # 合并 index sample 信息
    sample_keep = index_sample_df[
        [
            "PATIENT_ID",
            "SAMPLE_ID",
            "CANCER_TYPE",
            "CANCER_TYPE_DETAILED",
            "SAMPLE_TYPE",
            "PRIMARY_SITE",
            "MSI_TYPE",
            "TMB_NONSYNONYMOUS",
            "TUMOR_PURITY",
            "GENE_PANEL",
        ]
    ].copy()

    out = out.merge(sample_keep, on="PATIENT_ID", how="left")
    out = out.merge(treat_agg_df, on="PATIENT_ID", how="left")
    out = out.merge(mut_agg_df, on="PATIENT_ID", how="left")

    out["treatments"] = out["treatments"].fillna("[]")
    out["altered_genes"] = out["altered_genes"].fillna("[]")

    # 重命名成后续建模更好用的名字
    rename_map = {
        "PATIENT_ID": "patient_id",
        "SAMPLE_ID": "sample_id",
        "CANCER_TYPE": "cancer_type",
        "CANCER_TYPE_DETAILED": "cancer_type_detailed",
        "SAMPLE_TYPE": "sample_type",
        "PRIMARY_SITE": "primary_site",
        "CURRENT_AGE_DEID": "age",
        "GENDER": "sex",
        "RACE": "race",
        "ETHNICITY": "ethnicity",
        "STAGE_HIGHEST_RECORDED": "stage",
        "MSI_TYPE": "msi_type",
        "TMB_NONSYNONYMOUS": "tmb",
        "TUMOR_PURITY": "tumor_purity",
        "GENE_PANEL": "gene_panel",
        "SMOKING_PREDICTIONS_3_CLASSES": "smoking_history",
        "PRIOR_MED_TO_MSK": "prior_treatment_to_msk",
    }
    out = out.rename(columns=rename_map)

    keep_cols = [
        "patient_id",
        "sample_id",
        "cancer_type",
        "cancer_type_detailed",
        "sample_type",
        "primary_site",
        "age",
        "sex",
        "race",
        "ethnicity",
        "stage",
        "msi_type",
        "tmb",
        "tumor_purity",
        "gene_panel",
        "smoking_history",
        "prior_treatment_to_msk",
        "os_time",
        "os_event",
        "treatments",
        "altered_genes",
    ]

    for c in keep_cols:
        if c not in out.columns:
            out[c] = ""

    out = out[keep_cols].copy()

    # 基本清洗
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["tmb"] = pd.to_numeric(out["tmb"], errors="coerce")
    out["tumor_purity"] = pd.to_numeric(out["tumor_purity"], errors="coerce")

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved to: {out_csv}")
    print(f"Rows: {len(out):,}")
    print(out.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="MSK-CHORD txt 文件所在目录")
    parser.add_argument("--out_csv", default="msk_chord_patient_level.csv")
    args = parser.parse_args()

    main(args.data_dir, args.out_csv)