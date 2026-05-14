from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_cbio_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")


def parse_json_list(text) -> list[str]:
    if pd.isna(text) or str(text).strip() == "":
        return []
    s = str(text).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return []


def normalize_stage(x: str) -> str:
    x = str(x).strip()
    if x in {"Stage 4", "4"}:
        return "Stage 4"
    if x in {"Stage 1-3", "1", "2", "3"}:
        return "Stage 1-3"
    return "Unknown"


def normalize_os_event(x) -> int:
    s = str(x).strip()
    return 1 if s in {"1", "1.0", "1:DECEASED"} else 0


def build_sample_ambiguity_table(sample_path: str) -> pd.DataFrame:
    sdf = read_cbio_tsv(sample_path)

    keep_cols = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "CANCER_TYPE",
        "CANCER_TYPE_DETAILED",
        "SAMPLE_TYPE",
    ]
    for c in keep_cols:
        if c not in sdf.columns:
            raise ValueError(f"clinical sample 文件缺少列: {c}")

    def nunique_nonempty(series: pd.Series) -> int:
        vals = {str(x).strip() for x in series if str(x).strip() != ""}
        return len(vals)

    agg = (
        sdf.groupby("PATIENT_ID")
        .agg(
            n_samples=("SAMPLE_ID", nunique_nonempty),
            n_cancer_type=("CANCER_TYPE", nunique_nonempty),
            n_cancer_type_detailed=("CANCER_TYPE_DETAILED", nunique_nonempty),
            n_sample_type=("SAMPLE_TYPE", nunique_nonempty),
        )
        .reset_index()
    )

    agg["is_multi_sample"] = agg["n_samples"] > 1
    agg["is_multi_cancer_type"] = agg["n_cancer_type"] > 1
    agg["is_multi_cancer_type_detailed"] = agg["n_cancer_type_detailed"] > 1

    return agg


def main(msk_csv: str, clinical_sample: str, out_csv: str, qc_json: str):
    df = pd.read_csv(msk_csv).copy()

    required = [
        "patient_id", "sample_id", "cancer_type", "cancer_type_detailed",
        "sample_type", "primary_site", "age", "sex", "race", "ethnicity",
        "stage", "msi_type", "tmb", "tumor_purity", "gene_panel",
        "smoking_history", "prior_treatment_to_msk", "os_time", "os_event",
        "treatments", "altered_genes"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"msk_chord_patient_level.csv 缺少列: {missing}")

    # 基本类型清洗
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["tmb"] = pd.to_numeric(df["tmb"], errors="coerce")
    df["tumor_purity"] = pd.to_numeric(df["tumor_purity"], errors="coerce")
    df["os_time"] = pd.to_numeric(df["os_time"], errors="coerce")
    df["os_event"] = df["os_event"].apply(normalize_os_event)

    # 类别标准化
    for col in [
        "cancer_type", "cancer_type_detailed", "sample_type", "primary_site",
        "sex", "race", "ethnicity", "stage", "msi_type",
        "gene_panel", "smoking_history", "prior_treatment_to_msk"
    ]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": "Unknown"})

    df["stage"] = df["stage"].apply(normalize_stage)

    # 列表字段
    df["treatments_list"] = df["treatments"].apply(parse_json_list)
    df["altered_genes_list"] = df["altered_genes"].apply(parse_json_list)

    df["num_treatments"] = df["treatments_list"].apply(len)
    df["num_altered_genes"] = df["altered_genes_list"].apply(len)

    # 缺失标志
    df["missing_tmb"] = df["tmb"].isna().astype(int)
    df["missing_tumor_purity"] = df["tumor_purity"].isna().astype(int)

    # 从 clinical sample 判断 patient 是否多癌种
    amb = build_sample_ambiguity_table(clinical_sample)
    amb = amb.rename(columns={"PATIENT_ID": "patient_id"})
    df = df.merge(amb, on="patient_id", how="left")

    for c in [
        "n_samples", "n_cancer_type", "n_cancer_type_detailed", "n_sample_type",
        "is_multi_sample", "is_multi_cancer_type", "is_multi_cancer_type_detailed"
    ]:
        if c not in df.columns:
            df[c] = 0

    # 严格队列规则
    strict = df.copy()
    strict = strict[strict["os_time"].notna()]
    strict = strict[strict["os_time"] > 0]
    strict = strict[strict["age"].notna()]
    strict = strict[strict["cancer_type"] != "Unknown"]
    strict = strict[strict["is_multi_cancer_type"] == False]

    # 给模型直接可用的一些额外字段
    strict["stage_is_4"] = (strict["stage"] == "Stage 4").astype(int)
    strict["prior_treatment_binary"] = (
        strict["prior_treatment_to_msk"].str.contains("Prior medications", case=False, na=False)
    ).astype(int)
    strict["sample_is_metastasis"] = (
        strict["sample_type"].str.lower().eq("metastasis")
    ).astype(int)

    # 保存时保留 json string，不保留 list 对象列
    out_cols = [
        "patient_id", "sample_id", "cancer_type", "cancer_type_detailed",
        "sample_type", "primary_site", "age", "sex", "race", "ethnicity",
        "stage", "stage_is_4", "msi_type", "tmb", "tumor_purity",
        "missing_tmb", "missing_tumor_purity", "gene_panel",
        "smoking_history", "prior_treatment_to_msk", "prior_treatment_binary",
        "sample_is_metastasis",
        "os_time", "os_event",
        "num_treatments", "num_altered_genes",
        "treatments", "altered_genes",
        "n_samples", "n_cancer_type", "n_cancer_type_detailed",
        "is_multi_sample", "is_multi_cancer_type", "is_multi_cancer_type_detailed",
    ]
    strict = strict[out_cols].copy()

    strict.to_csv(out_csv, index=False, encoding="utf-8-sig")

    qc = {
        "input_rows": int(len(df)),
        "strict_rows": int(len(strict)),
        "dropped_rows": int(len(df) - len(strict)),
        "num_multi_sample_patients": int(df["is_multi_sample"].fillna(False).sum()),
        "num_multi_cancer_type_patients": int(df["is_multi_cancer_type"].fillna(False).sum()),
        "os_event_rate": float(strict["os_event"].mean()) if len(strict) else None,
        "median_os_time": float(strict["os_time"].median()) if len(strict) else None,
        "missing_tmb_rate": float(strict["missing_tmb"].mean()) if len(strict) else None,
        "missing_tumor_purity_rate": float(strict["missing_tumor_purity"].mean()) if len(strict) else None,
        "top_cancer_types": strict["cancer_type"].value_counts().head(10).to_dict(),
    }

    with open(qc_json, "w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)

    print(f"Saved strict cohort to: {out_csv}")
    print(f"Saved QC summary to: {qc_json}")
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msk_csv", required=True)
    parser.add_argument("--clinical_sample", required=True)
    parser.add_argument("--out_csv", default="os_cohort_strict.csv")
    parser.add_argument("--qc_json", default="os_cohort_qc.json")
    args = parser.parse_args()

    main(
        msk_csv=args.msk_csv,
        clinical_sample=args.clinical_sample,
        out_csv=args.out_csv,
        qc_json=args.qc_json,
    )