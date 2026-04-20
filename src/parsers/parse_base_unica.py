from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PARSED_DIR = Path("data/parsed")
OUTPUT_DIR = Path("data/parsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    PARSED_DIR / "fmc920_parsed.parquet",
    PARSED_DIR / "zefiro_parsed.parquet",
    PARSED_DIR / "abs_parsed.parquet",
]

OUTPUT_PARQUET = OUTPUT_DIR / "base_unica_parsed.parquet"
OUTPUT_CSV = OUTPUT_DIR / "base_unica_parsed.csv"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_parquet_or_fail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")
    return pd.read_parquet(path)


def union_columns(dfs: List[pd.DataFrame]) -> List[str]:
    cols = set()
    for df in dfs:
        cols.update(df.columns.tolist())
    return sorted(cols)


def align_columns(df: pd.DataFrame, all_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in all_columns:
        if col not in out.columns:
            out[col] = None
    return out[all_columns]


# ============================================================
# EXECUÇÃO
# ============================================================

def main() -> None:
    print("Lendo bases parsed individuais...")

    dfs = []
    for file in INPUT_FILES:
        df = load_parquet_or_fail(file)
        print(f"- {file.name}: {len(df)} registros")
        dfs.append(df)

    all_columns = union_columns(dfs)

    print(f"\nTotal de colunas na união dos schemas: {len(all_columns)}")

    aligned_dfs = [align_columns(df, all_columns) for df in dfs]

    all_parsed = pd.concat(aligned_dfs, ignore_index=True)

    # tipagens úteis
    if "event_time" in all_parsed.columns:
        all_parsed["event_time_dt"] = pd.to_datetime(
            all_parsed["event_time"],
            format="ISO8601",
            errors="coerce"
        )
    else:
        all_parsed["event_time_dt"] = pd.NaT

    if "ingestion_time" in all_parsed.columns:
        all_parsed["ingestion_time_dt"] = pd.to_datetime(
            all_parsed["ingestion_time"],
            format="ISO8601",
            errors="coerce"
        )
    else:
        all_parsed["ingestion_time_dt"] = pd.NaT

    # ordenação principal
    sort_cols = [c for c in ["event_time_dt", "hardware", "device_id"] if c in all_parsed.columns]
    if sort_cols:
        all_parsed = all_parsed.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    # salva também versão sem colunas auxiliares de ordenação
    all_parsed.to_parquet(OUTPUT_PARQUET, index=False)

    csv_export = all_parsed.drop(columns=["event_time_dt", "ingestion_time_dt"], errors="ignore")
    csv_export.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaídas geradas:")
    print(f"- {OUTPUT_PARQUET}")
    print(f"- {OUTPUT_CSV}")

    print("\nResumo consolidado:")
    print(f"Total de registros: {len(all_parsed)}")

    if "hardware" in all_parsed.columns:
        print("\nRegistros por hardware:")
        print(all_parsed["hardware"].value_counts(dropna=False).to_string())

    if "parse_ok" in all_parsed.columns:
        print("\nTaxa de parse_ok por hardware:")
        parse_summary = (
            all_parsed
            .groupby("hardware", dropna=False)["parse_ok"]
            .mean()
            .mul(100)
            .round(2)
        )
        print(parse_summary.to_string())

    if "data_quality_ok" in all_parsed.columns:
        print("\nTaxa de data_quality_ok por hardware:")
        dq_summary = (
            all_parsed
            .groupby("hardware", dropna=False)["data_quality_ok"]
            .mean()
            .mul(100)
            .round(2)
        )
        print(dq_summary.to_string())

    if "parse_error" in all_parsed.columns:
        print("\nPrincipais parse_error:")
        print(all_parsed["parse_error"].fillna("SEM_ERRO").value_counts().head(10).to_string())

    if "data_quality_flags" in all_parsed.columns:
        print("\nPrincipais data_quality_flags:")
        print(all_parsed["data_quality_flags"].fillna("SEM_FLAG").value_counts().head(15).to_string())


if __name__ == "__main__":
    main()