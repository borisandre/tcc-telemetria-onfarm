from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_PARSED_PATH = Path("data/parsed/base_unica_parsed.parquet")
SCHEDULE_PATH = Path("data/context/Dados de Operação.xlsx")

OUTPUT_DIR = Path("data/normalized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "base_enriquecida_agenda.parquet"
OUTPUT_CSV = OUTPUT_DIR / "base_enriquecida_agenda.csv"


# ============================================================
# UTILITÁRIOS
# ============================================================

def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\xa0", " ").strip()


def parse_datetime_safe(value) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(value, format="ISO8601", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(value, dayfirst=True, errors="coerce")
    return ts if not pd.isna(ts) else None


def combine_date_time_br(date_value, time_value) -> Optional[pd.Timestamp]:
    date_ts = pd.to_datetime(date_value, dayfirst=True, errors="coerce")
    if pd.isna(date_ts):
        return None

    time_str = clean_str(time_value)
    if not time_str:
        return None

    time_ts = pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")
    if pd.isna(time_ts):
        time_ts = pd.to_datetime(time_str, errors="coerce")
    if pd.isna(time_ts):
        return None

    return pd.Timestamp(
        year=date_ts.year,
        month=date_ts.month,
        day=date_ts.day,
        hour=time_ts.hour,
        minute=time_ts.minute,
        second=time_ts.second,
    )


def normalize_hardware_name(hw: str) -> str:
    s = clean_str(hw).upper()
    mapping = {
        "FMC920": "FMC920",
        "ZEFIRO": "ZEFIRO",
        "ABS": "ABS",
    }
    return mapping.get(s, s)


def to_float_br(value):
    if pd.isna(value):
        return None
    s = clean_str(value)
    if not s:
        return None

    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return None


# ============================================================
# LEITURA DA BASE PARSED
# ============================================================

def load_base_parsed() -> pd.DataFrame:
    if not BASE_PARSED_PATH.exists():
        raise FileNotFoundError(f"Base parsed não encontrada: {BASE_PARSED_PATH.resolve()}")

    df = pd.read_parquet(BASE_PARSED_PATH)

    if "event_time" not in df.columns:
        raise ValueError("A base parsed precisa conter a coluna 'event_time'.")

    df["event_time_dt"] = pd.to_datetime(df["event_time"], format="ISO8601", errors="coerce")
    if df["event_time_dt"].isna().any():
        df["event_time_dt"] = pd.to_datetime(df["event_time"], errors="coerce")

    df["hardware"] = df["hardware"].apply(normalize_hardware_name)
    return df


# ============================================================
# LEITURA E PREPARAÇÃO DA AGENDA
# ============================================================

def load_schedule() -> pd.DataFrame:
    if not SCHEDULE_PATH.exists():
        raise FileNotFoundError(f"Arquivo de agenda não encontrado: {SCHEDULE_PATH.resolve()}")

    df = pd.read_excel(SCHEDULE_PATH, dtype=str)
    df.columns = [clean_str(c) for c in df.columns]

    required = [
        "Hardware",
        "Data",
        "Hora Inicio",
        "Hora Termino",
        "Semente",
        "Peso Kg",
        "Qtd. Sacas",
        "Nº Máquina",
        "Estado Cliente",
        "Cidade Cliente",
        "Local de execução",
        "Referência Latitude",
        "Referência Longitude",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no Excel da agenda: {missing}")

    for col in df.columns:
        df[col] = df[col].apply(clean_str)

    df["hardware"] = df["Hardware"].apply(normalize_hardware_name)
    df["scheduled_start"] = df.apply(lambda r: combine_date_time_br(r["Data"], r["Hora Inicio"]), axis=1)
    df["scheduled_end"] = df.apply(lambda r: combine_date_time_br(r["Data"], r["Hora Termino"]), axis=1)

    # Se eventualmente cruzar meia-noite
    mask = df["scheduled_end"] < df["scheduled_start"]
    df.loc[mask, "scheduled_end"] = df.loc[mask, "scheduled_end"] + pd.to_timedelta(1, unit="D")

    df["weight_kg_schedule"] = df["Peso Kg"].apply(to_float_br)
    df["bags_qty_schedule"] = df["Qtd. Sacas"].apply(to_float_br)
    df["reference_latitude"] = df["Referência Latitude"].apply(to_float_br)
    df["reference_longitude"] = df["Referência Longitude"].apply(to_float_br)
    df["machine_number"] = df["Nº Máquina"]

    # session_id formal
    df = df.reset_index(drop=True)
    df["schedule_row_id"] = df.index + 1
    df["session_id_schedule"] = df.apply(
        lambda r: f"{r['hardware']}_{pd.Timestamp(r['scheduled_start']).strftime('%Y%m%d_%H%M%S')}_{r['schedule_row_id']:04d}",
        axis=1
    )

    return df


# ============================================================
# ENRIQUECIMENTO
# ============================================================

def enrich_with_schedule(base_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    enriched_parts = []

    # guarda as colunas originais da base
    base_columns = list(base_df.columns)

    for hardware in sorted(base_df["hardware"].dropna().unique()):
        base_hw = base_df[base_df["hardware"] == hardware].copy()
        sched_hw = schedule_df[schedule_df["hardware"] == hardware].copy()

        if len(sched_hw) == 0:
            # sem agenda correspondente para esse hardware
            base_hw["session_id_schedule"] = None
            base_hw["scheduled_start"] = pd.NaT
            base_hw["scheduled_end"] = pd.NaT
            base_hw["is_within_schedule"] = 0
            base_hw["seed_type_schedule"] = None
            base_hw["weight_kg_schedule"] = None
            base_hw["bags_qty_schedule"] = None
            base_hw["machine_number"] = None
            base_hw["state_client_schedule"] = None
            base_hw["city_schedule"] = None
            base_hw["execution_place_schedule"] = None
            base_hw["reference_latitude"] = None
            base_hw["reference_longitude"] = None
            base_hw["schedule_row_id"] = None
            enriched_parts.append(base_hw)
            continue

        # reset índice e cria chave técnica
        base_hw = base_hw.reset_index(drop=False).rename(columns={"index": "_base_index"})
        sched_hw = sched_hw.reset_index(drop=True)

        # evita conflito de nomes no merge:
        # remove a coluna hardware da agenda, pois já estamos iterando por hardware
        sched_hw = sched_hw.drop(columns=["hardware"], errors="ignore")

        base_hw["_tmp_key"] = 1
        sched_hw["_tmp_key"] = 1

        merged = base_hw.merge(sched_hw, on="_tmp_key", how="left").drop(columns="_tmp_key")

        # eventos dentro da janela
        matched = merged[
            (merged["event_time_dt"] >= merged["scheduled_start"]) &
            (merged["event_time_dt"] <= merged["scheduled_end"])
        ].copy()

        # se um evento casar com mais de uma janela, fica com a mais próxima do início
        if len(matched) > 0:
            matched["delta_to_start_sec"] = (
                matched["event_time_dt"] - matched["scheduled_start"]
            ).dt.total_seconds().abs()

            matched = (
                matched
                .sort_values(["_base_index", "delta_to_start_sec", "scheduled_start"])
                .drop_duplicates(subset=["_base_index"], keep="first")
            )

        # índices não casados
        if len(matched) > 0:
            unmatched_idx = set(base_hw["_base_index"]) - set(matched["_base_index"])
        else:
            unmatched_idx = set(base_hw["_base_index"])

        unmatched = base_hw[base_hw["_base_index"].isin(unmatched_idx)].copy()

        # renomeia colunas da agenda nos matched
        if len(matched) > 0:
            matched["is_within_schedule"] = 1
            matched = matched.rename(columns={
                "session_id_schedule": "session_id_schedule",
                "Semente": "seed_type_schedule",
                "Estado Cliente": "state_client_schedule",
                "Cidade Cliente": "city_schedule",
                "Local de execução": "execution_place_schedule",
            })

        # completa colunas dos unmatched
        if len(unmatched) > 0:
            unmatched["session_id_schedule"] = None
            unmatched["scheduled_start"] = pd.NaT
            unmatched["scheduled_end"] = pd.NaT
            unmatched["is_within_schedule"] = 0
            unmatched["seed_type_schedule"] = None
            unmatched["weight_kg_schedule"] = None
            unmatched["bags_qty_schedule"] = None
            unmatched["machine_number"] = None
            unmatched["state_client_schedule"] = None
            unmatched["city_schedule"] = None
            unmatched["execution_place_schedule"] = None
            unmatched["reference_latitude"] = None
            unmatched["reference_longitude"] = None
            unmatched["schedule_row_id"] = None

        extra_cols = [
            "session_id_schedule",
            "scheduled_start",
            "scheduled_end",
            "is_within_schedule",
            "seed_type_schedule",
            "weight_kg_schedule",
            "bags_qty_schedule",
            "machine_number",
            "state_client_schedule",
            "city_schedule",
            "execution_place_schedule",
            "reference_latitude",
            "reference_longitude",
            "schedule_row_id",
        ]

        cols_to_keep = base_columns + extra_cols

        if len(matched) > 0:
            # garante que todas as colunas existam
            for c in cols_to_keep:
                if c not in matched.columns:
                    matched[c] = None
            enriched_parts.append(matched[cols_to_keep])

        if len(unmatched) > 0:
            for c in cols_to_keep:
                if c not in unmatched.columns:
                    unmatched[c] = None
            enriched_parts.append(unmatched[cols_to_keep])

    enriched = pd.concat(enriched_parts, ignore_index=True)

    # session_id final
    if "session_id" in enriched.columns:
        enriched["session_id"] = enriched["session_id_schedule"].fillna(enriched["session_id"])
    else:
        enriched["session_id"] = enriched["session_id_schedule"]

    # campos finais preferenciais
    if "seed_type" in enriched.columns:
        enriched["seed_type_final"] = enriched["seed_type_schedule"].fillna(enriched["seed_type"])
    else:
        enriched["seed_type_final"] = enriched["seed_type_schedule"]

    if "weight_kg" in enriched.columns:
        enriched["weight_kg_final"] = enriched["weight_kg_schedule"].fillna(enriched["weight_kg"])
    else:
        enriched["weight_kg_final"] = enriched["weight_kg_schedule"]

    if "city" in enriched.columns:
        enriched["city_final"] = enriched["city_schedule"].fillna(enriched["city"])
    else:
        enriched["city_final"] = enriched["city_schedule"]

    if "state_client" in enriched.columns:
        enriched["state_client_final"] = enriched["state_client_schedule"].fillna(enriched["state_client"])
    else:
        enriched["state_client_final"] = enriched["state_client_schedule"]

    if "execution_place" in enriched.columns:
        enriched["execution_place_final"] = enriched["execution_place_schedule"].fillna(enriched["execution_place"])
    else:
        enriched["execution_place_final"] = enriched["execution_place_schedule"]

    enriched = enriched.sort_values(["event_time_dt", "hardware", "device_id"], na_position="last").reset_index(drop=True)

    return enriched

# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    print("Lendo base parsed...")
    base_df = load_base_parsed()
    print(f"Registros na base parsed: {len(base_df)}")

    print("Lendo agenda operacional...")
    schedule_df = load_schedule()
    print(f"Linhas de agenda: {len(schedule_df)}")

    print("Enriquecendo eventos com contexto de agenda...")
    enriched = enrich_with_schedule(base_df, schedule_df)

    enriched.to_parquet(OUTPUT_PARQUET, index=False)
    enriched.drop(columns=["event_time_dt"], errors="ignore").to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaídas geradas:")
    print(f"- {OUTPUT_PARQUET}")
    print(f"- {OUTPUT_CSV}")

    print("\nResumo:")
    print(f"Total de registros enriquecidos: {len(enriched)}")

    if "is_within_schedule" in enriched.columns:
        print("\nEventos dentro da agenda:")
        print(enriched["is_within_schedule"].value_counts(dropna=False).to_string())

    if "hardware" in enriched.columns:
        print("\nCobertura por hardware (% dentro da agenda):")
        cov = (
            enriched.groupby("hardware")["is_within_schedule"]
            .mean()
            .mul(100)
            .round(2)
        )
        print(cov.to_string())

    print("\nSessões por hardware:")
    if "session_id_schedule" in enriched.columns:
        sess = (
            enriched[enriched["session_id_schedule"].notna()]
            .groupby("hardware")["session_id_schedule"]
            .nunique()
        )
        print(sess.to_string())


if __name__ == "__main__":
    main()