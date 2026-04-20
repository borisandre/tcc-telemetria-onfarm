from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

INPUT_PATH = Path("data/normalized/base_enriquecida_agenda.parquet")

OUTPUT_DIR = Path("data/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "metricas_por_sessao.parquet"
OUTPUT_CSV = OUTPUT_DIR / "metricas_por_sessao.csv"


# ============================================================
# UTILITÁRIOS
# ============================================================

def safe_mean(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return float(s.mean())


def safe_quantile(series: pd.Series, q: float) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return float(s.quantile(q))


def safe_rate_percent(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return float(s.mean() * 100.0)


def count_not_null(series: pd.Series) -> int:
    return int(series.notna().sum())


# ============================================================
# LEITURA
# ============================================================

def load_base() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Base enriquecida não encontrada: {INPUT_PATH.resolve()}")

    df = pd.read_parquet(INPUT_PATH)

    if "event_time" in df.columns:
        df["event_time_dt"] = pd.to_datetime(df["event_time"], format="ISO8601", errors="coerce")
        if df["event_time_dt"].isna().any():
            df["event_time_dt"] = pd.to_datetime(df["event_time"], errors="coerce")

    if "scheduled_start" in df.columns:
        df["scheduled_start_dt"] = pd.to_datetime(df["scheduled_start"], errors="coerce")

    if "scheduled_end" in df.columns:
        df["scheduled_end_dt"] = pd.to_datetime(df["scheduled_end"], errors="coerce")

    return df


# ============================================================
# CÁLCULO DAS MÉTRICAS
# ============================================================

def build_session_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Considera apenas eventos associados a uma sessão real de agenda
    df_sessions = df[df["session_id_schedule"].notna()].copy()

    if len(df_sessions) == 0:
        return pd.DataFrame()

    group_cols = ["hardware", "session_id_schedule"]

    rows = []

    for (hardware, session_id), g in df_sessions.groupby(group_cols):
        g = g.sort_values("event_time_dt", na_position="last").reset_index(drop=True)

        scheduled_start = g["scheduled_start_dt"].dropna().min() if "scheduled_start_dt" in g.columns else pd.NaT
        scheduled_end = g["scheduled_end_dt"].dropna().max() if "scheduled_end_dt" in g.columns else pd.NaT

        session_duration_min = None
        if pd.notna(scheduled_start) and pd.notna(scheduled_end):
            session_duration_min = float((scheduled_end - scheduled_start).total_seconds() / 60.0)

        first_event_time = g["event_time_dt"].dropna().min() if "event_time_dt" in g.columns else pd.NaT
        last_event_time = g["event_time_dt"].dropna().max() if "event_time_dt" in g.columns else pd.NaT

        observed_span_min = None
        if pd.notna(first_event_time) and pd.notna(last_event_time):
            observed_span_min = float((last_event_time - first_event_time).total_seconds() / 60.0)

        messages_total = len(g)
        messages_within_schedule = int((pd.to_numeric(g["is_within_schedule"], errors="coerce").fillna(0) == 1).sum()) \
            if "is_within_schedule" in g.columns else messages_total

        messages_outside_schedule = messages_total - messages_within_schedule

        pct_within_schedule = (messages_within_schedule / messages_total * 100.0) if messages_total > 0 else None

        gps_valid_rate_pct = safe_rate_percent(g["gps_valid"]) if "gps_valid" in g.columns else None
        parse_ok_rate_pct = safe_rate_percent(g["parse_ok"]) if "parse_ok" in g.columns else None
        data_quality_ok_rate_pct = safe_rate_percent(g["data_quality_ok"]) if "data_quality_ok" in g.columns else None
        machine_operating_rate_pct = safe_rate_percent(g["machine_operating"]) if "machine_operating" in g.columns else None

        avg_rssi_dbm = safe_mean(g["rssi_dbm"]) if "rssi_dbm" in g.columns else None
        min_rssi_dbm = pd.to_numeric(g["rssi_dbm"], errors="coerce").min() if "rssi_dbm" in g.columns else None
        max_rssi_dbm = pd.to_numeric(g["rssi_dbm"], errors="coerce").max() if "rssi_dbm" in g.columns else None

        avg_delay_ms = safe_mean(g["delay_ms"]) if "delay_ms" in g.columns else None
        p95_delay_ms = safe_quantile(g["delay_ms"], 0.95) if "delay_ms" in g.columns else None
        max_delay_ms = pd.to_numeric(g["delay_ms"], errors="coerce").max() if "delay_ms" in g.columns else None

        avg_speed_kmh = safe_mean(g["speed_kmh"]) if "speed_kmh" in g.columns else None
        avg_speed_payload_kmh = safe_mean(g["speed_kmh_estimated"]) if "speed_kmh_estimated" in g.columns else None

        position_available_pct = None
        if "latitude" in g.columns and "longitude" in g.columns:
            position_available = g["latitude"].notna() & g["longitude"].notna()
            position_available_pct = float(position_available.mean() * 100.0)

        late_rate_pct = safe_rate_percent(g["is_late"]) if "is_late" in g.columns else None
        buffered_rate_pct = safe_rate_percent(g["is_buffered"]) if "is_buffered" in g.columns else None

        machine_operating_msg_count = int((pd.to_numeric(g["machine_operating"], errors="coerce").fillna(0) == 1).sum()) \
            if "machine_operating" in g.columns else 0

        machine_on_msg_count = int((pd.to_numeric(g["machine_on"], errors="coerce").fillna(0) == 1).sum()) \
            if "machine_on" in g.columns else 0

        anomaly_rate_pct = None
        if "data_quality_flags" in g.columns:
            anomaly_rate_pct = float(g["data_quality_flags"].notna().mean() * 100.0)

        row = {
            "session_id": session_id,
            "hardware": hardware,

            "scheduled_start": scheduled_start,
            "scheduled_end": scheduled_end,
            "session_duration_min": session_duration_min,

            "first_event_time": first_event_time,
            "last_event_time": last_event_time,
            "observed_span_min": observed_span_min,

            "messages_total": messages_total,
            "messages_within_schedule": messages_within_schedule,
            "messages_outside_schedule": messages_outside_schedule,
            "pct_within_schedule": pct_within_schedule,

            "gps_valid_rate_pct": gps_valid_rate_pct,
            "position_available_pct": position_available_pct,

            "avg_rssi_dbm": avg_rssi_dbm,
            "min_rssi_dbm": min_rssi_dbm,
            "max_rssi_dbm": max_rssi_dbm,

            "avg_delay_ms": avg_delay_ms,
            "p95_delay_ms": p95_delay_ms,
            "max_delay_ms": max_delay_ms,

            "avg_speed_kmh": avg_speed_kmh,
            "avg_speed_payload_kmh": avg_speed_payload_kmh,

            "parse_ok_rate_pct": parse_ok_rate_pct,
            "data_quality_ok_rate_pct": data_quality_ok_rate_pct,
            "anomaly_rate_pct": anomaly_rate_pct,

            "late_rate_pct": late_rate_pct,
            "buffered_rate_pct": buffered_rate_pct,

            "machine_on_msg_count": machine_on_msg_count,
            "machine_operating_msg_count": machine_operating_msg_count,
            "machine_operating_rate_pct": machine_operating_rate_pct,

            # contexto da sessão
            "seed_type": g["seed_type_final"].dropna().iloc[0] if "seed_type_final" in g.columns and g["seed_type_final"].notna().any() else None,
            "weight_kg": g["weight_kg_final"].dropna().iloc[0] if "weight_kg_final" in g.columns and g["weight_kg_final"].notna().any() else None,
            "bags_qty": g["bags_qty_schedule"].dropna().iloc[0] if "bags_qty_schedule" in g.columns and g["bags_qty_schedule"].notna().any() else None,
            "machine_number": g["machine_number"].dropna().iloc[0] if "machine_number" in g.columns and g["machine_number"].notna().any() else None,
            "city": g["city_final"].dropna().iloc[0] if "city_final" in g.columns and g["city_final"].notna().any() else None,
            "state_client": g["state_client_final"].dropna().iloc[0] if "state_client_final" in g.columns and g["state_client_final"].notna().any() else None,
            "execution_place": g["execution_place_final"].dropna().iloc[0] if "execution_place_final" in g.columns and g["execution_place_final"].notna().any() else None,
            "reference_latitude": g["reference_latitude"].dropna().iloc[0] if "reference_latitude" in g.columns and g["reference_latitude"].notna().any() else None,
            "reference_longitude": g["reference_longitude"].dropna().iloc[0] if "reference_longitude" in g.columns and g["reference_longitude"].notna().any() else None,
        }

        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # ordenação útil
    if "scheduled_start" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(["scheduled_start", "hardware"], na_position="last").reset_index(drop=True)

    return metrics_df


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    print("Lendo base enriquecida...")
    df = load_base()
    print(f"Registros na base enriquecida: {len(df)}")

    print("Calculando métricas por sessão...")
    metrics_df = build_session_metrics(df)

    if len(metrics_df) == 0:
        print("Nenhuma sessão encontrada para calcular métricas.")
        return

    metrics_df.to_parquet(OUTPUT_PARQUET, index=False)
    metrics_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaídas geradas:")
    print(f"- {OUTPUT_PARQUET}")
    print(f"- {OUTPUT_CSV}")

    print("\nResumo:")
    print(f"Total de sessões: {len(metrics_df)}")

    if "hardware" in metrics_df.columns:
        print("\nSessões por hardware:")
        print(metrics_df["hardware"].value_counts(dropna=False).to_string())

    print("\nMédia de mensagens por sessão por hardware:")
    print(
        metrics_df.groupby("hardware")["messages_total"]
        .mean()
        .round(2)
        .to_string()
    )

    print("\nMédia de GPS válido (%) por hardware:")
    if "gps_valid_rate_pct" in metrics_df.columns:
        print(
            metrics_df.groupby("hardware")["gps_valid_rate_pct"]
            .mean()
            .round(2)
            .to_string()
        )

    print("\nMédia de latência (ms) por hardware:")
    if "avg_delay_ms" in metrics_df.columns:
        print(
            metrics_df.groupby("hardware")["avg_delay_ms"]
            .mean()
            .round(2)
            .to_string()
        )


if __name__ == "__main__":
    main()