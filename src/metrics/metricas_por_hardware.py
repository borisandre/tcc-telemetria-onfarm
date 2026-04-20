from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("data/metrics/metricas_por_sessao.parquet")

OUTPUT_DIR = Path("data/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "metricas_por_hardware.parquet"
OUTPUT_CSV = OUTPUT_DIR / "metricas_por_hardware.csv"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def safe_median(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def safe_p95(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(0.95)) if len(s) else None


# ============================================================
# LEITURA
# ============================================================

def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError("Arquivo de métricas por sessão não encontrado.")
    return pd.read_parquet(INPUT_PATH)


# ============================================================
# AGREGAÇÃO
# ============================================================

def build_hardware_metrics(df):

    results = []

    for hw, g in df.groupby("hardware"):

        row = {
            "hardware": hw,

            # volume
            "sessions_count": len(g),
            "messages_avg": safe_mean(g["messages_total"]),
            "messages_p95": safe_p95(g["messages_total"]),

            # cobertura
            "coverage_avg_pct": safe_mean(g["pct_within_schedule"]),
            "coverage_p95_pct": safe_p95(g["pct_within_schedule"]),

            # GPS
            "gps_valid_avg_pct": safe_mean(g["gps_valid_rate_pct"]),
            "gps_valid_p95_pct": safe_p95(g["gps_valid_rate_pct"]),
            "position_available_avg_pct": safe_mean(g["position_available_pct"]),

            # latência
            "delay_avg_ms": safe_mean(g["avg_delay_ms"]),
            "delay_p95_ms": safe_p95(g["avg_delay_ms"]),
            "delay_worst_ms": g["max_delay_ms"].max(),

            # sinal
            "rssi_avg_dbm": safe_mean(g["avg_rssi_dbm"]),
            "rssi_worst_dbm": g["min_rssi_dbm"].min(),

            # operação
            "operating_rate_avg_pct": safe_mean(g["machine_operating_rate_pct"]),
            "operating_msgs_avg": safe_mean(g["machine_operating_msg_count"]),

            # qualidade
            "parse_ok_avg_pct": safe_mean(g["parse_ok_rate_pct"]),
            "data_quality_avg_pct": safe_mean(g["data_quality_ok_rate_pct"]),
            "anomaly_rate_avg_pct": safe_mean(g["anomaly_rate_pct"]),

            # buffer / atraso
            "late_rate_avg_pct": safe_mean(g["late_rate_pct"]),
            "buffered_rate_avg_pct": safe_mean(g["buffered_rate_pct"]),
        }

        results.append(row)

    df_hw = pd.DataFrame(results)

    return df_hw


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    print("Lendo métricas por sessão...")
    df = load_data()

    print("Calculando métricas por hardware...")
    hw_df = build_hardware_metrics(df)

    hw_df.to_parquet(OUTPUT_PARQUET, index=False)
    hw_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaídas geradas:")
    print(OUTPUT_PARQUET)
    print(OUTPUT_CSV)

    print("\nResumo:")
    print(hw_df)


if __name__ == "__main__":
    main()