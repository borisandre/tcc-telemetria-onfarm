from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

BASE_METRICS_DIR = Path("data/metrics")
BASE_NORMALIZED_DIR = Path("data/normalized")

OUTPUT_DIR = Path("data/outputs/figures")
APPENDIX_DIR = OUTPUT_DIR / "appendix"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

RANKING_PATH = BASE_METRICS_DIR / "ranking_topsis.csv"
AHP_PATH = BASE_METRICS_DIR / "pesos_ahp.csv"
METRICS_HW_PATH = BASE_METRICS_DIR / "metricas_por_hardware.csv"
METRICS_SESSION_PATH = BASE_METRICS_DIR / "metricas_por_sessao.csv"
DECISION_PATH = BASE_METRICS_DIR / "matriz_decisao_topsis.csv"

ENRICHED_PARQUET_PATH = BASE_NORMALIZED_DIR / "base_enriquecida_agenda.parquet"
ENRICHED_CSV_PATH = BASE_NORMALIZED_DIR / "base_enriquecida_agenda.csv"

DPI = 300
FIG_W = 8
FIG_H = 5

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def save_fig(fig: plt.Figure, filename: str, appendix: bool = False) -> None:
    fig.tight_layout()
    target_dir = APPENDIX_DIR if appendix else OUTPUT_DIR
    fig.savefig(target_dir / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax, fmt: str = "{:.2f}", pad_fraction: float = 0.01) -> None:
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax != ymin else 1
    pad = yspan * pad_fraction

    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h):
            continue
        ax.text(
            p.get_x() + p.get_width() / 2,
            h + pad,
            fmt.format(h),
            ha="center",
            va="bottom"
        )


def reorder_hardware(df: pd.DataFrame, col: str = "hardware") -> pd.DataFrame:
    order = ["FMC920", "ZEFIRO", "ABS"]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


def load_csv_or_fail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")
    return pd.read_csv(path)


def standardize_hardware_names(df: pd.DataFrame, col: str = "hardware") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).str.upper().str.strip()
    return df


def load_enriched_base() -> pd.DataFrame:
    if ENRICHED_PARQUET_PATH.exists():
        df = pd.read_parquet(ENRICHED_PARQUET_PATH)
    elif ENRICHED_CSV_PATH.exists():
        df = pd.read_csv(ENRICHED_CSV_PATH)
    else:
        raise FileNotFoundError("Base enriquecida não encontrada em parquet ou csv.")

    if "hardware" in df.columns:
        df = standardize_hardware_names(df)

    if "event_time" in df.columns:
        df["event_time_dt"] = pd.to_datetime(df["event_time"], errors="coerce")
    elif "event_time_dt" in df.columns:
        df["event_time_dt"] = pd.to_datetime(df["event_time_dt"], errors="coerce")

    return df


# ============================================================
# LEITURA DOS DADOS
# ============================================================

def load_all():
    ranking = load_csv_or_fail(RANKING_PATH)
    ahp = load_csv_or_fail(AHP_PATH)
    metrics = load_csv_or_fail(METRICS_HW_PATH)
    metrics_session = load_csv_or_fail(METRICS_SESSION_PATH)
    decision = load_csv_or_fail(DECISION_PATH)
    enriched = load_enriched_base()

    ranking = standardize_hardware_names(ranking)
    metrics = standardize_hardware_names(metrics)
    metrics_session = standardize_hardware_names(metrics_session)
    decision = standardize_hardware_names(decision)

    ranking = reorder_hardware(ranking)
    metrics = reorder_hardware(metrics)
    metrics_session = reorder_hardware(metrics_session)
    decision = reorder_hardware(decision)

    return ranking, ahp, metrics, metrics_session, decision, enriched


# ============================================================
# GRÁFICOS PRINCIPAIS
# ============================================================

def plot_ranking_topsis(ranking: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(ranking["hardware"], ranking["score_topsis"])
    ax.set_title("Ranking final das alternativas pelo método TOPSIS")
    ax.set_xlabel("Hardware")
    ax.set_ylabel("Coeficiente de proximidade")
    ax.set_ylim(0, max(1.05, ranking["score_topsis"].max() * 1.12))
    add_bar_labels(ax, fmt="{:.3f}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, "fig_01_ranking_topsis.png")


def plot_ahp_weights(ahp: pd.DataFrame) -> None:
    ahp_plot = ahp.sort_values("peso", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(ahp_plot["criterio"], ahp_plot["peso"])
    ax.set_title("Pesos dos critérios obtidos pelo AHP")
    ax.set_xlabel("Peso")
    ax.set_ylabel("Critério")

    xmax = ahp_plot["peso"].max() * 1.18
    ax.set_xlim(0, xmax)

    for i, (_, row) in enumerate(ahp_plot.iterrows()):
        ax.text(
            row["peso"] + xmax * 0.01,
            i,
            f"{row['peso']:.3f}",
            va="center"
        )

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    save_fig(fig, "fig_02_pesos_ahp.png")


def plot_gps_quality(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(metrics["hardware"], metrics["gps_valid_avg_pct"])
    ax.set_title("Taxa média de GPS válido por hardware")
    ax.set_xlabel("Hardware")
    ax.set_ylabel("GPS válido (%)")
    ax.set_ylim(0, 105)
    add_bar_labels(ax, fmt="{:.1f}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, "fig_03_gps_valido.png")


def plot_latency(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(metrics["hardware"], metrics["delay_avg_ms"])
    ax.set_title("Latência média por hardware")
    ax.set_xlabel("Hardware")
    ax.set_ylabel("Latência média (ms)")
    ymax = metrics["delay_avg_ms"].max() * 1.18
    ax.set_ylim(0, ymax)
    add_bar_labels(ax, fmt="{:.0f}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, "fig_04_latencia_media.png")


def plot_schedule_coverage(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(metrics["hardware"], metrics["coverage_avg_pct"])
    ax.set_title("Cobertura média da agenda operacional")
    ax.set_xlabel("Hardware")
    ax.set_ylabel("Cobertura (%)")
    ax.set_ylim(0, 105)
    add_bar_labels(ax, fmt="{:.1f}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, "fig_05_cobertura_agenda.png")


def plot_data_quality(metrics: pd.DataFrame) -> None:
    cols = ["parse_ok_avg_pct", "data_quality_avg_pct", "anomaly_rate_avg_pct"]
    plot_df = metrics[["hardware"] + cols].copy()

    x = np.arange(len(plot_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, plot_df["parse_ok_avg_pct"], width, label="Parse OK (%)")
    ax.bar(x, plot_df["data_quality_avg_pct"], width, label="Qualidade dos dados (%)")
    ax.bar(x + width, plot_df["anomaly_rate_avg_pct"], width, label="Anomalias (%)")

    ax.set_title("Indicadores médios de qualidade dos dados")
    ax.set_xlabel("Hardware")
    ax.set_ylabel("Percentual (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["hardware"])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=1)

    save_fig(fig, "fig_06_qualidade_dados.png")


def plot_decision_matrix_grouped(decision: pd.DataFrame) -> None:
    criteria = [c for c in decision.columns if c != "hardware"]
    x = np.arange(len(criteria))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (_, row) in enumerate(decision.iterrows()):
        offset = (i - 1) * width
        ax.bar(x + offset, row[criteria].astype(float).values, width, label=row["hardware"])

    ax.set_title("Desempenho relativo por critério na matriz de decisão")
    ax.set_xlabel("Critério")
    ax.set_ylabel("Valor")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    save_fig(fig, "fig_07_matriz_decisao_criterios.png")


# ============================================================
# DIFERENCIAL
# ============================================================

def plot_radar(decision: pd.DataFrame) -> None:
    criteria = [c for c in decision.columns if c != "hardware"]
    values = decision[criteria].astype(float)

    num_vars = len(criteria)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for _, row in decision.iterrows():
        vals = row[criteria].astype(float).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=row["hardware"])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_title("Perfil comparativo das alternativas por critério", y=1.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)
    ax.set_ylim(0, max(1.0, values.to_numpy().max() * 1.05))
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=False)

    save_fig(fig, "fig_08_radar_criterios.png")


def plot_latency_boxplot(metrics_session: pd.DataFrame) -> None:
    plot_df = metrics_session[["hardware", "avg_delay_ms"]].copy()
    plot_df["avg_delay_ms"] = pd.to_numeric(plot_df["avg_delay_ms"], errors="coerce")
    plot_df = plot_df.dropna(subset=["avg_delay_ms"])

    order = ["FMC920", "ZEFIRO", "ABS"]
    data = [plot_df.loc[plot_df["hardware"] == hw, "avg_delay_ms"].values for hw in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=order, patch_artist=False, showfliers=True)
    ax.set_title("Distribuição da latência média por sessão")
    ax.set_xlabel("Hardware")
    ax.set_ylabel("Latência média por sessão (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    save_fig(fig, "fig_09_boxplot_latencia_sessao.png")


# ============================================================
# OPCIONAIS / APÊNDICE
# ============================================================

def plot_heatmap_correlation(metrics: pd.DataFrame) -> None:
    use_cols = [
        "coverage_avg_pct",
        "gps_valid_avg_pct",
        "position_available_avg_pct",
        "delay_avg_ms",
        "rssi_avg_dbm",
        "parse_ok_avg_pct",
        "data_quality_avg_pct",
        "anomaly_rate_avg_pct",
        "operating_rate_avg_pct",
    ]
    corr_df = metrics[use_cols].apply(pd.to_numeric, errors="coerce")
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, aspect="auto")

    ax.set_title("Correlação entre métricas agregadas por hardware")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "ap_heatmap_correlacao_metricas.png", appendix=True)


def plot_time_series_events(enriched: pd.DataFrame) -> None:
    if "event_time_dt" not in enriched.columns:
        return

    plot_df = enriched.dropna(subset=["event_time_dt"]).copy()
    if len(plot_df) == 0:
        return

    plot_df["event_date"] = plot_df["event_time_dt"].dt.floor("D")
    agg = (
        plot_df.groupby(["event_date", "hardware"])
        .size()
        .reset_index(name="events_count")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for hw in ["FMC920", "ZEFIRO", "ABS"]:
        sub = agg[agg["hardware"] == hw].sort_values("event_date")
        if len(sub) > 0:
            ax.plot(sub["event_date"], sub["events_count"], marker="o", linewidth=1.8, label=hw)

    ax.set_title("Série temporal do volume diário de eventos")
    ax.set_xlabel("Data")
    ax.set_ylabel("Quantidade de eventos")
    ax.grid(axis="both", linestyle="--", alpha=0.30)
    ax.legend(frameon=False)

    save_fig(fig, "ap_serie_temporal_eventos.png", appendix=True)


def plot_error_bars(metrics_session: pd.DataFrame) -> None:
    plot_df = metrics_session.copy()
    plot_df["avg_delay_ms"] = pd.to_numeric(plot_df["avg_delay_ms"], errors="coerce")
    plot_df["gps_valid_rate_pct"] = pd.to_numeric(plot_df["gps_valid_rate_pct"], errors="coerce")

    summary = (
        plot_df.groupby("hardware", observed=False)
        .agg(
            delay_mean=("avg_delay_ms", "mean"),
            delay_std=("avg_delay_ms", "std"),
            gps_mean=("gps_valid_rate_pct", "mean"),
            gps_std=("gps_valid_rate_pct", "std"),
        )
        .reset_index()
    )
    summary = reorder_hardware(summary)

    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        summary["delay_mean"],
        width,
        yerr=summary["delay_std"],
        capsize=4,
        label="Latência média por sessão (ms)"
    )
    ax.bar(
        x + width / 2,
        summary["gps_mean"],
        width,
        yerr=summary["gps_std"],
        capsize=4,
        label="GPS válido por sessão (%)"
    )

    ax.set_title("Comparação com barras de erro por hardware")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["hardware"])
    ax.set_xlabel("Hardware")
    ax.set_ylabel("Valor")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    save_fig(fig, "ap_barras_erro_metricas.png", appendix=True)


# ============================================================
# TABELAS DE APOIO
# ============================================================

def export_summary_tables(ranking: pd.DataFrame, ahp: pd.DataFrame, decision: pd.DataFrame) -> None:
    ranking.to_csv(OUTPUT_DIR / "tabela_ranking_topsis.csv", index=False, encoding="utf-8-sig")
    ahp.to_csv(OUTPUT_DIR / "tabela_pesos_ahp.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUTPUT_DIR / "tabela_matriz_decisao.csv", index=False, encoding="utf-8-sig")


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    ranking, ahp, metrics, metrics_session, decision, enriched = load_all()

    # principais
    plot_ranking_topsis(ranking)
    plot_ahp_weights(ahp)
    plot_gps_quality(metrics)
    plot_latency(metrics)
    plot_schedule_coverage(metrics)
    plot_data_quality(metrics)
    plot_decision_matrix_grouped(decision)

    # diferencial
    plot_radar(decision)
    plot_latency_boxplot(metrics_session)

    # apêndice
    plot_heatmap_correlation(metrics)
    plot_time_series_events(enriched)
    plot_error_bars(metrics_session)

    export_summary_tables(ranking, ahp, decision)

    print("Figuras geradas em:")
    print(OUTPUT_DIR.resolve())
    print("Apêndices em:")
    print(APPENDIX_DIR.resolve())


if __name__ == "__main__":
    main()