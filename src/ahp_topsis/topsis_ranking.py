from pathlib import Path
import numpy as np
import pandas as pd

from ahp_criterios import CRITERIOS
from ahp_model import calcular_ahp
from ahp_criterios import MATRIZ_COMPARACAO


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("data/metrics/metricas_por_hardware.csv")

OUTPUT_DIR = Path("data/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DECISION_MATRIX = OUTPUT_DIR / "matriz_decisao_topsis.csv"
OUTPUT_RANKING = OUTPUT_DIR / "ranking_topsis.csv"


# ============================================================
# NOTA DE COMPLEXIDADE (fixa, congelada)
# menor = melhor
# ============================================================

COMPLEXIDADE_MAP = {
    "FMC920": 2.0,
    "ZEFIRO": 3.0,
    "ABS": 5.0
}


# ============================================================
# LEITURA
# ============================================================

def load_metrics():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH.resolve()}")
    df = pd.read_csv(INPUT_PATH)
    return df


# ============================================================
# AJUDAS
# ============================================================

def minmax_benefit(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series([1.0] * len(s), index=s.index)
    return (s - min_v) / (max_v - min_v)


def minmax_cost(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series([1.0] * len(s), index=s.index)
    return (max_v - s) / (max_v - min_v)


# ============================================================
# CONSTRUÇÃO DOS 6 CRITÉRIOS FINAIS
# ============================================================

def build_decision_matrix(df_hw: pd.DataFrame) -> pd.DataFrame:
    df = df_hw.copy()

    df["hardware"] = df["hardware"].astype(str).str.upper().str.strip()

    # componentes normalizados
    messages_norm = minmax_benefit(df["messages_avg"])
    coverage_norm = minmax_benefit(df["coverage_avg_pct"])

    gps_valid_norm = minmax_benefit(df["gps_valid_avg_pct"])
    pos_avail_norm = minmax_benefit(df["position_available_avg_pct"])

    delay_norm_cost = minmax_cost(df["delay_avg_ms"])

    parse_norm = minmax_benefit(df["parse_ok_avg_pct"])
    dq_norm = minmax_benefit(df["data_quality_avg_pct"])
    anomaly_good_norm = minmax_cost(df["anomaly_rate_avg_pct"])

    operating_norm = minmax_benefit(df["operating_rate_avg_pct"])

    complexity_raw = df["hardware"].map(COMPLEXIDADE_MAP)
    complexity_norm_cost = minmax_cost(complexity_raw)

    # 6 critérios finais em escala 0..1
    df_decision = pd.DataFrame({
        "hardware": df["hardware"],

        "Conectividade": (coverage_norm + messages_norm) / 2.0,

        "Qualidade_Posicionamento": (gps_valid_norm + pos_avail_norm) / 2.0,

        # Para TOPSIS, podemos usar o valor bruto de custo ou o valor já transformado.
        # Aqui vamos usar o bruto e tratar como custo no algoritmo.
        "Latencia": pd.to_numeric(df["delay_avg_ms"], errors="coerce"),

        "Qualidade_Dados": (parse_norm + dq_norm + anomaly_good_norm) / 3.0,

        "Aderencia_Operacional": (operating_norm + coverage_norm) / 2.0,

        # Mantemos a nota bruta de custo
        "Complexidade_Integracao": complexity_raw,
    })

    return df_decision


# ============================================================
# TOPSIS
# ============================================================

def topsis(decision_df: pd.DataFrame, pesos: np.ndarray, benefit_criteria: list[bool]) -> pd.DataFrame:
    alternatives = decision_df.iloc[:, 0]
    matrix = decision_df.iloc[:, 1:].astype(float).to_numpy()

    # 1. normalização vetorial
    norm = np.sqrt((matrix ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    matrix_norm = matrix / norm

    # 2. ponderação
    matrix_weighted = matrix_norm * pesos

    # 3. solução ideal e anti-ideal
    ideal = np.zeros(matrix_weighted.shape[1])
    anti_ideal = np.zeros(matrix_weighted.shape[1])

    for j in range(matrix_weighted.shape[1]):
        if benefit_criteria[j]:
            ideal[j] = matrix_weighted[:, j].max()
            anti_ideal[j] = matrix_weighted[:, j].min()
        else:
            ideal[j] = matrix_weighted[:, j].min()
            anti_ideal[j] = matrix_weighted[:, j].max()

    # 4. distâncias
    dist_ideal = np.sqrt(((matrix_weighted - ideal) ** 2).sum(axis=1))
    dist_anti_ideal = np.sqrt(((matrix_weighted - anti_ideal) ** 2).sum(axis=1))

    # 5. coeficiente de proximidade
    closeness = dist_anti_ideal / (dist_ideal + dist_anti_ideal)

    result = pd.DataFrame({
        "hardware": alternatives,
        "distancia_ideal": dist_ideal,
        "distancia_anti_ideal": dist_anti_ideal,
        "score_topsis": closeness
    }).sort_values("score_topsis", ascending=False).reset_index(drop=True)

    result["ranking"] = result.index + 1
    return result


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    print("Lendo métricas por hardware...")
    df_hw = load_metrics()

    print("Construindo matriz de decisão TOPSIS...")
    decision_df = build_decision_matrix(df_hw)

    # pesos do AHP congelados a partir da matriz consistente
    pesos, CR, lambda_max, CI = calcular_ahp(MATRIZ_COMPARACAO)

    print("\nPesos AHP utilizados:")
    for c, p in zip(CRITERIOS, pesos):
        print(f"{c}: {p:.6f}")
    print(f"\nCR: {CR:.4f}")

    # benefício(True) / custo(False)
    benefit_criteria = [
        True,   # Conectividade
        True,   # Qualidade_Posicionamento
        False,  # Latencia
        True,   # Qualidade_Dados
        True,   # Aderencia_Operacional
        False,  # Complexidade_Integracao
    ]

    ranking_df = topsis(decision_df, pesos, benefit_criteria)

    decision_df.to_csv(OUTPUT_DECISION_MATRIX, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(OUTPUT_RANKING, index=False, encoding="utf-8-sig")

    print("\nMatriz de decisão:")
    print(decision_df)

    print("\nRanking TOPSIS:")
    print(ranking_df)

    print("\nArquivos gerados:")
    print(OUTPUT_DECISION_MATRIX)
    print(OUTPUT_RANKING)


if __name__ == "__main__":
    main()