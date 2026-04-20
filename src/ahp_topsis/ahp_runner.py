import pandas as pd

from ahp_criterios import CRITERIOS, MATRIZ_COMPARACAO
from ahp_model import calcular_ahp


def main():

    print("Executando AHP...\n")

    pesos, CR, lambda_max, CI = calcular_ahp(MATRIZ_COMPARACAO)

    df_result = pd.DataFrame({
        "criterio": CRITERIOS,
        "peso": pesos
    }).sort_values("peso", ascending=False)

    print(df_result)

    print("\n--- Consistência ---")
    print(f"Lambda max: {lambda_max:.4f}")
    print(f"CI: {CI:.4f}")
    print(f"CR: {CR:.4f}")

    if CR < 0.10:
        print("Matriz consistente")
    else:
        print("Matriz inconsistente")

    # export
    df_result.to_csv("data/metrics/pesos_ahp.csv", index=False)


if __name__ == "__main__":
    main()