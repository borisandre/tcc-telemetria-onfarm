import numpy as np

# Índice aleatório de Saaty
RI_DICT = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
    5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49
}


def calcular_ahp(matriz):

    n = matriz.shape[0]

    # ============================================================
    # 1. AUTOVETOR (método do autovalor)
    # ============================================================

    autovalores, autovetores = np.linalg.eig(matriz)

    max_index = np.argmax(autovalores.real)
    max_autovalor = autovalores[max_index].real

    vetor_prioridade = autovetores[:, max_index].real

    # normalização
    pesos = vetor_prioridade / np.sum(vetor_prioridade)

    # ============================================================
    # 2. CONSISTÊNCIA
    # ============================================================

    CI = (max_autovalor - n) / (n - 1)

    RI = RI_DICT[n]

    if RI == 0:
        CR = 0
    else:
        CR = CI / RI

    return pesos.real, CR, max_autovalor, CI