import numpy as np
from math import sqrt

def config():
    param = np.genfromtxt('config.csv', delimiter=',', dtype=None,encoding='utf-8')
    return param

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normData(X):
    epsilon = 1e-10
    muX = np.mean(X, axis=0)
    sigmaX = np.std(X, axis=0) 
    u = (X - muX) / (sigmaX + epsilon)
    return sigmoid(u)


def entropyDataSet(X, param):
    xT = X.transpose()
    m = int(param[0])
    tau = int(param[1])
    c = int(param[2])

    N = xT.shape[1]
    M = N - (m - 1) * tau
    if M < 1:
        return 1e-10  # O manejar el error de manera apropiada

    embeddingsMap = []
    # Embedding y mapeo
    for carac in xT:
        # Construcción de embeddings
        embeddings = np.array([carac[i:i + (m - 1) * tau + 1:tau] for i in range(int(M))])
        # Normalización de embeddings
        min_val = np.min(embeddings)
        max_val = np.max(embeddings)
        embeddings_norm = (embeddings - min_val) / (max_val - min_val + 1e-10)
        # Discretización en c clases
        embeddingMap = np.floor(embeddings_norm * c).astype(int) + 1  # Valores entre 1 y c
        embeddingMap[embeddingMap > c] = c  # Asegurar que no exceda c
        embeddingsMap.append(embeddingMap)
    embeddingsMap = np.array(embeddingsMap, dtype=object)

    # Cálculo de patrones
    cElevado_M = c ** np.arange(m - 1, -1, -1)
    patrones = []
    for carac in embeddingsMap:
        patronesCarac = np.dot((carac - 1), cElevado_M) + 1
        patrones.append(patronesCarac)
    patrones = np.array(patrones, dtype=object)

    # Cálculo de frecuencias y probabilidades
    frecuencias = []
    for pat in patrones:
        _, frecuencia = np.unique(pat, return_counts=True)
        prob = frecuencia / frecuencia.sum()
        frecuencias.append(prob)
    frecuencias = np.array(frecuencias, dtype=object)

    # Cálculo de la entropía de dispersión normalizada
    DENorm = 0
    epsilon = 1e-10
    for prob in frecuencias:
        De = -np.sum(prob * np.log2(prob + epsilon))
        DENorm += De / np.log2(c ** m)

    return DENorm / xT.shape[0]


def entropyConditional(X, param):
    xT = X.flatten()  # Asegurar que xT es un arreglo unidimensional
    largo = len(xT)
    m = int(param[0])
    tau = int(param[1])
    c = int(param[2])

    N = largo
    M = N - (m - 1) * tau
    if M < 1:
        return 1e-10
    embeddings = np.array([xT[i:i + (m - 1) * tau + 1:tau] for i in range(int(M))])

    # Normalizar los embeddings
    min_val = np.min(embeddings)
    max_val = np.max(embeddings)
    embeddings_norm = (embeddings - min_val) / (max_val - min_val + 1e-10)

    # Discretizar los embeddings en c clases
    embeddingMap = np.floor(embeddings_norm * c).astype(int) + 1  # Valores entre 1 y c
    embeddingMap[embeddingMap > c] = c  # Asegurar que no exceda c

    # Calcular los patrones
    cElevado_M = c ** np.arange(m - 1, -1, -1)
    patrones = np.dot((embeddingMap - 1), cElevado_M) + 1

    # Obtener las frecuencias y probabilidades
    patronesUnicos, frecuencia = np.unique(patrones, return_counts=True)
    probabilidades = frecuencia / frecuencia.sum()

    # Calcular la entropía de dispersión y normalizar
    epsilon = 1e-10
    De = -np.sum(probabilidades * np.log2(probabilidades + epsilon))
    DENorm = De / np.log2(c ** m)

    return DENorm
    

